import logging
from torchtext import data
from lib.models.utils import create_model, get_cosine_schedule_with_warmup
from lib.pate.utils import build_optimizer
import torch
from tqdm import tqdm
import numpy as np

def train_imdb(
    labeled_dataset,
    unlabeled_dataset,
    test_dataset,
    fixmatch_config,
    learning_config,
    device,
    n_classes,
    writer,
    writer_tag,
    checkpoint_path,
    text_field,
    label_field
):
    logging.info("Launching FixMatch training for IMDB")

    batch_size = learning_config.batch_size

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (labeled_dataset, unlabeled_dataset, test_dataset),
        batch_size=batch_size,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)

    model = create_model(fixmatch_config.model, num_classes=n_classes)
    model = model.to(device)
    
    pretrained_embeddings = text_field.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = text_field.vocab.stoi[text_field.unk_token]
    PAD_IDX = text_field.vocab.stoi[text_field.pad_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(fixmatch_config.model.embedding_dim)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(fixmatch_config.model.embedding_dim)


    no_decay = ["bias", "bn"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": learning_config.optim.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = build_optimizer(grouped_parameters, learning_config.optim)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        fixmatch_config.warmup,
        num_training_steps=len(train_iterator) * learning_config.epochs,
    )

    ema_model = None
    if fixmatch_config.use_ema:
        from lib.models.ema import ModelEMA

        ema_model = ModelEMA(device, model, fixmatch_config.ema_decay)

    model.zero_grad()

    best_acc = 0
    test_accs = []
    
    criterion = torch.nn.BCEWithLogitsLoss()


    logging.info("starting training")

    for epoch in tqdm(range(learning_config.epochs)):
        model.train()
        logging.info(f"Epoch {epoch}. Memory {torch.cuda.memory_allocated(device)}")
        for batch in train_iterator:
            optimizer.zero_grad()
            
            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            if ema_model is not None:
                ema_model.update(model)

        if ema_model is not None:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc = test_imdb(device, test_iterator, test_model)

        writer.add_scalar(f"train_{writer_tag}/1.train_loss", loss.item(), epoch)
        writer.add_scalar(f"test_{writer_tag}/1.test_acc", test_acc, epoch)
        writer.add_scalar(f"test_{writer_tag}/2.test_loss", test_loss, epoch)

        if test_acc > best_acc:
            best_acc = test_acc

            if checkpoint_path:
                logging.info(f"Saving checkpoint to {checkpoint_path}")
                torch.save(model.state_dict(), checkpoint_path)

                if ema_model:
                    torch.save(test_model.state_dict(), checkpoint_path + "_ema")

        test_accs.append(test_acc)
        logging.info("Best top-1 acc: {:.2f}".format(best_acc))
        logging.info("Mean top-1 acc: {:.2f}\n".format(np.mean(test_accs[-20:])))

    final_model = ema_model.ema if ema_model else model
    return final_model, best_acc, test_loss

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def test_imdb(device, test_loader, model):
    epoch_loss = 0
    epoch_acc = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in test_loader:
            text, text_lengths = batch.text
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, batch.label)
            
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(test_loader), epoch_acc / len(test_loader)
