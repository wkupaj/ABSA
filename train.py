from __future__ import absolute_import, division, print_function

import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler

from code1.data_utils import AeProcessor, AscProcessor, create_dataset, convert_examples_to_features
from code1.model import XLMRMultiLabelClassification
from code1.train_utils import evaluate_model


def train():
    data_dir = 'data/aspect_on_review/extracted/'
    dropout = 0.2
    adam_epsilon = 1e-08
    eval_batch_size = 32
    gradient_accumulation_steps = 4
    learning_rate = 6e-05
    max_grad_norm = 1.0
    max_seq_length = 128
    num_train_epochs = 50
    output_dir = 'models/kpwr_n82_base/'
    pretrained_path = 'models/roberta_large_fairseq'
    seed = 42
    train_batch_size = 8
    warmup_proportion = 0.0
    weight_decay = 0.01

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=os.path.join(output_dir, "log.txt"))
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logger = logging.getLogger(__name__)

    train_batch_size = train_batch_size // gradient_accumulation_steps

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mode = 'ae'
    if mode == 'asc':
        processor = AscProcessor()
    elif mode == 'ae':
        processor = AeProcessor()

    label_list = processor.get_labels(data_dir)
    print(*label_list, sep="\n")

    num_labels = len(label_list)
    train_examples = processor.get_train_examples(data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs

    hidden_size = 768 if 'base' in pretrained_path else 1024

    device = "cpu"
    logger.info(device)

    model = XLMRMultiLabelClassification(pretrained_path=pretrained_path,
                                         n_labels=num_labels, hidden_size=hidden_size,
                                         dropout_p=dropout, device=device, mode=mode)

    model.to(device)
    no_decay = ['bias', 'final_layer_norm.weight']

    params = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in params if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = int(warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    train_features = convert_examples_to_features(train_examples, max_seq_length, model.encode_word)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    train_data = create_dataset(train_features)

    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    # getting validation samples
    val_examples = processor.get_dev_examples(data_dir)
    val_features = convert_examples_to_features(val_examples, max_seq_length, model.encode_word)

    val_data = create_dataset(val_features)

    best_val_f1 = 0.0
    criterion = nn.BCELoss()

    for epoch_no in range(1, num_train_epochs + 1):
        logger.info("Epoch %d" % epoch_no)
        tr_loss = 0

        model.train()
        steps = len(train_dataloader)
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, label_ids = batch

            model_result = model(input_ids)
            loss = criterion(model_result, label_ids.type(torch.float))
            print(model_result)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            tr_loss += loss.item()
            # if step % 100 == 99:
            #     logger.info('Step = %d/%d; Loss = %.4f' % (step+1, steps, tr_loss / (step+1)))
            logger.info('Step = %d/%d; Loss = %.4f' % (step + 1, steps, tr_loss / (step + 1)))

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

        logger.info("\nTesting on validation set...")
        f1, report = evaluate_model(model, val_data, eval_batch_size, device)
        if f1 > best_val_f1:
            best_val_f1 = f1
            logger.info("\nFound better f1=%.4f on validation set. Saving model\n" % f1)
            logger.info("%s\n" % report)

            torch.save(model.state_dict(), open(os.path.join(output_dir, 'model.pt'), 'wb'))

        else:
            logger.info("\nNo better F1 score: {}\n".format(f1))


if __name__ == "__main__":
    try:
        train()
    except ValueError as er:
        print("[ERROR] %s" % er)
