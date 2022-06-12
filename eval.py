from __future__ import absolute_import, division, print_function

import os
import random

import numpy as np
import torch
from sklearn.metrics import classification_report

from pytorch_transformers import AdamW, WarmupLinearSchedule
from core.data_utils import AeProcessor, create_dataset, convert_examples_to_features, AscProcessor
from core.model import XLMRMultiLabelClassification


def eval():
    data_dir = 'data/aspect_on_sentence/extracted/'
    dropout = 0.2
    eval_batch_size = 32
    max_seq_length = 128
    model_dir = 'models/ae_absa/'
    pretrained_path = 'models/roberta_base_fairseq'
    seed = 42

    data_dir = 'data/aspect_on_sentence/extracted/'
    dropout = 0.2
    adam_epsilon = 1e-08
    eval_batch_size = 32
    gradient_accumulation_steps = 4
    learning_rate = 6e-05
    max_grad_norm = 1.0
    max_seq_length = 128
    num_train_epochs = 3
    pretrained_path = 'models/roberta_large_fairseq'
    seed = 42
    train_batch_size = 32
    warmup_proportion = 0.0
    weight_decay = 0.01

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

    device = "cuda:0"
    print(device)

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
    model.to(device)

    state_dict = torch.load(open(os.path.join(model_dir, 'model.pt'), 'rb'), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print("Loaded saved model")

    eval_examples = processor.get_test_examples(data_dir)
    eval_features = convert_examples_to_features(eval_examples, max_seq_length, model.encode_word)

    print("***** Running evaluation *****")
    print("  Num examples = %d", len(eval_examples))
    print("  Batch size = %d", eval_batch_size)

    eval_data = create_dataset(eval_features)
    y_pred_list = []
    y_test = []
    with torch.no_grad():
        model.eval()
        for X_eval, y_eval in eval_data:
            X_eval, y_eval = X_eval.to(device), y_eval.to(device)

            y_eval_pred = model(X_eval)
            y_test.append(y_eval.cpu().numpy())
            y_pred_list.append(y_eval_pred.cpu().numpy())

    print(classification_report(y_test, y_pred_list))


if __name__ == "__main__":
    try:
        eval()
    except ValueError as er:
        print("[ERROR] %s" % er)
