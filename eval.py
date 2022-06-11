from __future__ import absolute_import, division, print_function

import logging
import os
import random
import sys

import numpy as np
import torch

from core.data_utils import AeProcessor, create_dataset, convert_examples_to_features
from core.model import XLMRMultiLabelClassification
from core.train_utils import evaluate_model


def eval():
    data_dir = 'data/aspect_on_review/extracted/'
    dropout = 0.2
    adam_epsilon = 1e-08
    eval_batch_size = 32
    gradient_accumulation_steps = 4
    learning_rate = 6e-05
    max_grad_norm = 1.0
    max_seq_length = 128
    num_train_epochs = 50
    output_dir = 'models/absa/'
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    processor = AeProcessor()
    label_list = processor.get_labels(data_dir)
    print(*label_list, sep="\n")

    num_labels = len(label_list)


    hidden_size = 768 if 'base' in pretrained_path else 1024

    # device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    device = "cpu"
    logger.info(device)

    model = XLMRMultiLabelClassification(pretrained_path=pretrained_path,
                                         n_labels=num_labels, hidden_size=hidden_size,
                                         dropout_p=dropout, device=device)

    model.to(device)

    eval_examples = processor.get_test_examples(data_dir)
    eval_features = convert_examples_to_features(eval_examples, max_seq_length, model.encode_word)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", eval_batch_size)

    eval_data = create_dataset(eval_features)
    f1_score, report = evaluate_model(model, eval_data, eval_batch_size, device)

    logger.info("\n%s", report)
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Writing results to file *****")
        writer.write(report)
        logger.info("Done.")


if __name__ == "__main__":
    try:
        eval()
    except ValueError as er:
        print("[ERROR] %s" % er)
