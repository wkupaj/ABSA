from torch.utils.data import SequentialSampler, DataLoader
import torch
from sklearn.metrics import f1_score


def evaluate_model(model, eval_dataset, batch_size, device):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

    model.eval()

    f1 = 0.0
    for input_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids)

        f1 = f1 + f1_score([i for i in torch.round(logits[0])], [i for i in label_ids[0]], average='macro')

    return f1, "report"
