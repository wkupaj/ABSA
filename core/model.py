import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.roberta import XLMRModel


class XLMRMultiLabelClassification(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size, dropout_p=0.2, label_ignore_idx=0,
                 head_init_range=0.04, device='cuda', mode='ae'):
        super().__init__()

        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.classification_head = nn.Linear(hidden_size, n_labels)
        if mode == 'ae':
            self.act_fn = nn.Sigmoid()
        else:
            self.act_fn = nn.Softmax()

        self.n_labels = n_labels
        self.label_ignore_idx = label_ignore_idx
        self.device = device

    def forward(self, inputs_ids):
        transformer_out, _ = self.model(inputs_ids, features_only=True)
        x = transformer_out[:, 0]
        x = F.relu(self.dense(x))
        # x = self.dropout(pooler)
        # x = self.dense(F.relu(x))
        x = self.dropout(x)
        x = self.classification_head(x)
        return self.act_fn(x)

    def encode_word(self, s):
        tensor_ids = self.xlmr.encode(s)
        return tensor_ids.cpu().numpy().tolist()[1:-1]
