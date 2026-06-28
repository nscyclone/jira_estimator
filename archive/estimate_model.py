import torch.nn as nn
from transformers import AutoModel

class EstimateRuBERT(nn.Module):
    def __init__(self, model_name='cointegrated/rubert-tiny2'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size

        # Regression head - 1 number for FTE prediction
        self.regression_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get embedding of [CLS]-token
        hidden_states = outputs[0]
        cls_output = hidden_states[:, 0, :]

        logits_estimate = self.regression_head(cls_output).squeeze(-1)

        return logits_estimate
