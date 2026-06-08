import torch.nn as nn
from transformers import AutoModel

class MultiTaskRuBERT(nn.Module):
    def __init__(self, model_name='cointegrated/rubert-tiny2', num_risk_classes=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size

        # Regression head - 1 number for FTE prediction
        self.regression_head = nn.Linear(hidden_size, 1)

        # Classification head - `num_risk_classes` numbers for risk class prediction
        self.classification_head = nn.Linear(hidden_size, num_risk_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get embedding of [CLS]-token
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits_estimate = self.regression_head(cls_output).squeeze(-1)
        logits_risk = self.classification_head(cls_output)

        return logits_estimate, logits_risk
