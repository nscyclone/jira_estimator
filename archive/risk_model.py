import torch.nn as nn
from transformers import AutoModel

class RiskRuBERT(nn.Module):
    def __init__(self, model_name='cointegrated/rubert-tiny2', num_risk_classes=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        hidden_size = self.bert.config.hidden_size

        # Classification head - `num_risk_classes` numbers for risk class prediction
        self.classification_head = nn.Linear(hidden_size, num_risk_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get embedding of [CLS]-token
        hidden_states = outputs[0]
        cls_output = hidden_states[:, 0, :]

        logits_risk = self.classification_head(cls_output)

        return logits_risk
