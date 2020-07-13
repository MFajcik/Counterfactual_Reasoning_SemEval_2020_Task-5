import torch
import torch.nn as nn
from transformers import RobertaModel, BertModel, AlbertModel


class TransformerForECFR(nn.Module):
    def __init__(self, config, pad_token, sep_token):
        super().__init__()
        if config["model_type"].startswith("bert"):
            model_clz = BertModel
        elif config["model_type"].startswith("roberta"):
            model_clz = RobertaModel
        elif config["model_type"].startswith("albert"):
            model_clz = AlbertModel
        else:
            raise NotImplementedError(f"Unsupported model type {config['model_type']}.")
        self.transformer = model_clz.from_pretrained(config["model_type"], cache_dir=config["cache_dir"])
        if hasattr(self.transformer.pooler, 'dense'):
            hidden_size = self.transformer.pooler.dense.weight.shape[0]
        else:
            hidden_size = self.transformer.pooler.weight.shape[0]

        self.a_lin_outputs = torch.nn.Linear(hidden_size, 2)
        self.c_lin_outputs = torch.nn.Linear(hidden_size, 2)
        self.masking = True  # always mask
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.dropout = torch.nn.Dropout(p=config["dropout_rate"])

    def forward(self, batch):
        outputs = self.transformer(batch.sentence, attention_mask=batch.sentence_padding)
        # batch x sentence_len x d
        sequence_output = self.dropout(outputs[0])
        antecedent_log_s, antecedent_log_e = self.a_lin_outputs(sequence_output[:, 1:]).split(1, dim=-1)
        consequent_log_s, consequent_log_e = self.c_lin_outputs(sequence_output).split(1, dim=-1)
        antecedent_log_s.squeeze_(-1), antecedent_log_e.squeeze_(-1), \
        consequent_log_s.squeeze_(-1), consequent_log_e.squeeze_(-1)

        if self.masking:
            non_doc_token_mask = (batch.sentence[:, 1:] == self.pad_token) | (batch.sentence[:, 1:] == self.sep_token)
            antecedent_log_s.masked_fill_(non_doc_token_mask.bool(), float("-inf"))
            antecedent_log_e.masked_fill_(non_doc_token_mask.bool(), float("-inf"))

            non_doc_token_mask = (batch.sentence == self.pad_token) | (batch.sentence == self.sep_token)
            consequent_log_s.masked_fill_(non_doc_token_mask.bool(), float("-inf"))
            consequent_log_e.masked_fill_(non_doc_token_mask.bool(), float("-inf"))

        return antecedent_log_s, antecedent_log_e, consequent_log_s, consequent_log_e
