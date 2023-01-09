import torch.nn as nn
from transformers import PreTrainedModel, AutoConfig, AutoModel

from .configuration_sentence_embedder import SentenceEmbedderConfig

class SentenceEmbedderModel(PreTrainedModel):
    config_class = SentenceEmbedderConfig

    def __init__(self, config):
        super().__init__(config)
        if config.init_backbone:
            self.backbone = AutoModel.from_pretrained(config.backbone_name)
        else:
            backbone_config = AutoConfig.from_pretrained(config.backbone_name)
            self.backbone = AutoModel.from_config(backbone_config)
        self.projection = nn.Linear(self.backbone.config.hidden_size, config.output_size)


    def forward(self, input_ids, attention_mask, head=None):
        outputs = self.backbone(input_ids, attention_mask)
        last_hidden_state = self.projection(outputs.last_hidden_state)
        outputs.last_hidden_state = last_hidden_state
        return outputs