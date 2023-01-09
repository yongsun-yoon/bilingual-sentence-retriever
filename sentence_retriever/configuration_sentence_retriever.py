from transformers import PretrainedConfig

class SentenceRetrieverConfig(PretrainedConfig):
    model_type = 'sentence_retriever'

    def __init__(
        self, 
        backbone_name: str = 'xlm-roberta-base',
        output_size: int = 768,
        init_backbone: bool = False,
        **kwargs
    ):
        self.backbone_name = backbone_name
        self.output_size = output_size
        self.init_backbone = init_backbone
        super().__init__(**kwargs)