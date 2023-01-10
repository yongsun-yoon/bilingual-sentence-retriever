# Multilingual Sentence Embedder

## Introduction
* 사전학습된 영어 문장 임베딩 모델을 knowledge distillation 하여 다국어 문장 임베딩 모델을 학습합니다. [1]
* 성능 개선을 위해 word-level loss를 추가하였습니다.
* 학습 데이터로 한국어-영어 병렬 말뭉치 데이터를 사용했습니다. [url](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126)


## Train
* 먼저 cfg.yaml에서 data_dir과 ckpt_dir을 수정해주세요.
* 데이터는 csv 형식을 사용했으며, 한국어는 'ko', 영어는 'en' 칼럼으로 구성했습니다.
```
# setup
pip install -q hydra-core transformers datasets wandb ctranslate2 sentencepiece
git clone https://github.com/yongsun-yoon/multilingual-sentence-embedder.git
cd multilingual-sentence-embedder

# bilingual corpus를 사용한 학습
python train_bilingual_corpus.py

# monolingual corpus를 번역하여 학습
ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-200-distilled-600M --quantization int8 --force
python train_multilingual_translated.py
```

## Usage
```python
from transformers import AutoTokenizer, AutoModel

model_name = 'yongsun-yoon/bilingual-sentence-embedder-mMiniLMv2-L6-H384'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
```


## Reference
[1] Reimers, N., & Gurevych, I. (2020). Making monolingual sentence embeddings multilingual using knowledge distillation. arXiv preprint arXiv:2004.09813.
