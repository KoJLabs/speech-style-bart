# Speech Style Convert Model
말투를 변환해주는 모델 입니다. [gogamza/kobart-base-v2](https://huggingface.co/gogamza/kobart-base-v2) 모델을 [korean SmileStyle Dataset](https://github.com/smilegate-ai/korean_smile_style_dataset)을 활용해 fine-tuning 했습니다. 모델은 [huggingface](https://huggingface.co/KoJLabs/bart-speech-style-converter)에 업로드 되어있습니다.

## Model Details
* Model Description: Speech style converter model based on gogamza/kobart-base-v2
* Developed by: Juhwan, Lee and Jisu, Kim
* Model Type: Text-generation
* Language: Korean
* License: CC-BY-4.0

## Dataset
* [korean SmileStyle Dataset](https://github.com/smilegate-ai/korean_smile_style_dataset)
* Randomly split train/valid dataset (9:1)

## BLEU Score
* 25.35

## Package
```bash
poetry install
poetry shell
```

## Usage
아래와 같은 옵션을 통해 말투를 생성할 수 있습니다.

* formal: 문어체
* informal: 구어체
* android: 안드로이드
* azae: 아재
* chat: 채팅
* choding: 초등학생
* emoticon: 이모티콘
* enfp: enfp
* gentle: 신사
* halbae: 할아버지
* halmae: 할머니
* joongding: 중학생
* king: 왕
* naruto: 나루토
* seonbi: 선비
* sosim: 소심한
* translator: 번역기

```python

from transformers import pipeline

model = "KoJLabs/bart-speech-style-converter"
tokenizer = AutoTokenizer.from_pretrained(model)

nlg_pipeline = pipeline('text2text-generation',model=model, tokenizer=tokenizer)
styles = ["문어체", "구어체", "안드로이드", "아재", "채팅", "초등학생", "이모티콘", "enfp", "신사", "할아버지", "할머니", "중학생", "왕", "나루토", "선비", "소심한", "번역기"]

for style in styles:
    text = f"{style} 형식으로 변환:오늘은 닭볶음탕을 먹었다. 맛있었다."
    out = nlg_pipeline(text, max_length=100)
    print(style, out[0]['generated_text'])
```