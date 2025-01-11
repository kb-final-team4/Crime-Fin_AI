from fastapi import FastAPI
from typing import List
from config import Config
from classifier import classify
from Texts.userText import UserText
from preprocessor import preprocess

app = FastAPI()


@app.post("/predict")
async def classify_user_text(user_text_list: List[UserText]):
    """
        사용자 입력 텍스트의 보이스피싱 여부를 predict
    """
    user_text_list = sorted(user_text_list, key=lambda user_text: user_text.id)
    lines_for_predict = []
    for user_text in user_text_list:
        lines_for_predict.append(preprocess(user_text.text))
    config = Config(model_fn="model.pt", gpu_id=-1, batch_size=16,
                    lines=lines_for_predict, pretrained_model_name='skt/kobert-base-v1')
    classified_lines = classify(config)
    classification_result = []
    for i, classified_line in enumerate(classified_lines):
        user_text = UserText(
            id=user_text_list[i].id,
            text=classified_line[2]
        )
        user_text.probability = classified_line[0]
        user_text.phishing = classified_line[1]
        classification_result.append(user_text)
    return classification_result
