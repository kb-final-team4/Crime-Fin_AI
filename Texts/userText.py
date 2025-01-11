from pydantic import BaseModel
from typing import Union


class UserText(BaseModel):
    id: int
    text: str
    phishing: Union[int, None] = None  # phishing은 int 값일수도, None일수도 있음
    probability: Union[float, None] = None
