# python
from datetime import datetime
from enum import Enum

# 3rdparty
from pydantic import BaseModel, Field


class HealthCheck(BaseModel):
    """Датакласс для описания статуса работы нейросетевого сервиса"""

    status_code: int
    """Код статуса работы нейросетевого сервиса"""
    datetime: datetime
    """Отсечка даты и времени"""


class GlaucomaSignsStatus(Enum):
    """Перечисление, описывающее возможные классы изображений глазного дна в зависимости от наличия или отсутствия признаков глаукомы"""

    GLAUCOMA = "есть признаки глаукомы"
    NO_GLAUCOMA = "нет признаков глаукомы"


class NeuralNetsServiceOutput(BaseModel):
    """Датакласс, описывающий выход нейросетевого сервиса"""

    predicted_class: str = Field(
        default=GlaucomaSignsStatus.NO_GLAUCOMA,
    )
    """Значение предсказанного классификатором класса в зависимости от наличия/отсутствия признаков глаукомы """
    predicted_class_confidence: float = Field(default=0.0)
    """Значение уверенности предсказанного классификатором класса"""
    cdr_value: float = Field(default=0.0)
    """Значение CDR"""
    rdar_value: float = Field(default=0.0)
    """Значение RDAR"""
