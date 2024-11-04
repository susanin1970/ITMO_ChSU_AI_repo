# 3rdparty
from pydantic import BaseModel, Field


class ClassificationConfig(BaseModel):
    """
    Датакласс, описывающий конфигурацию модели для классификации
    """

    name: str = Field(default="EfficientNet-B0")
    """Название модели нейронной сети для классификации"""
    path_to_weights: str = Field(default="")
    """Путь к весам модели нейронной сети для классификации"""
    use_cuda: bool = Field(default=False)
    """Использовать ли GPU"""


class CommonSettingsConfiguration(BaseModel):
    """Датакласс, описывающий общие настройки сервиса"""

    host: str = Field(default="localhost")
    """Хост сервиса"""
    port: int = Field(default=8000)
    """Порт сервиса"""


class NeuralNetsServiceConfig(BaseModel):
    """Датакласс, описывающий конфигурацию нейросетевого сервиса"""

    classification_config: ClassificationConfig = Field(default=ClassificationConfig())
    """Конфигурация модели для классификации"""

    common_settings: CommonSettingsConfiguration = Field(
        default=CommonSettingsConfiguration()
    )
    """Общие настройки сервиса"""
