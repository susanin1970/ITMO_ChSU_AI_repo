# 3rdparty
from pydantic import BaseModel, Field

class CommonSettingsConfiguration(BaseModel):
    """Датакласс, описывающий общие настройки сервиса"""

    host: str = Field(default="localhost")
    """Хост сервиса"""
    port: int = Field(default=8080)
    """Порт сервиса"""