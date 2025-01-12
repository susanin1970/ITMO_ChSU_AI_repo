# 3rdparty
from pydantic import BaseModel, Field


# Дата класс для добавления данных в БД
class GlaucomaPydantic(BaseModel):
    """
    Датакласс, описывающий модель добавления данных в БД
    """

    id: int = Field(default=0)
    timestamp: str = Field(default="")
    width: int = Field(default=0)
    height: int = Field(default=0)
    status: bool = Field(default=False)
    verify: bool = Field(default=False)
    cdr_value: float = Field(default=0.0)
    rdar_value: float = Field(default=0.0)


# Фильтр по данным БД
class FilterData(BaseModel):
    """
    Датакласс, описывающий модель фильтра для запроса на поиск данных в БД
    """

    imageId: int
    timestamps: str
    interval_width_min: int
    interval_width_max: int
    interval_height_min: int
    interval_height_max: int
    glaucomStatus: bool
    hasVerifiication: bool
