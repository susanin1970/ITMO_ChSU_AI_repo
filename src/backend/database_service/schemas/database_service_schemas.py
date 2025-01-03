# 3rdparty
from pydantic import BaseModel, Field

# Дата класс для добавления данных в БД
class GlaucomaPydantic(BaseModel):
    """
    Датакласс, описывающий модель добавления данных в БД
    """
    id : int = None
    timestamps : int
    width : int 
    height : int
    status : bool
    verify : bool
    imgCache : str = None

# Фильтр по данным БД
class FilterData(BaseModel):
    """
    Датакласс, описывающий модель фильтра для запроса на поиск данных в БД
    """
    imageId : int
    timestamps : int
    interval_width_min : int
    interval_width_max : int
    interval_height_min : int
    interval_height_max : int
    glaucomStatus : bool
    hasVerifiication : bool
