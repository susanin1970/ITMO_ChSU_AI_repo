# python
import logging


def get_logger() -> logging.Logger:
    """Функция для получения объекта логгера FastAPI

    Возвращает:
        * `logging.Logger`: объект логгера FastAPI
    """
    return logging.getLogger("fastapi")
