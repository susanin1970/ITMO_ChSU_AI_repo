# python
import logging
from typing import Literal, TypeAlias

LogLevelTypes: TypeAlias = (
    Literal[
        "CRITICAL",
        "FATAL",
        "ERROR",
        "WARN",
        "WARNING",
        "INFO",
        "DEBUG",
        "NOTSET",
    ]
    | int
)


def get_logger() -> logging.Logger:
    """Функция для получения объекта логгера FastAPI

    Возвращает:
        * `logging.Logger`: объект логгера FastAPI
    """
    return logging.getLogger("fastapi")


def configure_service_logger(level: LogLevelTypes) -> None:
    """Функция для выполнения конфигурирования процедуры логирования работы сервиса

    Параметры:
        - `level` (`LogLevelTypes`):
    """
    logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(logging_format)

    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers.clear()
    uvicorn_logger.propagate = False

    multipart_logger = logging.getLogger("multipart")
    multipart_logger.propagate = False

    pil_logger = logging.getLogger("PIL")
    pil_logger.propagate = False

    logger = get_logger()
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    uvicorn_logger.addHandler(console_handler)

    logger.setLevel(level)

    # uvicorn_acess_logger = logging.getLogger("uvicorn.access")
    # uvicorn_error_logger = logging.getLogger("uvicorn.error")
