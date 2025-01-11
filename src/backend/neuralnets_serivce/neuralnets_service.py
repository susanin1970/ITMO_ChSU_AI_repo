# python
import argparse
import json
import logging

# 3rdparty
import pydantic
import uvicorn

# project
from src.backend.neuralnets_serivce.schemas.service_config import (
    NeuralNetsServiceConfig,
)
from src.backend.neuralnets_serivce.tools.logging_tools import configure_service_logger
from src.backend.neuralnets_serivce.tools.service_tools import Server


def get_service(
    service_config: NeuralNetsServiceConfig,
    num_workers: int = 0,
    reload: bool = False,
):
    """Функция для инициализации нейросетевого сервиса

    Параметры:
        * `service_config` (`PipeInferenceServiceConfig`): объект конфигурации сервиса
        * `num_workers` (`int`, `optional`): число обработчиков
        * `reload` (`bool`, `optional`): перезагружать ли сервис
    """
    config = uvicorn.Config(
        "src.backend.neuralnets_serivce.app:app",
        host=service_config.common_settings.host,
        port=service_config.common_settings.port,
        log_level=logging.INFO,
        workers=num_workers,
        reload=reload,
        use_colors=False,
    )
    configure_service_logger(logging.INFO)
    return Server(config)


def main() -> None:
    """Точка входа в сервис"""
    service_config_path = "src/backend/neuralnets_serivce/configs/service_config.json"
    with open(service_config_path, "r") as json_service_config:
        service_config_dict = json.load(json_service_config)

    service_config_adapter: pydantic.TypeAdapter = pydantic.TypeAdapter(
        NeuralNetsServiceConfig
    )

    service_config_python = service_config_adapter.validate_python(service_config_dict)

    async_logger = logging.getLogger("asyncio")
    async_logger.propagate = False
    async_logger.handlers.clear()

    get_service(service_config_python).run()


if __name__ == "__main__":
    main()
