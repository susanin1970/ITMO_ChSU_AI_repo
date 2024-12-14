# python
from datetime import datetime

# 3rdparty
from fastapi import APIRouter, status

# project
from src.backend.neuralnets_serivce.schemas.service_output import HealthCheck
from src.backend.neuralnets_serivce.tools.logging_tools import get_logger()

logger = get_logger()
router = APIRouter(tags=["Info"])


@router.get(
    "/health",
    summary="Проверка сервиса на работоспособность",
    description="Проверяет сервис на работоспособность",
    response_description="HTTP Status Code 200",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def health_check() -> HealthCheck:
    """Метод проверки работоспособности сервиса

    Возвращает:
        * `HealthCheck`: объект HealthCheck
    """
    return HealthCheck(status_code=status.HTTP_200_OK, datetime=datetime.now())
