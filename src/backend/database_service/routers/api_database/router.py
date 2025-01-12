# python
from datetime import datetime

# 3rdparty
from fastapi import APIRouter, status

# project
from src.backend.neuralnets_serivce.schemas.service_output import HealthCheck
from src.backend.neuralnets_serivce.tools.logging_tools import get_logger

from src.backend.database_service.database.database_core import database_sqllite
from src.backend.database_service.entities.entity_glaucoma import GlaucomaEntity
from src.backend.database_service.schemas.database_service_schemas import (
    GlaucomaPydantic,
)
from src.backend.database_service.schemas.database_service_schemas import FilterData

logger = get_logger()
router = APIRouter(tags=["Info"])
sessions = database_sqllite()


@router.put("/database/")
def update_processing_result_data_to_bd(imageId: int):
    session = sessions.get_session()

    query = ""
    if imageId != None:
        query = session.query(GlaucomaEntity).filter(GlaucomaEntity.id == imageId)

    for data in query:
        data.verify = True

    session.commit()


@router.get(
    "/database", response_model=GlaucomaPydantic, status_code=status.HTTP_200_OK
)
def fetch_processing_result_data_from_db_by_id(imageId: int) -> GlaucomaPydantic:
    session = sessions.get_session()

    query = ""
    if imageId != None:
        query = session.query(GlaucomaEntity).filter(GlaucomaEntity.id == imageId)

    for data in query:
        return data


@router.post("/database/filter", response_model=list, status_code=status.HTTP_200_OK)
def fetch_processing_result_data_from_db(filter: FilterData) -> list:
    session = sessions.get_session()

    query = ""

    if filter.imageId != None:
        query = session.query(GlaucomaEntity).filter(
            GlaucomaEntity.id == filter.imageId
        )
    if filter.timestamps != None:
        query = session.query(GlaucomaEntity).filter(
            GlaucomaEntity.timestamps == filter.timestamps
        )
    if filter.interval_width_min != None:
        query = session.query(GlaucomaEntity).filter(
            GlaucomaEntity.width >= filter.interval_width_min
        )
    if filter.interval_height_min != None:
        query = session.query(GlaucomaEntity).filter(
            GlaucomaEntity.width >= filter.interval_height_min
        )
    if filter.interval_width_max != None:
        query = session.query(GlaucomaEntity).filter(
            GlaucomaEntity.width <= filter.interval_width_max
        )
    if filter.interval_height_max != None:
        query = session.query(GlaucomaEntity).filter(
            GlaucomaEntity.width <= filter.interval_height_max
        )
    if filter.glaucomStatus != None:
        query = session.query(GlaucomaEntity).filter(
            GlaucomaEntity.status == filter.glaucomStatus
        )
    if filter.hasVerifiication != None:
        query = session.query(GlaucomaEntity).filter(
            GlaucomaEntity.verify == filter.hasVerifiication
        )

    list = []
    for data in query:
        list.append(data)
    return list


@router.delete("/database")
def delete_processing_result_data_from_db(imageId: int):
    session = sessions.get_session()

    query = ""
    if imageId != None:
        query = session.query(GlaucomaEntity).filter(GlaucomaEntity.id == imageId)
        session.delete(query)

    session.commit()


@router.post("/database/")
def add_processing_result_data_to_db(record: GlaucomaPydantic):
    try:
        glaucoma = GlaucomaEntity(
            timestamp=record.timestamp,
            width=record.width,
            height=record.height,
            status=record.status,
            verify=record.verify,
            cdr_value=record.cdr_value,
            rdar_value=record.rdar_value,
        )

        session = sessions.get_session()

        session.add(glaucoma)
        session.commit()
    except Exception as Exc:
        print("Error: " + str(Exc))


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
