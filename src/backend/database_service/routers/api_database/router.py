# python
from datetime import datetime
from typing import List

# 3rdparty
from fastapi import APIRouter, HTTPException, status

# project
from src.backend.neuralnets_serivce.schemas.service_output import HealthCheck
from src.backend.neuralnets_serivce.tools.logging_tools import get_logger

from src.backend.database_service.database.database_core import GlaucomaSQLiteDatabase
from src.backend.database_service.entities.entity_glaucoma import GlaucomaEntity
from src.backend.database_service.schemas.database_service_schemas import (
    GlaucomaPydantic,
)
from src.backend.database_service.schemas.database_service_schemas import FilterData

logger = get_logger()
router = APIRouter(tags=["Info"])
sessions = GlaucomaSQLiteDatabase()


@router.put("/database/update_data")
def update_processing_result_data_to_bd(imageId: int):
    session = sessions.get_session()

    query = ""
    if imageId != None:
        query = session.query(GlaucomaEntity).filter(GlaucomaEntity.id == imageId)

    for data in query:
        data.verify = True

    session.commit()


@router.get(
    "/database/fetch_data_by_id",
    response_model=GlaucomaPydantic,
    status_code=status.HTTP_200_OK,
)
def fetch_processing_result_data_from_db_by_id(imageId: int) -> GlaucomaPydantic:
    session = sessions.get_session()

    query = ""
    if imageId != None:
        query = session.query(GlaucomaEntity).filter(GlaucomaEntity.id == imageId)

    for data in query:
        return data


@router.post(
    "/database/fetch_all_data",
    summary="Чтение всех данных из базы",
    description="Читает все данные из базы",
    response_description="HTTP Status Code 200",
    response_model=list,
    status_code=status.HTTP_200_OK,
)
def fetch_all_processing_results_data_from_db() -> list:
    try:
        session = sessions.get_session()
        query = session.query(GlaucomaEntity)
        all_processing_results = query.all()
        all_processing_results_list = []
        for result in all_processing_results:
            all_processing_results_list.append(
                GlaucomaPydantic(
                    id=result.id,
                    timestamp=result.timestamp,
                    width=result.width,
                    height=result.height,
                    status=result.status,
                    verify=result.verify,
                    cdr_value=result.cdr_value,
                    rdar_value=result.rdar_value,
                )
            )
        return all_processing_results_list
    except Exception as ex:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(ex)) from ex


@router.post(
    "/database/filter_data", response_model=list, status_code=status.HTTP_200_OK
)
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


@router.delete("/database/delete_data")
def delete_processing_result_data_from_db(imageId: int):
    session = sessions.get_session()

    query = ""
    if imageId != None:
        query = session.query(GlaucomaEntity).filter(GlaucomaEntity.id == imageId)
        session.delete(query)

    session.commit()


@router.post(
    "/database/add_data",
    summary="Добавление данных в базу",
    description="Добавляет данные в базу",
    response_description="HTTP Status Code 200",
    status_code=status.HTTP_200_OK,
)
def add_processing_result_data_to_db(record: GlaucomaPydantic) -> None:
    try:
        session = sessions.get_session()

        glaucoma = GlaucomaEntity(
            timestamp=record.timestamp,
            width=record.width,
            height=record.height,
            status=record.status,
            verify=record.verify,
            cdr_value=record.cdr_value,
            rdar_value=record.rdar_value,
        )

        session.add(glaucoma)
        session.commit()
    except Exception as ex:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(ex)) from ex


@router.put(
    "/database/verify_diagnosis",
    summary="Выполнение верификации диагноза для последней записи из базы данных",
    description="Выполняет верификацию диагноза для последней записи из базы данных",
    response_description="HTTP Status Code 200",
    response_model=GlaucomaPydantic,
    status_code=status.HTTP_200_OK,
)
def verify_diagnosis() -> GlaucomaPydantic:
    try:
        session = sessions.get_session()
        last_record = (
            session.query(GlaucomaEntity)
            .order_by(GlaucomaEntity.timestamp.desc())
            .first()
        )

        if not last_record:
            session.close()
            raise HTTPException(
                status_code=404, detail="Записей в базе данных не обнаружено"
            )

        last_record.verify = True

        session.commit()

        return GlaucomaPydantic(
            id=last_record.id,
            timestamp=last_record.timestamp,
            width=last_record.width,
            height=last_record.height,
            status=last_record.status,
            verify=last_record.verify,
            cdr_value=last_record.cdr_value,
            rdar_value=last_record.rdar_value,
        )
    except Exception as ex:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(ex)) from ex


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
