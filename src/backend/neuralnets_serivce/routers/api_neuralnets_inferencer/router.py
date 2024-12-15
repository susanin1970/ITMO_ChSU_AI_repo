# python
import io
from typing import Any

# 3rdparty
import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image

# project
from src.backend.neuralnets_serivce.routers.api_neuralnets_inferencer.neuralnets_inferencer import (
    NeuralNetsInferencer,
)
from src.backend.neuralnets_serivce.schemas.service_output import (
    NeuralNetsServiceOutput,
)
from src.backend.neuralnets_serivce.tools.logging_tools import get_logger

service_config_path = "src/backend/neuralnets_serivce/configs/service_config.json"

router = APIRouter(tags=["Pipes Neural Nets Inferencer"], prefix="")
logger = get_logger()
inferencer = NeuralNetsInferencer(service_config_path)


@router.post(
    "/inference",
    response_description="Результаты обработки изображения глазного дна каскадом сетей, содержащие информацию о статусе диагноза наличия/отсутствия признаков глаукомы, а также значения коэффициентов CDR и RDAR",
    description="Выполняет прогон изображения глазного дна каскадом нейронных сетей",
    response_model=NeuralNetsServiceOutput,
)
async def inference(
    image: UploadFile = File(..., description="Объект изображения глазного дна")
) -> NeuralNetsServiceOutput:
    """Эндпойнт для выполнения каскада сеетй на изображениях глазного дна

    Параметры:
        * `image` (`UploadFile`, `optional`): объект изображения глазного дна

    Возвращает:
        `NeuralNetsServiceOutput`: результаты обработки изображения глазного дна
    """
    image_content: bytes = await image.read()
    image_array: npt.NDArray[Any] = np.array(  # type: ignore
        Image.open(io.BytesIO(image_content))
    )
    image_array = image_array[200:1750, 750:2300]
    neuralnets_service_output = inferencer.inference(image_array)
    return neuralnets_service_output
