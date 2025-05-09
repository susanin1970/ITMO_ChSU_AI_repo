# python
import json
import logging
from datetime import datetime

# 3rdparty
import numpy as np
import numpy.typing as npt
from pydantic import TypeAdapter

# project
from src.backend.neuralnets_serivce.models.classifiers.efficientnet.efficientnet_onnx import (
    EfficientNet_ONNX,
)
from src.backend.neuralnets_serivce.models.segmentation_models.u2net.u2net_onnx import (
    U2Net_ONNX,
)
from src.backend.neuralnets_serivce.schemas.service_config import (
    NeuralNetsServiceConfig,
)
from src.backend.neuralnets_serivce.schemas.service_output import (
    GlaucomaSignsStatus,
    NeuralNetsServiceOutput,
)
from src.backend.neuralnets_serivce.tools.logging_tools import get_logger

logger = get_logger()


class NeuralNetsInferencer:
    """
    Класс, реализующий инфыеренс каскада моделей для обнаружения признаков глаукомы
    """

    def __init__(self, path_to_service_config: str) -> None:
        """
        Конструктор класса

        Параметры:
            * `service_config_path` (`str`): путь к файлу конфигурации сервиса

        """
        self._glaucoma_signs_status_list = [
            field.value for field in GlaucomaSignsStatus
        ]

        with open(path_to_service_config, "r") as service_config_file:
            service_config_dict = json.load(service_config_file)

        service_config_adapter = TypeAdapter(NeuralNetsServiceConfig)
        service_config_python = service_config_adapter.validate_python(
            service_config_dict
        )

        self._service_config_python = service_config_python

        logger.info("Инициализация нейросетевого сервиса")

        classifier_name = self._service_config_python.classification_config.name
        match classifier_name:
            case "EfficientNet-B0":
                self._classifier = EfficientNet_ONNX(
                    self._service_config_python.classification_config.path_to_weights,
                    self._service_config_python.classification_config.use_cuda,
                )
            case _:
                raise Exception(
                    f"Классификатор {classifier_name} не доступен для инициализации в сервисе"
                )

        logger.info(
            f"Инициализирован классификатор {classifier_name} с весовыми параметрами из {self._service_config_python.classification_config.path_to_weights}"
        )
        logger.info(
            f"Устройство для выполнения классификатора: {'GPU' if self._service_config_python.classification_config.use_cuda else 'CPU'}"
        )

        segmentation_model_name = self._service_config_python.segmentation_config.name
        match segmentation_model_name:
            case "U2Net":
                self._segmentation_model = U2Net_ONNX(
                    self._service_config_python.segmentation_config.path_to_weights,
                    self._service_config_python.segmentation_config.use_cuda,
                )
            case _:
                raise Exception(
                    f"Сегментационная модель {segmentation_model_name} не доступна для инициализации в сервисе"
                )
        logger.info(
            f"Инициализирована сегментационная модель {segmentation_model_name} с весовыми параметрами из {self._service_config_python.segmentation_config.path_to_weights}"
        )

    def inference(self, image: npt.NDArray[np.float32]) -> NeuralNetsServiceOutput:
        """Метод для выполнения инференса нейронных сетей

        Параметры:
             * `image` (npt.NDArray[np.float32]): объект изображения в формате np.array

        Возвращает:
            * `NeuralNetsServiceOutput`: объект выходных данных каскада сетей
        """
        neuralnets_service_output = NeuralNetsServiceOutput()
        timestamp_of_receiving_response: str = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S-%f"
        )
        logger.info(f"Запрос принят: {timestamp_of_receiving_response}")
        logger.info(f"Размерность принятого изображения {image.shape}")

        image_class_idx, confidence, classifier_inference_time = (
            self._classifier.classify(image)
        )
        image_class = self._glaucoma_signs_status_list[image_class_idx]
        logger.info(
            f"Статус наличия/отсутствия признаков глаукомы на изображении: {image_class}"
        )
        logger.info(
            f"Значение уверенности статуса наличия/отсутствия признаков глаукомы: {confidence}"
        )
        logger.info(f"Время инференса классификатора: {classifier_inference_time}")

        cdr_value, rdar_value, segmentation_model_inference_time = (
            self._segmentation_model.inference(image)
        )

        logger.info(f"Значение коэффициента CDR: {cdr_value}")
        logger.info(f"Значение коэффициента RDAR: {rdar_value}")
        logger.info(
            f"Время инференса модели сегментационной модели: {segmentation_model_inference_time}"
        )

        neuralnets_service_output.predicted_class = image_class
        neuralnets_service_output.predicted_class_confidence = float(confidence)
        neuralnets_service_output.cdr_value = cdr_value
        neuralnets_service_output.rdar_value = rdar_value

        return neuralnets_service_output
