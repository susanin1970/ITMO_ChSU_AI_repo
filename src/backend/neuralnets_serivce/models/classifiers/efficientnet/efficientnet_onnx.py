# python
import time
from typing import Any

# 3rdparty
import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime

# project
from src.backend.neuralnets_serivce.utils.classifiers.efficientnet_utils import (
    efficientnet_preprocessing,
)


class EfficientNet_ONNX:
    """
    Класс для выполнения классификатора EfficientNet в рамках сессии ONNXRuntime
    """

    def __init__(self, path_to_efficientnet_onnx_weights: str, use_cuda: bool) -> None:
        """Конструктор класса EfficientNet_ONNX

        Параметры:
            * `path_to_efficientnet_onnx` (`str`): путь к весам EfficientNet в формате ONNX
            * `use_cuda` (bool): использовать ли CUDA для выполнения
        """
        self.classifier = onnxruntime.InferenceSession(
            path_to_efficientnet_onnx_weights,
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if use_cuda
                else ["CPUExecutionProvider"]
            ),
        )
        self.efficientnet_input_width = self.classifier.get_inputs()[0].shape[3]
        self.efficientnet_input_height = self.classifier.get_inputs()[0].shape[2]
        self.efficientnet_input_channels = self.classifier.get_inputs()[0].shape[1]
        self.efficientnet_input_name = self.classifier.get_inputs()[0].name

    def classify(self, image: npt.NDArray[Any]) -> tuple[Any, float, float]:
        """Метод для выполнения классификатора EfficientNet на изображении

        Параметры:
            * `image` (`npt.NDArray[Any])`: объект изображения

        Возвращает:
            * `tuple[Any, float, float]`: кортеж с индексом класса изображения и временем выполнения
        """
        image_array = efficientnet_preprocessing(
            image, (self.efficientnet_input_width, self.efficientnet_input_height)
        )
        start_time = time.perf_counter()
        efficientnet_outputs = self.classifier.run(
            None, {self.efficientnet_input_name: image_array}
        )
        end_time = time.perf_counter()
        inference_time_ms = round((end_time - start_time) * 1000, 3)
        class_id = efficientnet_outputs[0].argmax()
        confidence = efficientnet_outputs[0].max()
        return class_id, confidence, inference_time_ms
