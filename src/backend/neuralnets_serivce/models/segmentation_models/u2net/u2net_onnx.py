# python
import time
from typing import Any, Tuple

# 3rdparty
import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime

# project
from src.backend.neuralnets_serivce.utils.segmentation_models.u2net_utils import (
    COLOR_MAP_DICT,
    OpticCupMaskBorderValuesOfPixels,
    OpticDiscMaskBorderValuesOfPixels,
    remap_image,
    u2net_preprocessing,
)


class U2Net_ONNX:
    """Класс для выполнения сегментационной модели U2Net в рамках сессии ONNXRuntime"""

    def __init__(self, path_to_u2net_onnx_weights: str, use_cuda: bool) -> None:
        """Конструктор класса U2Net_ONNX

        Параметры:
            * `path_to_efficientnet_onnx` (`str`): путь к весам U2Net в формате ONNX
            * `use_cuda` (bool): использовать ли CUDA для выполнения
        """
        self.u2net_session = onnxruntime.InferenceSession(
            path_to_u2net_onnx_weights,
            providers=(
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if use_cuda
                else ["CPUExecutionProvider"]
            ),
        )
        self.u2net_input_width = self.u2net_session.get_inputs()[0].shape[3]
        self.u2net_input_height = self.u2net_session.get_inputs()[0].shape[2]
        self.u2net_input_channels = self.u2net_session.get_inputs()[0].shape[1]
        self.u2net_input_name = self.u2net_session.get_inputs()[0].name

    def inference(self, image: npt.NDArray[Any]) -> Tuple[float, float]:
        """Метод для выполнения модели U2Net на изображении

        Параметры:
            * `image` (`npt.NDArray[Any])`: объект изображения

        Возвращает:
            * `Tuple[float, float]`: кортеж со значением коэффициента RDAR и временем выполнения
        """
        image_array = u2net_preprocessing(image)
        start_time = time.perf_counter() * 1000
        outputs = self.u2net_session.run(None, {self.u2net_input_name: image_array})
        end_time = time.perf_counter() * 1000
        inference_time_ms = round((end_time - start_time) * 1000, 3)

        # в первом выходе содержатся искомые маски объектов
        first_output = np.squeeze(outputs[0], axis=(0)).astype(np.uint8)
        first_output_color = cv2.cvtColor(first_output, cv2.COLOR_GRAY2BGR)
        first_output_color_mapped = np.array(
            remap_image(first_output_color, COLOR_MAP_DICT)
        )
        first_output_color_mapped_gray = cv2.cvtColor(
            first_output_color_mapped, cv2.COLOR_BGR2GRAY
        )

        mask_for_optic_disc = cv2.inRange(first_output_color_mapped_gray, 66, 89)
        mask_for_optic_cup = cv2.inRange(first_output_color_mapped_gray, 66, 66)

        optic_disc_area = np.sum(mask_for_optic_disc > 0)
        optic_cup_area = np.sum(mask_for_optic_cup > 0)

        rdar_value = optic_cup_area / optic_disc_area
        return rdar_value, inference_time_ms
