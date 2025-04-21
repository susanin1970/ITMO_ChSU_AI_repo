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
    calc_max_mask_diameter,
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

    def inference(self, image: npt.NDArray[Any]) -> Tuple[float, float, float]:
        """Метод для выполнения модели U2Net на изображении

        Параметры:
            * `image` (`npt.NDArray[Any])`: объект изображения

        Возвращает:
            * `Tuple[float, float, float]`: кортеж со значением коэффициентов CDR, RDAR и временем выполнения
        """
        image_array = u2net_preprocessing(image)
        start_time = time.perf_counter() * 1000
        outputs = self.u2net_session.run(None, {self.u2net_input_name: image_array})
        end_time = time.perf_counter() * 1000
        inference_time_ms = round(end_time - start_time, 3)

        # в первом выходе содержатся искомые маски объектов
        first_output = np.squeeze(outputs[0], axis=(0)).astype(np.uint8)
        first_output_color = cv2.cvtColor(first_output, cv2.COLOR_GRAY2BGR)
        first_output_color_mapped = np.array(
            remap_image(first_output_color, COLOR_MAP_DICT)
        )
        first_output_color_mapped_gray = cv2.cvtColor(
            first_output_color_mapped, cv2.COLOR_BGR2GRAY
        )

        mask_for_optic_disc = cv2.inRange(
            first_output_color_mapped_gray,
            OpticDiscMaskBorderValuesOfPixels.LOWER_VALUE.value,
            OpticDiscMaskBorderValuesOfPixels.UPPER_VALUE.value,
        )
        mask_for_optic_cup = cv2.inRange(
            first_output_color_mapped_gray,
            OpticCupMaskBorderValuesOfPixels.LOWER_VALUE.value,
            OpticCupMaskBorderValuesOfPixels.UPPER_VALUE.value,
        )

        optic_disc_area = np.sum(mask_for_optic_disc > 0)
        optic_cup_area = np.sum(mask_for_optic_cup > 0)

        if optic_disc_area == 0.0:
            rdar_value = 0.0
        else:
            rdar_value = round(optic_cup_area / optic_disc_area, 3)

        max_diameter_of_optic_disc_mask = calc_max_mask_diameter(mask_for_optic_disc)
        max_diameter_of_optic_cup_mask = calc_max_mask_diameter(mask_for_optic_cup)

        if max_diameter_of_optic_disc_mask == 0.0:
            cdr_value = 0.0
        else:
            cdr_value = round(
                max_diameter_of_optic_cup_mask / max_diameter_of_optic_disc_mask, 3
            )

        return cdr_value, rdar_value, inference_time_ms
