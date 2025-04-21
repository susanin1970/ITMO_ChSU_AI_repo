# python
from enum import Enum
from typing import Dict, List

# 3rdparty
import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist

COLOR_MAP_DICT = {
    0: [0, 0, 0],
    1: [234, 60, 15],
    2: [59, 26, 224],
}

MEAN = np.array([0.5, 0.5, 0.5])
STD = np.array([0.5, 0.5, 0.5])


class OpticDiscMaskBorderValuesOfPixels(Enum):
    """Перечисление граничных значений пикселей изображения с масками, полученными с помощью U2Net, для извлечения маски оптического диска"""

    LOWER_VALUE = 66
    UPPER_VALUE = 89


class OpticCupMaskBorderValuesOfPixels(Enum):
    """Перечисление граничных значений пикселей изображения с масками, полученными с помощью U2Net, для извлечения маски глазного бокала"""

    LOWER_VALUE = 66
    UPPER_VALUE = 66


def calc_max_mask_diameter(mask_image: npt.NDArray) -> float:
    """Функция для вычисления максимального диаметра маски оптического диска/глазного бокала

    Параметры:
        * `mask_image` (`npt.NDArray`): изображение с маской оптического диска/глазного бокала

    Возвращает:
        * `float`: значение максимального диаметра маски оптического диска/глазного бокала
    """
    try:
        contours, _ = cv2.findContours(
            mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        mask_contour = contours[0]
        points_of_mask = mask_contour.reshape(-1, 2)
        diameters_list = pdist(points_of_mask)
        max_diameter = np.max(diameters_list)
        return max_diameter
    except IndexError:
        return 0.0


def remap_image(
    image: npt.NDArray[np.float32], mapping: Dict[int, List[int]]
) -> npt.NDArray[np.float32]:
    """Функция для замены пикселей изображения

    Параметры:
        * `image` (`npt.NDArray[np.float32]`): объект исходного изображения
        * `mapping` (`Dict[int, List[int]]`): словарь маппинга старых значений пикселей на новые

    Возвращает:
        * `npt.NDArray[np.float32]`: объект измененного изображения
    """
    new_image = np.zeros_like(image)

    for old_value, new_value in mapping.items():
        mask = np.all(image == old_value, axis=-1)
        new_image[mask] = new_value

    return new_image


def u2net_preprocessing(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Функция для выполнения препроцессинга U2Net

    Параметры :
        * `image` (`npt.NDArray[Any]`): объект изображения

    Возвращает:
        * `npt.NDArray[Any]:` предобработанное изображение для подачи в сегментационную модель U2Net
    """
    image_array = image / 255.0
    image_array = (image_array - MEAN) / STD
    image_array = np.expand_dims(image_array, axis=(0))
    image_array = np.transpose(image_array, (0, 3, 1, 2))
    image_array = image_array.astype(np.float32)

    return image_array
