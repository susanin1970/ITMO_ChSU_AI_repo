# python
from enum import Enum
from typing import Dict, List

# 3rdparty
import cv2
import numpy as np
import numpy.typing as npt

COLOR_MAP_DICT = {
    0: [0, 0, 0],
    1: [234, 60, 15],
    2: [59, 26, 224],
}


class OpticDiscMaskBorderValuesOfPixels(Enum):
    """Перечисление граничных значений пикселей изображения с масками, полученными с помощью U2Net, для извлечения маски оптического диска"""

    LOWER_VALUE = 66
    UPPER_VALUE = 89


class OpticCupMaskBorderValuesOfPixels(Enum):
    """Перечисление граничных значений пикселей изображения с масками, полученными с помощью U2Net, для извлечения маски глазного бокала"""

    LOWER_VALUE = 66
    UPPER_VALUE = 66


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

def u2net_preprocessing(image: npt.NDArray[np.float32], mean: npt.NDArray[np.float32], std) -> npt.NDArray[np.float32]:
    """Функция для выполнения препроцессинга U2Net

    Параметры :
        * `image` (`npt.NDArray[Any]`): объект изображения

    Возвращает:
        * `npt.NDArray[Any]:` предобработанное изображение для подачи в сегментационную модель U2Net
    """
image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_array = image_array / 255.0
    image_array = (image_array - mean) / std
    image_array = np.expand_dims(image_array, axis=(0))
    image_array = np.transpose(image_array, (0, 3, 1, 2))
    image_array = image_array.astype(np.float32)
