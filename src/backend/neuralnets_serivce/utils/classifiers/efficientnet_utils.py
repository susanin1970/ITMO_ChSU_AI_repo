# python
from typing import Any, Tuple

# 3rdparty
import numpy as np
import numpy.typing as npt
from PIL import Image


def efficientnet_preprocessing(
    image: npt.NDArray[Any], efficientnet_input_shape: Tuple[int, int]
) -> npt.NDArray[Any]:
    """Функция для выполнения препроцессинга EfficientNet

    Параметры :
        * `image` (`npt.NDArray[Any]`): объект изображения
        * `efficientnet_input_shape` (`Tuple[int, int]`): кортеж с шириной и высотой входа EfficientNet

    Возвращает:
        * `npt.NDArray[Any]:` предобработанное изображение для подачи в классификатор EfficientNet
    """
    efficientnet_input_width, efficientnet_input_height = efficientnet_input_shape
    image_array = np.array(
        Image.fromarray(image).resize(
            (efficientnet_input_width, efficientnet_input_height), Image.BILINEAR
        )
    )
    image_array = image_array.transpose(2, 0, 1)
    image_array = np.expand_dims(image_array, axis=(0)).astype(np.float32)
    return image_array
