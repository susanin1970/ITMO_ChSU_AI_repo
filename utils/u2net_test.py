# python
import argparse
import os

# 3rdparty
import cv2
import numpy as np

# project
from src.backend.neuralnets_serivce.utils.segmentation_models.u2net_utils import (
    COLOR_MAP_DICT,
    OpticCupMaskBorderValuesOfPixels,
    OpticDiscMaskBorderValuesOfPixels,
    remap_image,
)
from utils.u2net_testing_utils import U2Net_ONNX


def dice_coefficient(y_true, y_pred) -> float:
    """
    Вычисляет коэффициент Дайса между двумя бинарными масками

    Args:
        y_true: бинарная маска наземной истины
        y_pred: бинарная маска предсказания модели

    Returns:
        dice: коэффициент Дайса (от 0 до 1)
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    intersection = np.sum(y_true_f & y_pred_f)

    # Избегаем деления на ноль
    if np.sum(y_true_f) + np.sum(y_pred_f) == 0:
        return 1.0

    dice = (2.0 * intersection) / (np.sum(y_true_f) + np.sum(y_pred_f))
    return dice


def arguments_parser():
    parser = argparse.ArgumentParser(
        description="Скрипт для тестирования модели U2Net в формате ONNX"
    )
    parser.add_argument(
        "-s", "--subset_path", type=str, help="Путь к выборке для тестирования"
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        help="Путь к чекпойнту обученной модели EfficientNet",
    )
    parser.add_argument(
        "-uc", "--use_cuda", type=bool, help="Использовать ли CUDA при тестировании"
    )
    args = parser.parse_args()
    return args


def main():
    args = arguments_parser()
    path_to_test_data = args.subset_path
    path_to_u2net_onnx_checkpoint = args.checkpoint_path
    use_cuda = args.use_cuda

    u2net_model = U2Net_ONNX(path_to_u2net_onnx_checkpoint, use_cuda)

    path_to_annotations = os.path.join(path_to_test_data, "labels")
    path_to_images = os.path.join(path_to_test_data, "images")

    optic_cup_dice_coeffs_list = []
    optic_disc_dice_coeffs_list = []
    for image, annotation in zip(
        os.listdir(path_to_images), os.listdir(path_to_annotations)
    ):
        path_to_image = os.path.join(path_to_images, image)
        path_to_annotation = os.path.join(path_to_annotations, annotation)

        image_array = cv2.imread(path_to_image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        mask_for_optic_cup_in_image_array, mask_for_optic_disc_in_image_array = (
            u2net_model.inference(image_array)
        )

        annotation_array = cv2.imread(path_to_annotation)

        annotation_color_mapped = np.array(
            remap_image(annotation_array, COLOR_MAP_DICT)
        )

        annotation_color_mapped_gray = cv2.cvtColor(
            annotation_color_mapped, cv2.COLOR_BGR2GRAY
        )

        mask_for_optic_disc_in_image_annotation = cv2.inRange(
            annotation_color_mapped_gray,
            OpticDiscMaskBorderValuesOfPixels.LOWER_VALUE.value,
            OpticDiscMaskBorderValuesOfPixels.UPPER_VALUE.value,
        )
        mask_for_optic_cup_in_image_annotation = cv2.inRange(
            annotation_color_mapped_gray,
            OpticCupMaskBorderValuesOfPixels.LOWER_VALUE.value,
            OpticCupMaskBorderValuesOfPixels.UPPER_VALUE.value,
        )

        optic_cup_dice_coeff = dice_coefficient(
            mask_for_optic_cup_in_image_array, mask_for_optic_cup_in_image_annotation
        )
        optic_disc_dice_coeff = dice_coefficient(
            mask_for_optic_disc_in_image_array, mask_for_optic_disc_in_image_annotation
        )

        optic_cup_dice_coeffs_list.append(optic_cup_dice_coeff)
        optic_disc_dice_coeffs_list.append(optic_disc_dice_coeff)

    print(
        f"Average Dice Coefficient for optic cup: {np.mean(optic_cup_dice_coeffs_list)}"
    )
    print(
        f"Average Dice Coefficient for opric disc: {np.mean(optic_disc_dice_coeffs_list)}"
    )
    print(
        f"Average Dice Coefficient for all classes: {(np.mean(optic_cup_dice_coeffs_list) + np.mean(optic_disc_dice_coeffs_list)) / 2}"
    )


if __name__ == "__main__":
    main()
