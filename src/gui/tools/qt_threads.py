# 3rdparty
import requests
from PyQt6 import QtCore
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import QMessageBox


class ProcessingImageThread(QThread):
    """Поток, определяющий обработку изображения нейросетевым сервисом"""

    processing_image_signal = QtCore.pyqtSignal(requests.Response)

    def __init__(self, path_to_image: str, parent=None):
        """Конструктор класса потока

        Параметры:
            * `path_to_image` (`str`): путь к изображению
        """
        QtCore.QThread.__init__(self, parent)
        self.path_to_image = path_to_image

    def run(self):
        """Метод запуска потока"""

        if not self.path_to_image:
            return

        try:
            response = requests.post(
                "http://localhost:8000/inference",
                files={"image": open(self.path_to_image, "rb")},
            )
        except TypeError:
            QMessageBox(
                QMessageBox.Icon.Critical,
                "Ошибка",
                "Загрузите изображение для обработки!",
                QMessageBox.StandardButton.Ok,
            ).exec()
            return

        self.processing_image_signal.emit(response)
