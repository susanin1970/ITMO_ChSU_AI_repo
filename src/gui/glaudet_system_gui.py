# 3rdparty
from pydantic import TypeAdapter
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QWidget,
    QTabWidget,
    QLineEdit,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
)
from PyQt6.QtGui import QPixmap
import requests
import sys

# project
from src.backend.neuralnets_serivce.schemas.service_output import (
    NeuralNetsServiceOutput,
)


class GlaucomaDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Glaudet")
        self.setGeometry(100, 100, 1280, 720)

        self.image_path = None
        self.processing_results = None  # Путь к выбранному изображению
        self.initUI()

        self.neuralnets_service_type_adapter = TypeAdapter(NeuralNetsServiceOutput)

    def initUI(self):
        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Main Window Tab
        self.main_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Главное окно")

        self.main_layout = QHBoxLayout()
        self.main_layout.setSpacing(10)  # Отступы между секциями

        # Left: Image Section
        self.image_section = QVBoxLayout()
        self.image_section.setSpacing(10)  # Отступы между элементами

        self.image_label = QLabel()
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setMinimumSize(768, 563)  # Увеличена высота на 10%
        self.image_label.setScaledContents(True)  # Заполняет ImageBox
        self.image_section.addWidget(self.image_label)

        # File path label
        self.image_path = QLineEdit()
        self.image_path.setReadOnly(True)
        self.image_section.addWidget(self.image_path)

        # Layout для кнопок
        self.buttons_layout = QVBoxLayout()
        self.buttons_layout.setSpacing(10)  # Отступы между кнопками

        self.open_image_button = QPushButton("Открыть изображение")
        self.open_image_button.setFixedSize(800, 40)  # Увеличен размер кнопок
        self.open_image_button.clicked.connect(self.open_image)
        self.buttons_layout.addWidget(self.open_image_button)

        self.process_image_button = QPushButton("Обработать изображение")
        self.process_image_button.setFixedSize(800, 40)  # Увеличен размер кнопок
        self.process_image_button.clicked.connect(self.process_image)
        self.buttons_layout.addWidget(self.process_image_button)

        self.image_section.addLayout(self.buttons_layout)

        self.main_layout.addLayout(self.image_section)

        # Right: Info Section
        self.info_section = QVBoxLayout()
        self.info_section.setSpacing(10)

        self.image_class_value = ""
        self.image_class_confidence = ""
        self.cdr_value = ""
        self.rdar_value = ""
        self.verificate_diagnosis = ""

        self.diagnosis_label = QLabel(
            f"Признаки глаукомы: {self.image_class_value}{self.image_class_confidence}\n"
            f"Значение CDR: - {self.cdr_value}\n"
            f"Значение RDAR: - {self.rdar_value}\n"
            f"Диагноз верифицирован: - {self.verificate_diagnosis}"
        )
        self.diagnosis_label.setStyleSheet("border: 1px solid black; padding: 5px;")
        self.diagnosis_label.setFixedHeight(120)
        self.info_section.addWidget(self.diagnosis_label)

        self.logo_label = QLabel()
        self.logo_label.setStyleSheet("border: 1px solid black;")
        self.logo_label.setMinimumHeight(256)
        self.logo_label.setScaledContents(True)  # Заполняет ImageBox
        self.info_section.addWidget(self.logo_label)

        # Загрузка логотипа
        logo_path = r"src\gui\assets\image.jpg"  # Укажите путь к логотипу
        pixmap = QPixmap(logo_path)
        self.logo_label.setPixmap(
            pixmap.scaled(
                self.logo_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        # Buttons for actions
        self.verify_button = QPushButton("Верифицировать результаты обработки")
        self.verify_button.setFixedHeight(40)
        self.info_section.addWidget(self.verify_button)

        self.show_areas_button = QPushButton(
            "Отобразить области изображения, ответственные за принятие решения"
        )
        self.show_areas_button.setFixedHeight(40)
        self.info_section.addWidget(self.show_areas_button)

        self.main_layout.addLayout(self.info_section)
        self.main_tab.setLayout(self.main_layout)

        # Event Log Tab
        self.log_tab = QWidget()
        self.tabs.addTab(self.log_tab, "Журнал событий")

        self.log_layout = QVBoxLayout()
        self.log_table = QTableWidget(0, 6)  # 6 столбцов
        self.log_table.setHorizontalHeaderLabels(
            [
                "ID изображения",
                "Таймстамп",
                "Ширина\nизображения",
                "Высота\nизображения",
                "Класс наличия\nглаукомы",
                "Верификация\nдиагноза",
            ]
        )
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.horizontalHeader().setDefaultSectionSize(200)
        self.log_table.setStyleSheet("border: 1px solid black;")
        self.log_layout.addWidget(self.log_table)

        # Add button under the table
        self.get_data_button = QPushButton("Получить данные из базы")
        self.get_data_button.clicked.connect(self.fetch_data_from_database)
        self.log_layout.addWidget(self.get_data_button)

        self.log_tab.setLayout(self.log_layout)

        # About Tab
        self.about_tab = QWidget()
        self.tabs.addTab(self.about_tab, "О программе")

        self.about_layout = QVBoxLayout()
        self.about_text = QLabel("Информация о программе для обнаружения глаукомы")
        self.about_layout.addWidget(self.about_text)
        self.about_tab.setLayout(self.about_layout)

    def open_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(
            self, "Выберите изображение", "", "Images (*.png *.xpm *.jpg)"
        )
        if image_path:
            self.image_path.setText(image_path)
            self.image_path = image_path
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))

    def process_image(self):
        if not self.image_path:
            return

        # Заглушка обработки
        try:
            response = requests.post(
                "http://localhost:8000/inference",
                files={"image": open(self.image_path, "rb")},
            )
        except TypeError:
            QMessageBox(
                QMessageBox.Icon.Critical,
                "Ошибка",
                "Загрузите изображение для обработки!",
                QMessageBox.StandardButton.Ok,
            ).exec()
            return

        response_object = self.neuralnets_service_type_adapter.validate_python(
            response.json()
        )

        self.image_class_value = (
            "отсутствуют"
            if response_object.predicted_class == "нет признаков глаукомы"
            else "присутствуют"
        )

        self.image_class_confidence = response_object.predicted_class_confidence
        self.cdr_value = response_object.cdr_value
        self.rdar_value = response_object.rdar_value

        self.diagnosis_label.setText(
            f"Признаки глаукомы: {self.image_class_value} с вероятностью {round(self.image_class_confidence, 3) * 100}%\n"
            f"Значение CDR: - {self.cdr_value}\n"
            f"Значение RDAR: - {self.rdar_value}\n"
            f"Диагноз верифицирован: - {self.verificate_diagnosis}"
        )

        # Обновление таблицы
        # row_position = self.event_table.rowCount()
        # self.event_table.insertRow(row_position)

        # self.event_table.setItem(row_position, 0, QTableWidgetItem(timestamp))
        # self.event_table.setItem(row_position, 1, QTableWidgetItem(str(height)))
        # self.event_table.setItem(row_position, 2, QTableWidgetItem(str(width)))
        # self.event_table.setItem(row_position, 3, QTableWidgetItem(diagnosis))
        # self.event_table.setItem(row_position, 4, QTableWidgetItem(str(is_verified)))
        # self.event_table.setItem(row_position, 5, QTableWidgetItem(f"{cdr} / {rdar}"))

    def verify_results(self):
        pass  # Заглушка для верификации

    def show_important_image_fields(self):
        pass  # Заглушка для отображения областей изображения

    def fetch_data_from_database(self):
        pass

    def send_data_to_database(self):
        pass

    def update_data_ro_database(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GlaucomaDetectionApp()
    window.show()
    sys.exit(app.exec())
