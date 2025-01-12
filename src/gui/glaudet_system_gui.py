# python
import datetime
import sys

# 3rdparty
import cv2
import requests
from pydantic import TypeAdapter
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# project
from src.backend.neuralnets_serivce.schemas.service_output import (
    NeuralNetsServiceOutput,
)
from src.backend.database_service.schemas.database_service_schemas import (
    GlaucomaPydantic,
)


class GlaucomaDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Glaudet")
        self.setGeometry(100, 100, 1280, 720)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowMaximizeButtonHint
        )

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
        self.open_image_button.adjustSize()  # Увеличен размер кнопок
        self.open_image_button.setFixedHeight(40)
        self.open_image_button.clicked.connect(self.open_image)
        self.buttons_layout.addWidget(self.open_image_button)

        self.process_image_button = QPushButton("Обработать изображение")
        self.process_image_button.adjustSize()  # Увеличен размер кнопок
        self.process_image_button.setFixedHeight(40)
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
        self.image_id = ""

        self.image_width = 0
        self.image_height = 0
        self.timestamp = ""

        self.glaucoma_processing_result = GlaucomaPydantic()

        self.diagnosis_label = QLabel(
            f"Идентификатор изображения: {self.image_id}",
            f"Признаки глаукомы: {self.image_class_value}{self.image_class_confidence}\n"
            f"Значение CDR: - {self.cdr_value}\n"
            f"Значение RDAR: - {self.rdar_value}\n"
            f"Диагноз верифицирован: - {self.verificate_diagnosis}",
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
        self.log_table = QTableWidget(0, 8)  # 6 столбцов
        self.log_table.setHorizontalHeaderLabels(
            [
                "ID изображения",
                "Таймстамп",
                "Ширина\nизображения",
                "Высота\nизображения",
                "Класс наличия\nглаукомы",
                "Верификация\nдиагноза",
                "Значение CDR",
                "Значение RDAR",
            ]
        )
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.horizontalHeader().setDefaultSectionSize(170)
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
        self.about_text = QLabel(
            """
            Система “GLAUDET” предназначена для помощи офтальмологам при постановке диагноза глаукомы глаза, одной из самых распространенных глазных болезней. 
            Система применима для использования в медицинской отрасли, а именно в поликлиниках, офтальмологических клиниках в ходе выполнения определенных мероприятий, например, скрининговые обследования населения.
            В основе данной системы методы искусственного интеллекта, а именно глубокие нейронные сети для решения задач компьютерного зрения.\n
            Система "GLAUDET" обеспечивает покрытие покрывает следующей функциональности:
            •	Выдача проекта заключения, которое включает предполагаемую постановку диагноза
            •	Выдача показателя CDR, т. е. соотношения диаметра глазного бокала к диаметру зрительного диска
            
            Возможности системы: 
            •	Чтение изображений глазного дна в форматах JPEG/BMP/TIFF/PNG
            •	Обработка изображений глазного дна каскадом нейронных сетей на предмет наличия признаков глаукомы глаза
            •	Запись результатов обработки в базу данных
            •	Чтение результатов обработки из базы данных
            •	Редактирование результатов обработки в базе данных с помощью механизма верификации
            •	Отображение записей результатов обработки в журнале событий
            •	Отображение областей изображений, ответственных за принятие решения
        
            Система рассчитана на работу с RGB-изображениями глазного дна  в формате JPG/PNG/BMP/TIFF, в которых исключены помехи и шумы, с отчетливо видимыми участками глазного бокала, зрительного диска и места схождения кровеносных сосудов   
            """
        )
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
            image_array = cv2.imread(image_path)
            image_height, image_width, _ = image_array.shape
            self.image_width = image_width
            self.image_height = image_height

    def process_image(self):
        if not self.image_path:
            return

        try:
            self.timestamp = str(datetime.datetime.now())
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

        if self.image_id == "":
            self.image_id = 0
        else:
            self.image_id += 1

        self.diagnosis_label.setText(
            f"Идектификатор изображения: {self.image_id}",
            f"Признаки глаукомы: {self.image_class_value} с вероятностью {round(self.image_class_confidence, 3) * 100}%\n"
            f"Значение CDR: {self.cdr_value}\n"
            f"Значение RDAR: {self.rdar_value}\n"
            f"Диагноз верифицирован: - {self.verificate_diagnosis}",
        )

        self.glaucoma_processing_result.id = self.image_id
        self.glaucoma_processing_result.timestamp = self.timestamp
        self.glaucoma_processing_result.width = self.image_width
        self.glaucoma_processing_result.height = self.image_height
        self.glaucoma_processing_result.verify = self.verificate_diagnosis
        self.glaucoma_processing_result.cdr_value = self.cdr_value
        self.glaucoma_processing_result.rdar_value = self.rdar_value

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

    def add_data_to_database(self):
        try:
            response = requests.post(
                "http://localhost:8080/database",
                data=self.glaucoma_processing_result.model_dump_json(),
            )
        except Exception as ex:
            print(ex)

    def update_data_ro_database(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GlaucomaDetectionApp()
    window.show()
    sys.exit(app.exec())
