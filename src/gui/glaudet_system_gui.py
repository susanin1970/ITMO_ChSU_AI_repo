# python
import datetime
import os
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
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# project
from src.backend.database_service.schemas.database_service_schemas import (
    GlaucomaPydantic,
)
from src.backend.neuralnets_serivce.schemas.service_output import (
    NeuralNetsServiceOutput,
)
from src.gui.tools.qt_delegates import TableCellDelegate
from src.gui.tools.qt_threads import ProcessingImageThread


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
        self.process_image_thread = None
        self.initUI()

        self.neuralnets_service_type_adapter = TypeAdapter(NeuralNetsServiceOutput)
        self.glaucoma_pydantic_type_adapter = TypeAdapter(GlaucomaPydantic)

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

        self.info_section = QVBoxLayout()
        self.info_section.setSpacing(10)

        self.image_class_value = ""
        self.image_class_confidence = ""
        self.cdr_value = ""
        self.rdar_value = ""
        self.verificate_diagnosis = ""
        self.image_id = ""
        self.last_directory = ".\\"

        self.image_width = 0
        self.image_height = 0
        self.timestamp = ""

        self.glaucoma_processing_result = GlaucomaPydantic()

        self.diagnosis_label = QLabel(
            f"Идентификатор изображения: {self.image_id}\n"
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
        self.verify_button.clicked.connect(self.verify_diagnosis)

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
        self.delegate = TableCellDelegate()
        self.log_table.setItemDelegate(self.delegate)
        self.log_layout.addWidget(self.log_table)

        # Add button under the table
        self.get_data_button = QPushButton("Получить данные из базы")
        self.get_data_button.clicked.connect(self.fetch_all_data_from_database)
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
            self,
            "Выберите изображение",
            self.last_directory,
            "Images (*.png *.xpm *.jpg)",
        )
        if image_path:
            self.last_directory = os.path.dirname(image_path)
            self.image_path.setText(image_path)
            self.image_path_str = image_path
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))
            image_array = cv2.imread(image_path)
            image_height, image_width, _ = image_array.shape
            self.image_width = image_width
            self.image_height = image_height

    def process_image(self):
        if (
            self.process_image_thread is not None
            and self.process_image_thread.isRunning()
        ):
            QMessageBox(
                QMessageBox.Icon.Information,
                "Информация",
                "Обработка изображения уже запущена.",
                QMessageBox.StandardButton.Ok,
            ).exec()
            return

        self.process_image_thread = ProcessingImageThread(self.image_path_str)
        self.process_image_thread.processing_image_signal.connect(
            self.finish_process_image_thread, Qt.ConnectionType.QueuedConnection
        )
        self.process_image_thread.finished.connect(
            self.on_finished_process_image_thread
        )
        self.timestamp = str(datetime.datetime.now())
        self.process_image_thread.start()

    def finish_process_image_thread(self, result):
        response_object = self.neuralnets_service_type_adapter.validate_python(
            result.json()
        )

        self.image_class_value = (
            "отсутствуют"
            if response_object.predicted_class == "нет признаков глаукомы"
            else "присутствуют"
        )

        self.image_class_confidence = response_object.predicted_class_confidence
        self.cdr_value = response_object.cdr_value
        self.rdar_value = response_object.rdar_value
        self.verificate_diagnosis = "нет"
        self.verificate_diagnosis_for_pydantic = (
            False if self.verificate_diagnosis.lower() == "нет" else True
        )

        if self.image_id == "":
            self.image_id = 0
        else:
            self.image_id += 1

        self.diagnosis_label.setText(
            f"Идектификатор изображения: {self.image_id}\n"
            f"Признаки глаукомы: {self.image_class_value} с вероятностью {round(self.image_class_confidence, 3) * 100}%\n"
            f"Значение CDR: {self.cdr_value}\n"
            f"Значение RDAR: {self.rdar_value}\n"
            f"Диагноз верифицирован: {self.verificate_diagnosis}",
        )

        self.glaucoma_processing_result.id = self.image_id
        self.glaucoma_processing_result.timestamp = self.timestamp
        self.glaucoma_processing_result.width = self.image_width
        self.glaucoma_processing_result.height = self.image_height
        self.glaucoma_processing_result.verify = self.verificate_diagnosis_for_pydantic
        self.glaucoma_processing_result.cdr_value = self.cdr_value
        self.glaucoma_processing_result.rdar_value = self.rdar_value

        self.add_data_to_database()

    def on_finished_process_image_thread(self):
        self.process_image_thread.deleteLater()
        self.process_image_thread = None

    def verify_diagnosis(self):
        # pass  # Заглушка для верификации
        try:
            response = requests.put("http://localhost:8080/database/verify_diagnosis")
            print(
                f"Статус-код от эндпойнта сервиса базы данных по верификации диагноза: {response.status_code}"
            )
            response_object = self.glaucoma_pydantic_type_adapter.validate_python(
                response.json()
            )

            self.verificate_diagnosis = (
                "нет" if response_object.verify is False else "да"
            )

            self.diagnosis_label.setText(
                f"Идектификатор изображения: {self.image_id}\n"
                f"Признаки глаукомы: {self.image_class_value} с вероятностью {round(self.image_class_confidence, 3) * 100}%\n"
                f"Значение CDR: {self.cdr_value}\n"
                f"Значение RDAR: {self.rdar_value}\n"
                f"Диагноз верифицирован: {self.verificate_diagnosis}",
            )
        except Exception as ex:
            print(ex)

    def show_important_image_fields(self):
        pass  # Заглушка для отображения областей изображения

    def fetch_all_data_from_database(self):
        try:
            response = requests.post("http://localhost:8080/database/fetch_all_data")
            print(
                f"Статус-код от энндпойнта сервиса базы данных по извлечению всех данных из базы: {response.status_code}"
            )
            all_fetched_data_from_db = response.json()
            rows_count = self.log_table.rowCount()
            columns_count = self.log_table.columnCount()

            table_data = []
            for row_id in range(rows_count):
                row_data = []
                for column_id in range(columns_count):
                    item = self.log_table.item(row_id, column_id).text()
                    if item in ["признаки глаукомы отсутствуют", "не верифицирован"]:
                        item = False
                    if item in ["признаки глаукомы присутствуют", "верифицирован"]:
                        item = True
                    row_data.append(item)
                table_data.append(row_data)

            for item in all_fetched_data_from_db:
                item_python = self.glaucoma_pydantic_type_adapter.validate_python(item)
                item_python_list = list(dict(item_python).values())
                item_python_list = [
                    (
                        str(item)
                        if (not isinstance(item, str) and not isinstance(item, bool))
                        else item
                    )
                    for item in item_python_list
                ]

                if item_python_list in table_data:
                    continue

                row_position = self.log_table.rowCount()
                self.log_table.insertRow(row_position)
                self.log_table.setItem(
                    row_position,
                    0,
                    QTableWidgetItem(str(item_python.id)),
                )
                self.log_table.setItem(
                    row_position,
                    1,
                    QTableWidgetItem(str(item_python.timestamp)),
                )
                self.log_table.setItem(
                    row_position,
                    2,
                    QTableWidgetItem(str(item_python.width)),
                )
                self.log_table.setItem(
                    row_position,
                    3,
                    QTableWidgetItem(str(item_python.height)),
                )
                self.log_table.setItem(
                    row_position,
                    4,
                    QTableWidgetItem(
                        "признаки глаукомы присутствуют"
                        if item_python.status is True
                        else "признаки глаукомы отсутствуют"
                    ),
                )
                self.log_table.setItem(
                    row_position,
                    5,
                    QTableWidgetItem(
                        "верифицирован"
                        if item_python.verify is True
                        else "не верифицирован"
                    ),
                )
                self.log_table.setItem(
                    row_position,
                    6,
                    QTableWidgetItem(str(item_python.cdr_value)),
                )
                self.log_table.setItem(
                    row_position,
                    7,
                    QTableWidgetItem(str(item_python.rdar_value)),
                )

        except Exception as ex:
            print(ex)

    def add_data_to_database(self):
        try:
            response = requests.post(
                "http://localhost:8080/database/add_data",
                json=self.glaucoma_processing_result.model_dump(),
            )
            print(
                f"Статус-код от эндпойнта сервиса базы данных по добавлению данных в базу: {response.status_code}"
            )

        except Exception as ex:
            print(ex)

    def update_data_to_database(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GlaucomaDetectionApp()
    window.show()
    sys.exit(app.exec())
