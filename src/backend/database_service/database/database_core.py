import os
import logging
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.backend.database_service.entities.entity_glaucoma import Base

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class GlaucomaSQLiteDatabase:
    def __init__(self):
        logger.debug("Инициализация GlaucomaSQLiteDatabase")

        # Проверяем и удаляем существующую БД
        if os.path.exists("glaucoma.db"):
            logger.debug("Удаление существующего файла базы данных")
            os.remove("glaucoma.db")

        # Создаем движок
        logger.debug("Создание движка базы данных")
        self.engine = create_engine(
            "sqlite:///glaucoma.db",
            echo=True,
            connect_args={"check_same_thread": False},
        )

        # Явно создаем все таблицы
        logger.debug("Создание таблиц")
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.debug("Таблицы созданы успешно")
        except Exception as e:
            logger.error(f"Ошибка при создании таблиц: {str(e)}")
            raise

        # Проверяем создание таблиц
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        logger.debug(f"Созданы следующие таблицы: {tables}")

        if "glaucoma" not in tables:
            logger.error("Таблицу glaucoma не удалось создать")
            raise Exception("Ошибка при создании таблицы glaucoma")

        logger.debug("Инициализация мастера создания сессий")
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        logger.debug("Инициализация GlaucomaSQLiteDatabase завершена")

    def get_session(self):
        return self.SessionLocal()
