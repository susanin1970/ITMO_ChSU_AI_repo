# 3rdparty
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, TIMESTAMP, Boolean

Base = declarative_base()


class GlaucomaEntity(Base):
    """
    Сущность из sqlalchemy, описывающий результат анализа на глаукому в БД
    """

    __tablename__ = "glaucoma"
    id = Column(Integer, primary_key=True)
    timestamps = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    status = Column(Boolean)
    verify = Column(Boolean)
    cdr_value = Column(Float)
    rdar_value = Column(Float)
