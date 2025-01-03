import sqlite3
from sqlalchemy import create_engine
from dataclasses import dataclass
from pydantic import BaseModel

engine = create_engine('sqlite:///glaucoma.db')

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, TIMESTAMP, Boolean

from sqlalchemy.orm import sessionmaker

# Бустрапер

print("initial app")

Base = declarative_base()
# model

# Дата класс для добавления данных в БД
class GlaucomaPydantic(BaseModel):
    id : int = None
    timestamps : int
    width : int 
    height : int
    status : bool
    verify : bool
    imgCache : str = None

# Фильтр по данным БД
class FilterData(BaseModel):
    imageId : int
    timestamps : int
    interval_width_min : int
    interval_width_max : int
    interval_height_min : int
    interval_height_max : int
    glaucomStatus : bool
    hasVerifiication : bool

class Glaucoma(Base):
    __tablename__ = 'glaucoma'
    id = Column(Integer, primary_key=True)
    timestamps = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    status = Column(Boolean)
    verify = Column(Boolean)
    imgCache = Column(String)

Base.metadata.create_all(engine)
name = "main"

from fastapi import FastAPI

app = FastAPI()

@app.put("/database/")
def update_processing_result_data_to_bd(imageId : int):
    Session = sessionmaker(bind=engine)
    session = Session()

    query = ''
    if imageId != None:
        query = session.query(Glaucoma).filter(Glaucoma.id == imageId)

    for data in query:
        data.verify = True
    
    session.commit()

@app.get("/database")
def fetch_processing_result_data_from_db_by_id(imageId : int):
    Session = sessionmaker(bind=engine)
    session = Session()

    query = ''
    if imageId != None:
        query = session.query(Glaucoma).filter(Glaucoma.id == imageId)
    
    for data in query:
        return data

@app.post("/database/filter")
def fetch_processing_result_data_from_db(filter : FilterData):
    Session = sessionmaker(bind=engine)
    session = Session()

    query = ''

    if filter.imageId != None:
        query = session.query(Glaucoma).filter(Glaucoma.id == filter.imageId)
    if filter.timestamps != None:
        query = session.query(Glaucoma).filter(Glaucoma.timestamp == filter.timestamps)
    if filter.interval_width_min != None:
        query = session.query(Glaucoma).filter(Glaucoma.width >= filter.interval_width_min)
    if filter.interval_height_min != None:
        query = session.query(Glaucoma).filter(Glaucoma.width >= filter.interval_height_min)
    if filter.interval_width_max != None:
        query = session.query(Glaucoma).filter(Glaucoma.width <= filter.interval_width_max)
    if filter.interval_height_max != None:
        query = session.query(Glaucoma).filter(Glaucoma.width <= filter.interval_height_max)
    if filter.glaucomStatus != None:
        query = session.query(Glaucoma).filter(Glaucoma.status == filter.glaucomStatus)
    if filter.hasVerifiication != None:
        query = session.query(Glaucoma).filter(Glaucoma.verify == filter.hasVerifiication)

    list = []
    for data in query:
        list.append(data)
    return list

@app.delete("/database")
def add_processing_result_data_to_db(imageId : int):
    Session = sessionmaker(bind=engine)
    session = Session()

    query = ''
    if imageId != None:
        query = session.query(Glaucoma).filter(Glaucoma.id == imageId)
        session.delete(query)

    session.commit()

@app.post("/database/")
def add_processing_result_data_to_db(database : GlaucomaPydantic):
    try:
        glaucoma = Glaucoma(timestamp = database.timestamp, width = database.width, height = database.height, status=database.status, verify=database.verify, imgCache = database.imgCache)

        Session = sessionmaker(bind=engine)
        session = Session() 
        session.add(glaucoma)
        session.commit()
    except Exception as Exc:
        print("Error: " + str(Exc))

if name == "main":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Example creating pizdantic => GlaucomaPydantic
# qw = GlaucomaIdenty(**dicti)
# add_processing_result_data_to_db(qw)