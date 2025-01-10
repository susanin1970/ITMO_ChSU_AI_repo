import sqlite3

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

class database_sqllite:
    def __init__(self):
        self.engine = create_engine('sqlite:///glaucoma.db')
        Base = declarative_base()
        Base.metadata.create_all(self.engine)
        pass

    def get_session(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session