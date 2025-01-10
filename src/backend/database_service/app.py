# python
import asyncio
import json
import os
import sys

# 3rdparty
import pydantic
from fastapi import FastAPI

# Конфигурация по роутеры

# project
from src.backend.database_service.routers.api_database import router as InfoRouter

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(
    title="Glaucoma Database Service API",
    version="0.1.0",
    description="",
    docs_url=None,
    redoc_url=None,
)

api_v1_prefix = ""
app.include_router(InfoRouter, prefix=api_v1_prefix)

app.docs_url = "/docs"
app.redoc_url = "/redocs"
app.setup()
