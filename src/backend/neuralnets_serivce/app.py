# python
import asyncio
import json
import os
import sys

# 3rdparty
import pydantic
from fastapi import FastAPI

# project
from src.backend.neuralnets_serivce.routers.api_info import router as InfoRouter
from src.backend.neuralnets_serivce.routers.api_neuralnets_inferencer import (
    router as NeuralNetsSerivceRouter,
)
from src.backend.neuralnets_serivce.schemas.service_config import (
    NeuralNetsServiceConfig,
)


service_config_path = "src/backend/neuralnets_serivce/configs/service_config.json"

with open(service_config_path, "r") as json_service_config:
    service_config_dict = json.load(json_service_config)

service_config_adapter: pydantic.TypeAdapter = pydantic.TypeAdapter(
    NeuralNetsServiceConfig
)

service_config_python = service_config_adapter.validate_python(service_config_dict)

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI(
    title="Pipe Neural Nets Inference Service API",
    version="0.1.0",
    description="",
    docs_url=None,
    redoc_url=None,
)
api_v1_prefix = ""

app.include_router(InfoRouter, prefix=api_v1_prefix)
app.include_router(NeuralNetsSerivceRouter, prefix=api_v1_prefix)

app.docs_url = "/docs"
app.redoc_url = "/redocs"
app.setup()
