@echo off
start python -m src.backend.neuralnets_serivce.neuralnets_service
echo "Neuralnets service of GLAUDET system is started"
start python -m src.backend.database_service.database_service
echo "Database service of GLAUDET system is started"
start python -m src.gui.glaudet_system_gui
echo "GUI of GLAUDET system is started"
