@echo off

set VENV_FOLDER=env

call %VENV_FOLDER%\Scripts\activate.bat

python SLOP_Detector_flask.py