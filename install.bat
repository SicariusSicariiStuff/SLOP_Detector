@echo off

set VENV_FOLDER=env

if not exist %VENV_FOLDER% (
    python -m venv %VENV_FOLDER%
)

call %VENV_FOLDER%\Scripts\activate

pip install -r requirements.txt

:: Keep terminal open
cmd /k