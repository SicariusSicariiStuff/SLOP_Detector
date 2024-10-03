@echo off

set VENV_FOLDER=env
set PYTHON_VERSION=3.11

if not exist %VENV_FOLDER% (
    python -m venv %VENV_FOLDER% --python=%PYTHON_VERSION%
)

call %VENV_FOLDER%\Scripts\activate

pip install -r requirements.txt

:: Keep terminal open
cmd /k