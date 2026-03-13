@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul 2>&1
if errorlevel 1 (
  echo Error: Could not change to launcher directory:
  echo   %SCRIPT_DIR%
  echo.
  pause
  exit /b 1
)

set "PYTHON_CMD="
call :probe_python "py -3"
if not defined PYTHON_CMD call :probe_python "python"
if not defined PYTHON_CMD call :probe_python "python3"

if not defined PYTHON_CMD (
  echo Error: Python 3 was not found.
  echo Tried commands: py -3, python, python3
  echo.
  echo Install Python 3 and make sure one of these commands works in Command Prompt.
  echo Then run this launcher again.
  echo.
  pause
  popd >nul
  exit /b 1
)

echo Using Python command: %PYTHON_CMD%
echo Checking required modules...
%PYTHON_CMD% -c "import tkinter, matplotlib" >nul 2>&1
if errorlevel 1 (
  echo Error: Required modules are missing: tkinter and/or matplotlib.
  echo.
  echo Install matplotlib with:
  echo   %PYTHON_CMD% -m pip install matplotlib
  echo.
  echo tkinter usually comes with the standard Python installer on Windows.
  echo If tkinter is missing, repair/reinstall Python and include Tcl/Tk support.
  echo.
  pause
  popd >nul
  exit /b 1
)

echo Launching Mars Colony interface...
%PYTHON_CMD% "%SCRIPT_DIR%interface.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo The interface exited with code %EXIT_CODE%.
  echo Review the error output above.
  echo.
  pause
)

popd >nul
exit /b %EXIT_CODE%

:probe_python
if defined PYTHON_CMD exit /b 0
set "CANDIDATE=%~1"
%CANDIDATE% -c "import sys; sys.exit(0)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=%CANDIDATE%"
exit /b 0
