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

set "TKINTER_OK=0"
set "MATPLOTLIB_OK=0"
call :check_module tkinter TKINTER_OK
call :check_module matplotlib MATPLOTLIB_OK

if not "%MATPLOTLIB_OK%"=="1" (
  echo.
  echo matplotlib is missing. Attempting automatic installation...
  call :ensure_pip
  if errorlevel 1 (
    echo.
    echo Error: Could not prepare pip, so matplotlib could not be installed.
    echo.
    pause
    popd >nul
    exit /b 1
  )

  %PYTHON_CMD% -m pip install matplotlib
  if errorlevel 1 (
    echo.
    echo Error: matplotlib installation failed.
    echo.
    pause
    popd >nul
    exit /b 1
  )

  call :check_module matplotlib MATPLOTLIB_OK
  if not "%MATPLOTLIB_OK%"=="1" (
    echo.
    echo Error: matplotlib is still unavailable after installation.
    echo.
    pause
    popd >nul
    exit /b 1
  )
)

if not "%TKINTER_OK%"=="1" (
  echo.
  echo Error: tkinter is missing.
  echo tkinter is part of the standard Windows Python installation.
  echo Repair or reinstall Python and include Tcl/Tk support, then run this launcher again.
  echo.
  pause
  popd >nul
  exit /b 1
)

echo.
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

:check_module
%PYTHON_CMD% -c "import %~1" >nul 2>&1
if errorlevel 1 (
  set "%~2=0"
) else (
  set "%~2=1"
)
exit /b 0

:ensure_pip
%PYTHON_CMD% -m pip --version >nul 2>&1
if not errorlevel 1 exit /b 0

echo pip is missing. Attempting to bootstrap with ensurepip...
%PYTHON_CMD% -m ensurepip --upgrade
if errorlevel 1 (
  echo Error: ensurepip failed.
  exit /b 1
)

%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
  echo Error: pip is still unavailable after ensurepip.
  exit /b 1
)

exit /b 0
