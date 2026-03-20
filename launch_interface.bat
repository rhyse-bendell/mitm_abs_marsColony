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
echo Running preflight checks...
%PYTHON_CMD% "%SCRIPT_DIR%scripts\preflight_check.py"
set "PREFLIGHT_EXIT=%ERRORLEVEL%"

if not "%PREFLIGHT_EXIT%"=="0" (
  echo.
  echo Preflight reported issues.
  choice /M "Attempt controlled dependency repair now"
  if errorlevel 2 (
    echo Repair skipped. Resolve issues manually, then rerun launcher.
    pause
    popd >nul
    exit /b %PREFLIGHT_EXIT%
  )
  call :ensure_pip
  if errorlevel 1 (
    echo.
    echo Error: pip is unavailable, so repair could not run.
    pause
    popd >nul
    exit /b 1
  )
  %PYTHON_CMD% "%SCRIPT_DIR%scripts\preflight_check.py" --repair
  if errorlevel 1 (
    echo.
    echo Error: repair did not resolve all required preflight checks.
    pause
    popd >nul
    exit /b 1
  )
)

set "MPLBACKEND=TkAgg"
echo MPLBACKEND set to %MPLBACKEND% for Tk-compatible launch.

if not exist "%SCRIPT_DIR%interface.py" (
  echo.
  echo Error: interface.py was not found at:
  echo   %SCRIPT_DIR%interface.py
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
