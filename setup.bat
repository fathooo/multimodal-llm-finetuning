@echo off
setlocal enabledelayedexpansion

REM Mostrar el valor de PYTHON_HOME para depuracion
echo Valor de PYTHON_HOME: "%PYTHON_HOME%"

REM Verificar si la variable de entorno PYTHON_HOME esta configurada
IF "%PYTHON_HOME%"=="" (
    echo.
    echo ERROR: La variable de entorno PYTHON_HOME no esta configurada
    echo.
    set /p "use_system_python=Deseas usar la version de Python instalada en el sistema S/N: "
    if /i "!use_system_python!"=="S" (
        set "PYTHON_EXEC=python"
    ) else (
        echo.
        echo Por favor configura PYTHON_HOME correctamente y vuelve a intentarlo
        echo.
        exit /b 1
    )
) ELSE (
    REM Verificar si PYTHON_HOME apunta directamente a python.exe
    if EXIST "%PYTHON_HOME%\python.exe" (
        set "PYTHON_EXEC=%PYTHON_HOME%\python.exe"
        echo Usando Python en %PYTHON_HOME%
    ) ELSE (
        REM Verificar si PYTHON_HOME es un archivo ejecutable
        if EXIST "%PYTHON_HOME%" (
            set "PYTHON_EXEC=%PYTHON_HOME%"
            echo Usando Python %PYTHON_EXEC%
        ) ELSE (
            echo.
            echo ERROR: La ruta de PYTHON_HOME no es valida o no se encontro python.exe
            echo.
            set /p "use_system_python=Deseas usar la version de Python instalada en el sistema S/N: "
            if /i "!use_system_python!"=="S" (
                set "PYTHON_EXEC=python"
            ) else (
                echo.
                echo Por favor configura PYTHON_HOME correctamente y vuelve a intentarlo
                echo.
                exit /b 1
            )
        )
    )
)

REM Verificar la existencia de Python
%PYTHON_EXEC% --version > nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Python no esta instalado o no se encuentra en el PATH
    echo.
    exit /b 1
) ELSE (
    for /f "tokens=*" %%a in ('%PYTHON_EXEC% --version') do set "PYTHON_VER=%%a"
    echo %PYTHON_VER% detectada
)

REM Verificar si el entorno virtual ya esta creado
IF NOT EXIST "venv\Scripts\activate" (
    echo Creando entorno virtual usando %PYTHON_EXEC%
    %PYTHON_EXEC% -m venv venv
) ELSE (
    echo El entorno virtual ya existe
)

REM Activar el entorno virtual
echo Activando el entorno virtual
call venv\Scripts\activate

REM Verificar si pip esta instalado y actualizarlo
echo Verificando pip
venv\Scripts\python -m ensurepip --default-pip
venv\Scripts\python -m pip install --upgrade pip

REM Instalar las dependencias
IF NOT EXIST "requirements.txt" (
    echo.
    echo ERROR: No se encontro el archivo requirements.txt
    echo.
    exit /b 1
)
echo Instalando dependencias
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

REM Confirmacion
echo Entorno virtual activado y dependencias instaladas
pause