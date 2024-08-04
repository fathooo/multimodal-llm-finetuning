@echo off
REM Verificar la existencia de la variable de entorno %python39%
IF "%python39%"=="" (
    echo La variable de entorno python39 no está configurada.
    set /p python39="Por favor ingresa la ruta completa de Python: "
    setx python39 "%python39%"
) ELSE (
    echo Usando la variable de entorno python39: %python39%
)

REM Verificar si el entorno virtual ya está creado
IF NOT EXIST "venv\Scripts\activate" (
    echo Creando entorno virtual usando %python39%...
    %python39% -m venv venv
) ELSE (
    echo El entorno virtual ya existe.
)

REM Activar el entorno virtual
echo Activando el entorno virtual...
call venv\Scripts\activate

REM Verificar si pip está instalado y actualizarlo
echo Verificando pip...
venv\Scripts\python -m ensurepip --default-pip
venv\Scripts\python -m pip install --upgrade pip

REM Instalar las dependencias
IF NOT EXIST "requirements.txt" (
    echo "ERROR: No se encontró el archivo requirements.txt"
    exit /b 1
)
echo Instalando dependencias...
pip install -r requirements.txt

REM Confirmación
echo Entorno virtual activado y dependencias instaladas.
pause