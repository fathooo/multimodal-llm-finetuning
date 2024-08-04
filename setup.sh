#!/bin/bash

# Verificar la existencia de la variable de entorno PYTHON39
if [ -z "$PYTHON39" ]; then
    echo "La variable de entorno PYTHON39 no está configurada."
    read -p "Por favor ingresa la ruta completa de Python: " PYTHON39
    echo "export PYTHON39=$PYTHON39" >> ~/.bashrc
    source ~/.bashrc
else
    echo "Usando la variable de entorno PYTHON39: $PYTHON39"
fi

# Verificar si el entorno virtual ya está creado
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual usando \$PYTHON39..."
    $PYTHON39 -m venv venv
else
    echo "El entorno virtual ya existe."
fi

# Activar el entorno virtual
echo "Activando el entorno virtual..."
source venv/bin/activate

# Verificación de `pip` e instalación si no está presente
echo "Verificando pip..."
if ! (pip --version); then
    echo "pip no encontrado, instalando pip..."
    $PYTHON39 -m ensurepip
fi

# Instalar las dependencias
echo "Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

# Confirmación
echo "Entorno virtual activado y dependencias instaladas."