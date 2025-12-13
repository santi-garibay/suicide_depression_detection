#!/bin/bash

# Script para iniciar Jupyter Notebook con el kernel correcto

# Activar el entorno virtual
source venv/bin/activate

# Iniciar Jupyter Lab
echo "🚀 Iniciando Jupyter Lab con Python 3.14.2..."
echo "📍 Directorio: $(pwd)"
echo "🐍 Python: $(python --version)"
echo ""

jupyter lab
