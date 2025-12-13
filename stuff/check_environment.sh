#!/bin/bash

echo "═══════════════════════════════════════════════"
echo "🔍 Diagnóstico del entorno Python y Jupyter"
echo "═══════════════════════════════════════════════"
echo ""

# Activar entorno virtual
source venv/bin/activate

echo "✓ Entorno virtual activado"
echo ""

echo "📍 Información del entorno:"
echo "  - Directorio actual: $(pwd)"
echo "  - Python ejecutable: $(which python)"
echo "  - Python versión: $(python --version)"
echo "  - Jupyter ejecutable: $(which jupyter)"
echo ""

echo "📦 Paquetes instalados:"
python -m pip list | grep -i jupyter
python -m pip list | grep -i ipykernel
echo ""

echo "🎯 Kernels disponibles:"
jupyter kernelspec list
echo ""

echo "═══════════════════════════════════════════════"
echo "✅ Diagnóstico completado"
echo "═══════════════════════════════════════════════"
