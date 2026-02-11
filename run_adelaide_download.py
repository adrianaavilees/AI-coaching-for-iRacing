#!/usr/bin/env python3
"""
Lanzador rápido para descargar CSVs de Adelaide / Ferrari 296 GT3
Ejecutar desde la raíz del proyecto:
    python run_adelaide_download.py
"""

import sys
import os

# Asegurar que el directorio raíz del proyecto está en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from get_data.garage61_adelaide_downloader import main

if __name__ == "__main__":
    main()
