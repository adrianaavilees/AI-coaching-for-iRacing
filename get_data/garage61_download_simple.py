#!/usr/bin/env python3
"""
DESCARGADOR SIMPLIFICADO DE CSVs - GARAGE61.NET
Versión con manejo automático de credenciales

Instalación:
pip install selenium webdriver-manager

Configuración:
1. Edita credentials.py con tus credenciales, O
2. Deja credentials.py vacío y el script te las pedirá

Uso:
python garage61_download_simple.py
"""

from get_data.garage61_csv_downloader import Garage61CSVDownloader
import time

# Intentar cargar credenciales desde el archivo
try:
    from get_data.credentials import GARAGE61_EMAIL, GARAGE61_PASSWORD
except ImportError:
    GARAGE61_EMAIL = None
    GARAGE61_PASSWORD = None
    print("⚠️  No se encontró credentials.py, se pedirán las credenciales interactivamente")

def main():
    print("\n" + "="*70)
    print(" "*10 + "DESCARGADOR DE CSVs - GARAGE61.NET (Simplificado)")
    print("="*70)
    print("\n📋 Configuración predefinida:")
    print("  🏎️  Coche: Ferrari 296 GT3")
    print("  🏁 Circuito: Miami International Autodrome")
    print("  📊 Vueltas: 100 (máximo)")
    print("  📁 Carpeta: ./ferrari_296_miami_csvs/")
    print("="*70 + "\n")
    
    downloader = None
    
    try:
        # Crear downloader (sin headless para ver el progreso)
        print("🚀 Iniciando descargador...\n")
        downloader = Garage61CSVDownloader(
            download_folder="./ferrari_296_miami_csvs",
            headless=False  # Cambiar a True para ejecutar sin ventana
        )
        
        # Hacer login
        print("PASO 1/4: Login en Garage61.net")
        print("-" * 70)
        if not downloader.hacer_login(usuario=GARAGE61_EMAIL, password=GARAGE61_PASSWORD):
            print("❌ No se pudo completar el login")
            return
        
        # Acceder a la página de vueltas
        print("\nPASO 2/4: Accediendo a la página de vueltas")
        print("-" * 70)
        if not downloader.acceder_pagina("https://garage61.net/app/laps/497/155;a=-1;g=2"):
            print("❌ No se pudo acceder a la página")
            return
        
        # Aplicar filtros
        print("\nPASO 3/4: Aplicando filtros")
        print("-" * 70)
        downloader.aplicar_filtros(coche="Ferrari 296 GT3", circuito="Miami")
        time.sleep(3)
        
        # Descargar CSVs
        print("\nPASO 4/4: Descargando CSVs")
        print("-" * 70)
        descargas = downloader.descargar_todas_vueltas(max_vueltas=100)
        
        # Resumen final
        print("\n" + "="*70)
        if descargas > 0:
            print("✅ DESCARGA COMPLETADA CON ÉXITO")
            print("="*70)
            print(f"  📊 Archivos descargados: {descargas}")
            print(f"  📁 Ubicación: ./ferrari_296_miami_csvs/")
        else:
            print("⚠️  NO SE DESCARGARON ARCHIVOS")
            print("="*70)
            print("\n💡 Posibles causas:")
            print("  1. Los selectores de botones necesitan ajustarse")
            print("  2. La estructura de la página ha cambiado")
            print("  3. No hay vueltas disponibles con esos filtros")
            print("\n🔧 Soluciones:")
            print("  1. Ejecuta: python garage61_inspector.py")
            print("  2. Revisa el screenshot: ./ferrari_296_miami_csvs/pagina_inicial.png")
            print("  3. Verifica manualmente que la página muestra vueltas")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Descarga interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if downloader:
            print("\n🧹 Cerrando navegador...")
            downloader.cerrar()
            print("✓ Proceso finalizado\n")

if __name__ == "__main__":
    main()