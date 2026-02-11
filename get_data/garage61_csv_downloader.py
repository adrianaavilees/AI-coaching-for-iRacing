#!/usr/bin/env python3
"""
Script para descargar CSVs individuales de vueltas desde garage61.net
Cada vuelta se descarga como un archivo CSV separado

Instalación:
pip install selenium webdriver-manager

Uso:
python garage61_csv_downloader.py
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import os
from pathlib import Path

class Garage61CSVDownloader:
    """Descarga CSVs individuales de vueltas desde garage61.net"""
    
    def __init__(self, download_folder="./garage61_csvs", headless=False):
        """
        Inicializa el descargador
        
        Args:
            download_folder: Carpeta donde se guardarán los CSVs
            headless: Si True, ejecuta Chrome sin interfaz gráfica
        """
        # Crear carpeta de descargas
        self.download_folder = os.path.abspath(download_folder)
        os.makedirs(self.download_folder, exist_ok=True)
        
        print(f"📁 Carpeta de descargas: {self.download_folder}")
        
        # Configurar Chrome
        chrome_options = Options()
        
        # IMPORTANTE: Configurar carpeta de descargas
        prefs = {
            "download.default_directory": self.download_folder,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        # Opciones adicionales
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        
        if headless:
            chrome_options.add_argument('--headless')
        
        print("🌐 Iniciando navegador Chrome...")
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        self.wait = WebDriverWait(self.driver, 20)
        
        print("✓ Navegador iniciado\n")
    
    def hacer_login(self, usuario=None, password=None):
        """
        Realiza el login en garage61.net
        
        Args:
            usuario: Email o nombre de usuario (None para input manual)
            password: Contraseña (None para input manual)
        
        Returns:
            True si el login fue exitoso, False en caso contrario
        """
        try:
            print("🔐 Iniciando proceso de login...")
            
            # Ir a la página principal primero
            self.driver.get("https://garage61.net")
            time.sleep(2)
            
            # Buscar el botón de login/sign in
            # Estos son los selectores más comunes
            selectores_login_btn = [
                "//a[contains(text(), 'Sign In') or contains(text(), 'Login') or contains(text(), 'Log In')]",
                "//button[contains(text(), 'Sign In') or contains(text(), 'Login')]",
                "//a[contains(@href, 'login') or contains(@href, 'signin')]",
                "//a[@class='login' or @id='login']",
            ]
            
            login_btn = None
            for selector in selectores_login_btn:
                try:
                    login_btn = self.driver.find_element(By.XPATH, selector)
                    if login_btn and login_btn.is_displayed():
                        print(f"   ✓ Botón de login encontrado")
                        login_btn.click()
                        time.sleep(2)
                        break
                except:
                    continue
            
            # Si no se encontró el botón, puede que ya estemos en la página de login
            # o que la URL sea diferente
            if not login_btn:
                print("   ⚠ No se encontró botón de login, intentando URL directa...")
                # Intentar URLs comunes de login
                urls_login = [
                    "https://garage61.net/login",
                    "https://garage61.net/signin",
                    "https://garage61.net/auth/login",
                ]
                for url in urls_login:
                    try:
                        self.driver.get(url)
                        time.sleep(2)
                        # Verificar si hay campos de login
                        campos = self.driver.find_elements(By.XPATH, "//input[@type='email' or @type='text' or @type='password']")
                        if len(campos) >= 2:
                            print(f"   ✓ Página de login encontrada: {url}")
                            break
                    except:
                        continue
            
            # Guardar screenshot de la página de login
            self.driver.save_screenshot(os.path.join(self.download_folder, 'login_page.png'))
            print("   📸 Screenshot guardado: login_page.png")
            
            # Si no se proporcionaron credenciales, pedirlas
            if usuario is None:
                print("\n" + "="*70)
                usuario = input("   Introduce tu email/usuario: ").strip()
            
            if password is None:
                import getpass
                password = getpass.getpass("   Introduce tu contraseña: ").strip()
                print()
            
            # Buscar campos de usuario y contraseña
            selectores_usuario = [
                "//input[@type='email']",
                "//input[@type='text' and (contains(@name, 'email') or contains(@name, 'user'))]",
                "//input[@id='email' or @id='username' or @name='email' or @name='username']",
                "//input[@placeholder='Email' or @placeholder='Username']",
            ]
            
            selectores_password = [
                "//input[@type='password']",
                "//input[@id='password' or @name='password']",
            ]
            
            # Encontrar campo de usuario
            campo_usuario = None
            for selector in selectores_usuario:
                try:
                    campo_usuario = self.driver.find_element(By.XPATH, selector)
                    if campo_usuario:
                        break
                except:
                    continue
            
            # Encontrar campo de contraseña
            campo_password = None
            for selector in selectores_password:
                try:
                    campo_password = self.driver.find_element(By.XPATH, selector)
                    if campo_password:
                        break
                except:
                    continue
            
            if not campo_usuario or not campo_password:
                print("   ✗ No se encontraron los campos de login")
                print("   Por favor, haz login manualmente en la ventana del navegador")
                print("   Presiona ENTER cuando hayas iniciado sesión...")
                input()
                return True
            
            # Rellenar campos
            print("   📝 Rellenando credenciales...")
            campo_usuario.clear()
            campo_usuario.send_keys(usuario)
            time.sleep(0.5)
            
            campo_password.clear()
            campo_password.send_keys(password)
            time.sleep(0.5)
            
            # Buscar botón de submit
            selectores_submit = [
                "//button[@type='submit']",
                "//input[@type='submit']",
                "//button[contains(text(), 'Sign In') or contains(text(), 'Login') or contains(text(), 'Log In')]",
                "//button[contains(@class, 'submit') or contains(@class, 'login')]",
            ]
            
            boton_submit = None
            for selector in selectores_submit:
                try:
                    boton_submit = self.driver.find_element(By.XPATH, selector)
                    if boton_submit and boton_submit.is_displayed():
                        break
                except:
                    continue
            
            if boton_submit:
                print("   🔓 Iniciando sesión...")
                boton_submit.click()
            else:
                # Intentar submit con Enter
                from selenium.webdriver.common.keys import Keys
                campo_password.send_keys(Keys.RETURN)
            
            # Esperar a que se complete el login
            time.sleep(5)
            
            # Verificar si el login fue exitoso
            # (comprobando si ya no estamos en la página de login)
            url_actual = self.driver.current_url
            if 'login' not in url_actual.lower() and 'signin' not in url_actual.lower():
                print("   ✓ Login exitoso!\n")
                return True
            else:
                print("   ⚠ No se pudo verificar el login automáticamente")
                print("   Si ves algún error en el navegador, corrige manualmente")
                print("   Presiona ENTER cuando hayas iniciado sesión correctamente...")
                input()
                return True
            
        except Exception as e:
            print(f"   ✗ Error durante el login: {e}")
            print("\n   Por favor, haz login manualmente en la ventana del navegador")
            print("   Presiona ENTER cuando hayas iniciado sesión...")
            input()
            return True
    
    def acceder_pagina(self, url="https://garage61.net/app/laps/497/155;a=-1;g=2"):
        """Accede a la página de garage61"""
        try:
            print(f"🔗 Accediendo a: {url}")
            self.driver.get(url)
            
            # Esperar a que la página cargue
            time.sleep(5)
            
            # Guardar screenshot
            self.driver.save_screenshot(os.path.join(self.download_folder, 'pagina_inicial.png'))
            print("✓ Página cargada\n")
            
            return True
            
        except Exception as e:
            print(f"✗ Error al acceder a la página: {e}")
            return False
    
    def aplicar_filtros(self, coche="Ferrari 296 GT3", circuito="Miami"):
        """
        Aplica los filtros de coche y circuito
        
        NOTA: Los selectores pueden variar según la estructura de la página.
        Si no funcionan automáticamente, necesitarás inspeccionarlos manualmente.
        """
        try:
            print(f"🔍 Aplicando filtros:")
            print(f"   Coche: {coche}")
            print(f"   Circuito: {circuito}")
            
            # Lista de posibles selectores para el filtro de coche
            selectores_coche = [
                "//select[@name='car' or @id='car' or contains(@class, 'car')]",
                "//select[contains(@class, 'car-select')]",
                "//input[@placeholder='Car' or @placeholder='Coche']",
                "//div[contains(@class, 'filter')]//select[1]",
            ]
            
            # Intentar encontrar y usar el filtro de coche
            filtro_aplicado = False
            for selector in selectores_coche:
                try:
                    elemento = self.driver.find_element(By.XPATH, selector)
                    
                    if elemento.tag_name == 'select':
                        from selenium.webdriver.support.select import Select
                        select = Select(elemento)
                        
                        # Buscar la opción que contiene el nombre del coche
                        for option in select.options:
                            if coche.lower() in option.text.lower() or "296" in option.text:
                                select.select_by_visible_text(option.text)
                                print(f"   ✓ Filtro de coche aplicado: {option.text}")
                                filtro_aplicado = True
                                time.sleep(2)
                                break
                    
                    elif elemento.tag_name == 'input':
                        elemento.clear()
                        elemento.send_keys(coche)
                        print(f"   ✓ Filtro de coche aplicado")
                        filtro_aplicado = True
                        time.sleep(2)
                    
                    if filtro_aplicado:
                        break
                        
                except:
                    continue
            
            if not filtro_aplicado:
                print("   ⚠ No se pudo aplicar el filtro de coche automáticamente")
                print("   La página puede ya estar filtrada o usar otra estructura")
            
            # Similar para circuito (si es necesario)
            # En tu caso, la URL ya incluye el circuito (497 = Miami)
            # Por lo que puede no ser necesario filtrar manualmente
            
            print()
            return True
            
        except Exception as e:
            print(f"   ✗ Error al aplicar filtros: {e}\n")
            return False
    
    def encontrar_botones_csv(self):
        """
        Encuentra todos los botones de exportar CSV en la página
        
        Retorna una lista de elementos clickeables
        """
        try:
            print("🔎 Buscando botones de exportar CSV...")
            
            # Posibles selectores para botones de exportar CSV
            # Ajusta estos según la estructura real de garage61.net
            selectores_botones = [
                "//button[contains(text(), 'CSV') or contains(text(), 'Export')]",
                "//a[contains(text(), 'CSV') or contains(text(), 'Export')]",
                "//button[contains(@class, 'export') or contains(@class, 'csv')]",
                "//a[contains(@class, 'export') or contains(@class, 'csv')]",
                "//i[contains(@class, 'download')]/..",  # Icono de descarga
                "//*[@title='Export' or @title='Download' or @title='CSV']",
                "//button[contains(@aria-label, 'export')]",
            ]
            
            botones = []
            for selector in selectores_botones:
                try:
                    elementos = self.driver.find_elements(By.XPATH, selector)
                    if elementos:
                        botones.extend(elementos)
                except:
                    continue
            
            # Eliminar duplicados
            botones_unicos = list(set(botones))
            
            print(f"   ✓ Encontrados {len(botones_unicos)} botones de exportar\n")
            return botones_unicos
            
        except Exception as e:
            print(f"   ✗ Error al buscar botones: {e}\n")
            return []
    
    def encontrar_filas_vueltas(self):
        """
        Encuentra todas las filas de vueltas en la tabla
        Cada fila debería tener un botón de exportar CSV
        """
        try:
            print("🔎 Buscando filas de vueltas en la tabla...")
            
            # Esperar a que la tabla cargue
            time.sleep(3)
            
            # Buscar la tabla de vueltas
            selectores_tabla = [
                "//table[contains(@class, 'laps')]",
                "//table[contains(@class, 'results')]",
                "//table",  # Última opción: cualquier tabla
            ]
            
            tabla = None
            for selector in selectores_tabla:
                try:
                    tabla = self.driver.find_element(By.XPATH, selector)
                    if tabla:
                        break
                except:
                    continue
            
            if not tabla:
                print("   ⚠ No se encontró ninguna tabla")
                return []
            
            # Buscar todas las filas de datos (tbody > tr)
            filas = tabla.find_elements(By.XPATH, ".//tbody/tr | .//tr[td]")
            
            print(f"   ✓ Encontradas {len(filas)} filas de vueltas\n")
            return filas
            
        except Exception as e:
            print(f"   ✗ Error al buscar filas: {e}\n")
            return []
    
    def descargar_csv_de_fila(self, fila, numero):
        """
        Descarga el CSV de una fila específica
        
        Args:
            fila: Elemento WebDriver de la fila
            numero: Número de vuelta (para logging)
        """
        try:
            # Buscar el botón de exportar CSV dentro de la fila
            selectores_boton = [
                ".//button[contains(text(), 'CSV') or contains(@class, 'export')]",
                ".//a[contains(text(), 'CSV') or contains(@class, 'export')]",
                ".//button[contains(@title, 'Export') or contains(@title, 'CSV')]",
                ".//a[contains(@title, 'Export') or contains(@title, 'CSV')]",
                ".//i[contains(@class, 'download')]/..",
                ".//*[contains(@class, 'download')]",
            ]
            
            boton = None
            for selector in selectores_boton:
                try:
                    boton = fila.find_element(By.XPATH, selector)
                    if boton and boton.is_displayed():
                        break
                except:
                    continue
            
            if not boton:
                print(f"   ⚠ Vuelta {numero}: No se encontró botón de exportar")
                return False
            
            # Hacer scroll hasta el botón para asegurar que es visible
            self.driver.execute_script("arguments[0].scrollIntoView(true);", boton)
            time.sleep(0.5)
            
            # Obtener info de la vuelta antes de descargar (opcional)
            try:
                celdas = fila.find_elements(By.TAG_NAME, 'td')
                info_vuelta = " | ".join([c.text.strip() for c in celdas[:3]])  # Primeras 3 columnas
            except:
                info_vuelta = f"Vuelta {numero}"
            
            # Contar archivos antes de descargar
            archivos_antes = set(os.listdir(self.download_folder))
            
            # Click en el botón
            try:
                boton.click()
            except:
                # Si el click normal falla, usar JavaScript
                self.driver.execute_script("arguments[0].click();", boton)
            
            print(f"   ✓ Vuelta {numero}: Descargando... ({info_vuelta})")
            
            # Esperar a que se descargue el archivo
            tiempo_espera = 0
            max_espera = 10
            while tiempo_espera < max_espera:
                archivos_despues = set(os.listdir(self.download_folder))
                archivos_nuevos = archivos_despues - archivos_antes
                
                # Verificar que no haya archivos .crdownload (descarga en progreso)
                archivos_descargando = [f for f in archivos_nuevos if f.endswith('.crdownload')]
                
                if archivos_nuevos and not archivos_descargando:
                    # Descarga completada
                    archivo_nuevo = list(archivos_nuevos)[0]
                    print(f"      → {archivo_nuevo}")
                    return True
                
                time.sleep(0.5)
                tiempo_espera += 0.5
            
            print(f"   ⚠ Vuelta {numero}: Timeout esperando descarga")
            return False
            
        except Exception as e:
            print(f"   ✗ Vuelta {numero}: Error - {e}")
            return False
    
    def descargar_todas_vueltas(self, max_vueltas=100):
        """
        Descarga los CSVs de todas las vueltas encontradas
        
        Args:
            max_vueltas: Número máximo de vueltas a descargar
        """
        print("="*70)
        print("DESCARGANDO CSVs DE VUELTAS")
        print("="*70 + "\n")
        
        # Encontrar todas las filas de vueltas
        filas = self.encontrar_filas_vueltas()
        
        if not filas:
            print("⚠ No se encontraron vueltas para descargar")
            return 0
        
        # Limitar al máximo especificado
        filas = filas[:max_vueltas]
        total_filas = len(filas)
        
        print(f"📊 Total de vueltas a descargar: {total_filas}")
        print(f"⏱️  Tiempo estimado: {total_filas * 2} segundos\n")
        
        descargas_exitosas = 0
        
        for i, fila in enumerate(filas, 1):
            if self.descargar_csv_de_fila(fila, i):
                descargas_exitosas += 1
            
            # Pequeña pausa entre descargas para no saturar
            if i < total_filas:
                time.sleep(1)
        
        print("\n" + "="*70)
        print(f"✓ DESCARGA COMPLETADA")
        print("="*70)
        print(f"   Exitosas: {descargas_exitosas}/{total_filas}")
        print(f"   Carpeta: {self.download_folder}")
        print("="*70 + "\n")
        
        return descargas_exitosas
    
    def cerrar(self):
        """Cierra el navegador"""
        if self.driver:
            self.driver.quit()
            print("✓ Navegador cerrado")


def main():
    """Función principal"""
    print("\n" + "="*70)
    print(" "*15 + "DESCARGADOR DE CSVs - GARAGE61.NET")
    print("="*70)
    print("\nConfiguración:")
    print("  🏎️  Coche: Ferrari 296 GT3")
    print("  🏁 Circuito: Miami International Autodrome (Grand Prix)")
    print("  📊 Máximo: 100 vueltas")
    print("  📁 Formato: Un CSV por vuelta")
    print("="*70 + "\n")
    
    # Preguntar si quiere headless mode
    print("¿Ejecutar en modo headless (sin ventana visible)?")
    print("  Recomendado: NO (para la primera vez)")
    headless_input = input("Headless mode? (s/N): ").strip().lower()
    headless = headless_input in ['s', 'y', 'yes', 'si', 'sí']
    print()
    
    # Preguntar por credenciales (opcional)
    print("Credenciales de Garage61.net")
    print("(Déjalo en blanco para introducirlas después interactivamente)")
    usuario = input("Email/Usuario: ").strip() or None
    
    if usuario:
        import getpass
        password = getpass.getpass("Contraseña: ").strip() or None
    else:
        password = None
    
    print()
    
    downloader = None
    
    try:
        # Crear downloader
        downloader = Garage61CSVDownloader(
            download_folder="./ferrari_296_miami_csvs",
            headless=headless
        )
        
        # PASO 1: Hacer login
        if not downloader.hacer_login(usuario=usuario, password=password):
            print("❌ Login fallido")
            return
        
        # PASO 2: Acceder a la página de vueltas
        if not downloader.acceder_pagina():
            print("❌ No se pudo acceder a la página")
            return
        
        # PASO 3: Aplicar filtros (puede no ser necesario si la URL ya está filtrada)
        downloader.aplicar_filtros(coche="Ferrari 296 GT3", circuito="Miami")
        
        # Esperar un momento para que se apliquen los filtros
        time.sleep(3)
        
        # PASO 4: Descargar todas las vueltas
        descargas = downloader.descargar_todas_vueltas(max_vueltas=100)
        
        if descargas == 0:
            print("\n⚠️  NOTA IMPORTANTE:")
            print("="*70)
            print("Si no se descargó ningún archivo, puede deberse a:")
            print("  1. Los selectores de botones necesitan ajustarse")
            print("  2. La página requiere interacción manual primero")
            print("  3. Los filtros no se aplicaron correctamente")
            print("\nRevisa:")
            print(f"  - Screenshot guardado en: {downloader.download_folder}/pagina_inicial.png")
            print("  - Ejecuta el script sin headless para ver qué pasa")
            print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Descarga interrumpida por el usuario")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if downloader:
            downloader.cerrar()


if __name__ == "__main__":
    main()