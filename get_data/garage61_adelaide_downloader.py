#!/usr/bin/env python3
"""
DESCARGADOR DE CSVs DE TELEMETRÍA - GARAGE61.NET
Adelaide Street Circuit / Ferrari 296 GT3

Estrategia híbrida:
  1. Login por Selenium (maneja el SPA Angular)
  2. Busca IDs de circuito/coche usando la API oficial (/api/v1/tracks, /api/v1/cars)
  3. Navega a la página de vueltas (leaderboard público)
  4. Captura datos de vueltas interceptando las peticiones de red (CDP)
  5. Descarga CSVs de las vueltas con telemetría disponible

Instalación:
  pip install selenium webdriver-manager requests

Configuración:
  Edita credentials.py con tus credenciales de garage61.net

Uso:
  python garage61_adelaide_downloader.py
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
TRACK_SEARCH = "Adelaide"            # Nombre del circuito a buscar
CAR_SEARCH   = "Ferrari 296 GT3"     # Nombre del coche a buscar
MAX_LAPS     = 100                   # Máximo de vueltas a descargar
DOWNLOAD_FOLDER = "./adelaide_ferrari296_csvs"
HEADLESS     = False                 # True = sin ventana visible

# Credenciales (intentar cargar desde credentials.py)
try:
    # Si se ejecuta como módulo desde la raíz
    from get_data.credentials import GARAGE61_EMAIL, GARAGE61_PASSWORD
except ImportError:
    try:
        # Si se ejecuta directamente desde get_data/
        from credentials import GARAGE61_EMAIL, GARAGE61_PASSWORD
    except ImportError:
        GARAGE61_EMAIL = None
        GARAGE61_PASSWORD = None

# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class Garage61Downloader:
    """Descargador de CSVs de telemetría de garage61.net"""

    BASE_URL = "https://garage61.net"

    def __init__(self, download_folder, headless=False):
        self.download_folder = os.path.abspath(download_folder)
        os.makedirs(self.download_folder, exist_ok=True)
        print(f"📁 Carpeta de descargas: {self.download_folder}")

        # ---- Chrome options ------------------------------------------------
        opts = Options()
        prefs = {
            "download.default_directory": self.download_folder,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        opts.add_experimental_option("prefs", prefs)
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1920,1080")

        # Log de rendimiento → captura peticiones de red
        opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        if headless:
            opts.add_argument("--headless=new")

        print("🌐 Iniciando navegador Chrome...")
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts,
        )
        self.driver.set_script_timeout(30)
        self.wait = WebDriverWait(self.driver, 30)

        # Activar monitorización de red CDP
        self.driver.execute_cdp_cmd("Network.enable", {})

        print("✓ Navegador listo\n")

    # -----------------------------------------------------------------------
    #  Helpers de red
    # -----------------------------------------------------------------------
    def _js_fetch_json(self, url):
        """Hacer fetch() JSON desde el navegador (reutiliza cookies de sesión)."""
        script = """
        var callback = arguments[arguments.length - 1];
        fetch(arguments[0], {credentials: 'include',
                             headers: {'Accept': 'application/json'}})
            .then(function(r) {
                if (!r.ok) return callback({ok: false, status: r.status});
                return r.json().then(function(d) { callback({ok: true, data: d}); });
            })
            .catch(function(e) { callback({ok: false, error: String(e)}); });
        """
        try:
            return self.driver.execute_async_script(script, url)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _js_fetch_text(self, url):
        """Hacer fetch() de texto desde el navegador."""
        script = """
        var callback = arguments[arguments.length - 1];
        fetch(arguments[0], {credentials: 'include'})
            .then(function(r) {
                if (!r.ok) return callback({ok: false, status: r.status});
                return r.text().then(function(d) { callback({ok: true, data: d}); });
            })
            .catch(function(e) { callback({ok: false, error: String(e)}); });
        """
        try:
            return self.driver.execute_async_script(script, url)
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _capture_network_responses(self, url_keyword):
        """
        Lee el Performance Log de Chrome para capturar respuestas HTTP
        que contengan url_keyword. Devuelve una lista de dicts con
        {url, body, requestId}.
        """
        captured = []
        try:
            logs = self.driver.get_log("performance")
            for entry in logs:
                try:
                    msg = json.loads(entry["message"])["message"]
                    if msg["method"] != "Network.responseReceived":
                        continue
                    resp_url = msg["params"]["response"]["url"]
                    if url_keyword not in resp_url:
                        continue
                    rid = msg["params"]["requestId"]
                    try:
                        body_resp = self.driver.execute_cdp_cmd(
                            "Network.getResponseBody", {"requestId": rid}
                        )
                        captured.append({
                            "url": resp_url,
                            "body": body_resp.get("body", ""),
                        })
                    except Exception:
                        pass  # respuesta ya purgada del caché
                except (KeyError, json.JSONDecodeError):
                    continue
        except Exception:
            pass
        return captured

    # -----------------------------------------------------------------------
    #  Login
    # -----------------------------------------------------------------------
    def login(self, email=None, password=None):
        """Iniciar sesión en garage61.net."""
        print("🔐 Iniciando login...")
        self.driver.get(f"{self.BASE_URL}/app/laps")
        time.sleep(4)

        # ¿Ya logueado?
        src = self.driver.page_source[:3000].lower()
        if "log in" not in src and "sign in" not in src and "e-mail" not in src:
            print("✓ Sesión activa detectada\n")
            return True

        # Pedir credenciales si no están configuradas
        if not email:
            email = input("   Email de Garage61: ").strip()
        if not password:
            import getpass
            password = getpass.getpass("   Contraseña: ").strip()

        try:
            # Campo email
            email_field = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='email']"))
            )
            email_field.clear()
            email_field.send_keys(email)
            time.sleep(0.3)

            # Campo contraseña
            pwd_field = self.driver.find_element(By.CSS_SELECTOR, "input[type='password']")
            pwd_field.clear()
            pwd_field.send_keys(password)
            time.sleep(0.3)

            # Enviar formulario
            try:
                submit = self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            except Exception:
                # Fallback: cualquier botón visible que contenga "Log"
                buttons = self.driver.find_elements(By.TAG_NAME, "button")
                submit = None
                for btn in buttons:
                    if btn.is_displayed() and ("log" in btn.text.lower() or "sign" in btn.text.lower()):
                        submit = btn
                        break
                if not submit and buttons:
                    submit = buttons[0]

            if submit:
                submit.click()
            else:
                from selenium.webdriver.common.keys import Keys
                pwd_field.send_keys(Keys.RETURN)

            time.sleep(6)

            if "login" not in self.driver.current_url.lower():
                print("✓ Login exitoso\n")
                return True
            else:
                self.driver.save_screenshot(
                    os.path.join(self.download_folder, "login_debug.png")
                )
                print("⚠ No se pudo verificar el login automáticamente")
                print("   Screenshot: login_debug.png")
                resp = input("   ¿Lograste iniciar sesión? (s/N): ").strip().lower()
                return resp in ("s", "si", "sí", "y", "yes")

        except Exception as e:
            print(f"⚠ Error en login automático: {e}")
            print("   Por favor, inicia sesión manualmente en el navegador abierto.")
            input("   Presiona ENTER cuando hayas iniciado sesión... ")
            return True

    # -----------------------------------------------------------------------
    #  Buscar IDs de circuito y coche
    # -----------------------------------------------------------------------
    def find_track_and_car_ids(self, track_search, car_search):
        """Obtener IDs numéricos para circuito y coche usando la API."""
        print(f"🔍 Buscando IDs para '{track_search}' y '{car_search}'...")

        track_id = None
        car_id = None

        # --- Circuito ---
        res = self._js_fetch_json(f"{self.BASE_URL}/api/v1/tracks")
        if res and res.get("ok"):
            tracks = res["data"]
            # Búsqueda exacta primero, luego parcial
            for t in tracks:
                name = t.get("name", "")
                if track_search.lower() == name.lower():
                    track_id = t["id"]
                    print(f"   ✓ Circuito: {name}  (ID: {track_id})")
                    break
            if not track_id:
                for t in tracks:
                    name = t.get("name", "")
                    if track_search.lower() in name.lower():
                        track_id = t["id"]
                        print(f"   ✓ Circuito: {name}  (ID: {track_id})")
                        break
            if not track_id:
                print(f"   ⚠ No se encontró circuito con '{track_search}'")
                print("   Opciones similares:")
                for t in tracks:
                    n = t.get("name", "")
                    if any(w in n.lower() for w in track_search.lower().split()):
                        print(f"      • {n}  (ID: {t['id']})")
        else:
            print(f"   ✗ Error consultando circuitos: {res}")

        # --- Coche ---
        res = self._js_fetch_json(f"{self.BASE_URL}/api/v1/cars")
        if res and res.get("ok"):
            cars = res["data"]
            for c in cars:
                name = c.get("name", "")
                if car_search.lower() == name.lower():
                    car_id = c["id"]
                    print(f"   ✓ Coche:    {name}  (ID: {car_id})")
                    break
            if not car_id:
                for c in cars:
                    name = c.get("name", "")
                    if car_search.lower() in name.lower():
                        car_id = c["id"]
                        print(f"   ✓ Coche:    {name}  (ID: {car_id})")
                        break
            if not car_id:
                print(f"   ⚠ No se encontró coche con '{car_search}'")
                for c in cars:
                    n = c.get("name", "")
                    if any(w in n.lower() for w in car_search.lower().split()):
                        print(f"      • {n}  (ID: {c['id']})")
        else:
            print(f"   ✗ Error consultando coches: {res}")

        print()
        return track_id, car_id

    # -----------------------------------------------------------------------
    #  Navegar a la página de vueltas
    # -----------------------------------------------------------------------
    def navigate_to_laps(self, track_id, car_id):
        """Ir a la leaderboard de vueltas."""
        # a=-1 → temporada actual ; g=2 → agrupar: ninguno (todas las vueltas)
        url = f"{self.BASE_URL}/app/laps/{track_id}/{car_id};a=-1;g=2"
        print(f"🏁 Navegando al leaderboard:\n   {url}")

        # Limpiar logs previos de rendimiento
        try:
            self.driver.get_log("performance")
        except Exception:
            pass

        self.driver.get(url)
        time.sleep(8)  # Esperar al SPA Angular

        self.driver.save_screenshot(
            os.path.join(self.download_folder, "laps_page.png")
        )
        print("   📸 Screenshot: laps_page.png\n")

    # -----------------------------------------------------------------------
    #  Extraer datos de vueltas
    # -----------------------------------------------------------------------
    def extract_laps(self):
        """
        Extraer información de las vueltas disponibles.
        Intenta 3 estrategias en orden de fiabilidad:
          1. Captura de respuestas de red (CDP)
          2. JavaScript fetch a la API desde el navegador
          3. Extracción del DOM
        """
        print("🔎 Extrayendo datos de vueltas...\n")

        # --- Estrategia 1: Interceptar respuestas de red --------------------
        laps = self._strategy_network_capture()
        if laps:
            print(f"   ✓ [Red CDP] {len(laps)} vueltas capturadas\n")
            return laps

        # --- Estrategia 2: Re-cargar y capturar ----------------------------
        print("   → Recargando página para capturar peticiones...")
        try:
            self.driver.get_log("performance")
        except Exception:
            pass
        self.driver.refresh()
        time.sleep(8)
        laps = self._strategy_network_capture()
        if laps:
            print(f"   ✓ [Red CDP reload] {len(laps)} vueltas capturadas\n")
            return laps

        # --- Estrategia 3: DOM scraping ------------------------------------
        laps = self._strategy_dom_scraping()
        if laps:
            print(f"   ✓ [DOM] {len(laps)} vueltas encontradas\n")
            return laps

        print("   ⚠ No se encontraron vueltas automáticamente")
        self._save_debug_info()
        return []

    def _strategy_network_capture(self):
        """Capturar datos de vueltas desde las respuestas de red (CDP)."""
        # La app Angular hace llamadas fetch/XHR para cargar los datos de vueltas
        responses = self._capture_network_responses("laps")
        all_laps = []

        for resp in responses:
            try:
                data = json.loads(resp["body"])
            except (json.JSONDecodeError, TypeError):
                continue

            items = []
            # Formato API: {total: N, items: [...]}
            if isinstance(data, dict) and "items" in data:
                items = data["items"]
            elif isinstance(data, list):
                items = data

            for item in items:
                if isinstance(item, dict) and "lapTime" in item and "id" in item:
                    all_laps.append(item)

        return all_laps

    def _strategy_dom_scraping(self):
        """Extraer IDs de vuelta inspeccionando el DOM."""
        laps = []
        try:
            # Buscar links que apunten a análisis o contengan lap IDs
            selectors = [
                "a[href*='analysis']",
                "a[href*='analyze']",
                "[routerlink*='analysis']",
                "a[href*='/laps/']",
                "tr[class*='lap']",
                ".lap-row",
                "[class*='lap-item']",
                "[class*='leaderboard'] tr",
                "table tbody tr",
            ]

            for sel in selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, sel)
                if not elements:
                    continue

                for el in elements:
                    href = el.get_attribute("href") or ""
                    routerlink = el.get_attribute("routerlink") or ""
                    combined = href + " " + routerlink

                    # Buscar IDs en los links
                    id_match = re.search(r'([a-f0-9]{8,})', combined)
                    if id_match:
                        laps.append({
                            "id": id_match.group(1),
                            "_source": "dom",
                            "_href": href,
                        })

                if laps:
                    break

            # Deduplicar
            seen = set()
            unique = []
            for lap in laps:
                lid = lap.get("id")
                if lid and lid not in seen:
                    seen.add(lid)
                    unique.append(lap)
            return unique

        except Exception as e:
            print(f"   DOM scraping error: {e}")
            return []

    def _save_debug_info(self):
        """Guardar información de debug para diagnóstico."""
        try:
            # HTML de la página
            html_path = os.path.join(self.download_folder, "debug_page.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            print(f"   📄 HTML guardado: debug_page.html")

            # Screenshot
            self.driver.save_screenshot(
                os.path.join(self.download_folder, "debug_screenshot.png")
            )
            print(f"   📸 Screenshot: debug_screenshot.png")

            # Listar elementos interactivos
            print("\n   — Elementos interactivos en la página —")
            for tag in ("a", "button"):
                els = self.driver.find_elements(By.TAG_NAME, tag)
                visible = [e for e in els if e.is_displayed()]
                print(f"   <{tag}>: {len(visible)} visibles / {len(els)} totales")
                for e in visible[:10]:
                    txt = e.text.strip()[:60]
                    href = (e.get_attribute("href") or "")[:80]
                    if txt or href:
                        print(f"      '{txt}' → {href}")
        except Exception as e:
            print(f"   Error guardando debug info: {e}")

    # -----------------------------------------------------------------------
    #  Descarga de CSV
    # -----------------------------------------------------------------------
    def download_csv(self, lap_id, driver_name="", lap_time=0):
        """
        Descargar CSV de telemetría para una vuelta.
        Intenta:
          A) fetch() al endpoint /api/v1/laps/{id}/csv desde el navegador
          B) Navegación directa al URL de descarga
        """
        # Nombre de archivo limpio
        safe_driver = re.sub(r'[\\/*?:"<>|]', '_', str(driver_name))
        if lap_time:
            time_str = f"{float(lap_time):.3f}s"
        else:
            time_str = "unknown"
        filename = f"{safe_driver}_{time_str}_{lap_id[:8]}.csv" if safe_driver else f"lap_{lap_id[:12]}.csv"
        filepath = os.path.join(self.download_folder, filename)

        # No re-descargar
        if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
            return True, filename

        csv_url = f"{self.BASE_URL}/api/v1/laps/{lap_id}/csv"

        # --- Intento A: fetch() desde el navegador --------------------------
        result = self._js_fetch_text(csv_url)
        if result and result.get("ok"):
            content = result["data"]
            # Verificar que es CSV válido (tiene comas y suficiente contenido)
            if content and "," in content and len(content) > 200:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                return True, filename

        # --- Intento B: navegar directamente al URL -------------------------
        try:
            # Guardar URL actual
            current_url = self.driver.current_url
            # Abrir en nueva pestaña
            self.driver.execute_script(f"window.open('{csv_url}', '_blank');")
            time.sleep(2)

            # Cambiar a la nueva pestaña
            handles = self.driver.window_handles
            if len(handles) > 1:
                self.driver.switch_to.window(handles[-1])
                time.sleep(1)
                # Si la página cargó contenido CSV, lo capturamos
                page_text = self.driver.page_source
                # Cerrar pestaña y volver
                self.driver.close()
                self.driver.switch_to.window(handles[0])

                # Si el texto parece CSV (no una página HTML de error)
                if page_text and "<html" not in page_text[:200].lower() and "," in page_text:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(page_text)
                    return True, filename

            # Comprobar si se descargó un archivo
            time.sleep(2)
            new_files = [
                f for f in os.listdir(self.download_folder)
                if f.endswith(".csv") and not f.startswith("debug")
            ]
            if new_files:
                return True, new_files[-1]

        except Exception:
            # Asegurar que volvemos a la ventana original
            try:
                handles = self.driver.window_handles
                self.driver.switch_to.window(handles[0])
            except Exception:
                pass

        return False, filename

    # -----------------------------------------------------------------------
    #  Descarga via navegación UI (fallback)
    # -----------------------------------------------------------------------
    def download_via_ui(self, lap_element_or_url):
        """
        Fallback: descargar CSV navegando por la interfaz de usuario.
        Si la API no funciona, intentamos clickar en la vuelta y buscar
        el botón de CSV en la página de análisis.
        """
        try:
            if isinstance(lap_element_or_url, str):
                self.driver.get(lap_element_or_url)
            else:
                # Click en el elemento
                self.driver.execute_script("arguments[0].click();", lap_element_or_url)

            time.sleep(5)

            # Buscar botón de descarga CSV en la página de análisis
            csv_selectors = [
                "button[title*='CSV']",
                "a[title*='CSV']",
                "button[aria-label*='CSV']",
                "a[aria-label*='CSV']",
                "[class*='download']",
                "[class*='export']",
                "button[title*='ownload']",
                "a[title*='ownload']",
                # Material icons
                "button mat-icon",
                "button .material-icons",
            ]

            for sel in csv_selectors:
                try:
                    btn = self.driver.find_element(By.CSS_SELECTOR, sel)
                    if btn.is_displayed():
                        btn.click()
                        time.sleep(3)
                        return True
                except Exception:
                    continue

            return False

        except Exception as e:
            print(f"   UI download error: {e}")
            return False

    # -----------------------------------------------------------------------
    #  Flujo principal
    # -----------------------------------------------------------------------
    def run(self, track_search, car_search, max_laps=100):
        """Ejecutar el proceso completo de descarga."""
        # 1. Buscar IDs
        track_id, car_id = self.find_track_and_car_ids(track_search, car_search)

        if not track_id:
            print("⚠ No se encontró el circuito.")
            track_id = input("   Introduce el Track ID manualmente (o ENTER para cancelar): ").strip()
            if not track_id:
                return 0

        if not car_id:
            print("⚠ No se encontró el coche.")
            car_id = input("   Introduce el Car ID manualmente (o ENTER para cancelar): ").strip()
            if not car_id:
                return 0

        # 2. Navegar a la página de vueltas
        self.navigate_to_laps(track_id, car_id)

        # 3. Extraer datos de vueltas
        laps = self.extract_laps()

        if not laps:
            print("\n💡 SUGERENCIAS:")
            print("   1. Revisa laps_page.png para verificar que la página cargó correctamente")
            print("   2. Asegúrate de que hay vueltas disponibles para este circuito/coche")
            print("   3. Prueba a cambiar g=2 por g=0 en la URL (diferente agrupación)")
            print("   4. Revisa debug_page.html para ver la estructura de la página")
            print()

            # Ofrecer modo interactivo
            print("¿Quieres intentar el modo interactivo? (s/N)")
            if input().strip().lower() in ("s", "si", "sí"):
                return self._interactive_mode(max_laps)
            return 0

        # 4. Descargar CSVs
        print(f"📊 Descargando CSVs de hasta {max_laps} vueltas...\n")
        print(f"{'Nº':>4}  {'Estado':<6}  {'Piloto':<30}  {'Tiempo':<12}  {'Archivo'}")
        print("-" * 90)

        downloaded = 0
        skipped = 0
        errors = 0

        for i, lap in enumerate(laps):
            if downloaded >= max_laps:
                break

            lap_id = lap.get("id")
            if not lap_id:
                continue

            # Info del piloto y tiempo
            driver_info = lap.get("driver") or {}
            driver_name = driver_info.get("name", "Desconocido")
            lap_time = lap.get("lapTime", 0)

            # ¿Telemetría visible?
            can_view = lap.get("canViewTelemetry", None)
            if can_view is False:
                skipped += 1
                print(f"{i+1:>4}  {'⛔':>6}  {driver_name:<30}  {lap_time:<12}  Telemetría bloqueada")
                continue

            # Intentar descargar
            success, filename = self.download_csv(lap_id, driver_name, lap_time)

            if success:
                downloaded += 1
                print(f"{i+1:>4}  {'✓':>6}  {driver_name:<30}  {lap_time:<12.3f}  {filename}")
            else:
                errors += 1
                print(f"{i+1:>4}  {'✗':>6}  {driver_name:<30}  {lap_time:<12}  Error descarga")

            # Pausa entre descargas (evitar saturar)
            time.sleep(0.5)

        # Resumen
        print("\n" + "=" * 70)
        print("✅ DESCARGA COMPLETADA")
        print("=" * 70)
        print(f"   ✓ Descargados:  {downloaded}")
        print(f"   ⛔ Bloqueados:   {skipped}")
        print(f"   ✗ Errores:      {errors}")
        print(f"   📁 Carpeta:      {self.download_folder}")
        print("=" * 70 + "\n")

        return downloaded

    def _interactive_mode(self, max_laps):
        """
        Modo interactivo: el usuario navega el sitio y proporciona
        lap IDs para descargar.
        """
        print("\n📋 MODO INTERACTIVO")
        print("=" * 50)
        print("   Navega la web de Garage61 en el navegador abierto.")
        print("   Cuando encuentres una vuelta que quieras descargar,")
        print("   copia su Lap ID y pégalo aquí.")
        print()
        print("   Para encontrar el Lap ID:")
        print("   1. Haz clic en una vuelta del leaderboard")
        print("   2. La URL contendrá algo como /analysis/XXXXX")
        print("   3. Ese XXXXX es el ID de la vuelta")
        print()
        print("   Escribe 'exit' para terminar.\n")

        downloaded = 0
        while downloaded < max_laps:
            lap_id = input(f"   [{downloaded + 1}/{max_laps}] Lap ID: ").strip()
            if lap_id.lower() in ("exit", "quit", "q", ""):
                break

            success, filename = self.download_csv(lap_id)
            if success:
                downloaded += 1
                print(f"      ✓ Descargado: {filename}")
            else:
                print("      ✗ No se pudo descargar. ¿Telemetría disponible?")

        print(f"\n   Total descargados: {downloaded}\n")
        return downloaded

    # -----------------------------------------------------------------------
    #  Utilidad: inspeccionar la página
    # -----------------------------------------------------------------------
    def inspect_page(self):
        """
        Herramienta de diagnóstico: muestra qué hay en la página actual.
        Útil para ajustar selectores si la descarga automática falla.
        """
        print("\n🔬 INSPECCIÓN DE PÁGINA")
        print("=" * 50)
        print(f"   URL: {self.driver.current_url}")
        print(f"   Título: {self.driver.title}")

        # Contar elementos
        for tag in ("a", "button", "table", "tr", "input", "select"):
            count = len(self.driver.find_elements(By.TAG_NAME, tag))
            if count > 0:
                print(f"   <{tag}>: {count}")

        # Links interesantes
        print("\n   — Links con posible relación a vueltas —")
        links = self.driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            href = link.get_attribute("href") or ""
            text = link.text.strip()[:50]
            if any(kw in href.lower() for kw in ("laps", "analysis", "csv", "telemetry", "analyze")):
                if link.is_displayed():
                    print(f"   '{text}' → {href[:100]}")

        # Botones visibles
        print("\n   — Botones visibles —")
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons[:20]:
            if btn.is_displayed():
                text = btn.text.strip()[:60]
                title = btn.get_attribute("title") or ""
                aria = btn.get_attribute("aria-label") or ""
                print(f"   [{text}] title='{title}' aria='{aria}'")

        # Tablas
        print("\n   — Tablas —")
        tables = self.driver.find_elements(By.TAG_NAME, "table")
        for idx, table in enumerate(tables):
            rows = table.find_elements(By.TAG_NAME, "tr")
            print(f"   Tabla {idx}: {len(rows)} filas")
            if rows:
                first_row_text = rows[0].text.strip()[:100]
                print(f"      Primera fila: {first_row_text}")

        print()

    # -----------------------------------------------------------------------
    #  Cierre
    # -----------------------------------------------------------------------
    def close(self):
        """Cerrar el navegador."""
        if self.driver:
            self.driver.quit()
            print("✓ Navegador cerrado")


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    print("\n" + "=" * 70)
    print(" " * 8 + "DESCARGADOR DE CSVs — GARAGE61.NET")
    print("=" * 70)
    print(f"\n   🏁 Circuito:  Adelaide Street Circuit")
    print(f"   🏎️  Coche:     Ferrari 296 GT3")
    print(f"   📊 Máximo:    {MAX_LAPS} vueltas")
    print(f"   📁 Carpeta:   {DOWNLOAD_FOLDER}")
    print(f"   👁️  Headless:  {'Sí' if HEADLESS else 'No (verás el navegador)'}")
    print("=" * 70 + "\n")

    downloader = None

    try:
        downloader = Garage61Downloader(DOWNLOAD_FOLDER, headless=HEADLESS)

        # LOGIN
        print("PASO 1/3: Login")
        print("-" * 50)
        if not downloader.login(GARAGE61_EMAIL, GARAGE61_PASSWORD):
            print("❌ Login fallido. Abortando.")
            return

        # BUSCAR + NAVEGAR + DESCARGAR
        print("PASO 2/3: Buscar circuito y coche")
        print("-" * 50)
        downloaded = downloader.run(TRACK_SEARCH, CAR_SEARCH, MAX_LAPS)

        if downloaded == 0:
            print("\n💡 DIAGNÓSTICO ADICIONAL")
            print("-" * 50)
            print("   Ejecutando inspección de la página...")
            downloader.inspect_page()
            print("\n   Si necesitas ajustar algo, revisa los archivos en:")
            print(f"   {downloader.download_folder}")
            print("   - laps_page.png     → captura de la página de vueltas")
            print("   - debug_page.html   → HTML de la página")
            print("   - debug_screenshot.png → captura adicional")

    except KeyboardInterrupt:
        print("\n\n⚠ Interrumpido por el usuario")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if downloader:
            print("\n🧹 Cerrando...")
            downloader.close()
            print("✓ Proceso finalizado\n")


if __name__ == "__main__":
    main()
