"""
Downloader for public laps from Garage61.net, using Playwright to automate browsing and httpx for direct CSV downloads.

    1. Opens Garage61 in a visible Chromium browser.
    2. You log in manually once.
    3. Script navigates to /app/laps?tracks=53&cars=155 automatically.
    4. Intercepts ALL XHR responses containing lap data.
    5. Scrolls until no more laps load (auto-detects end of list).
    6. Downloads CSV for each captured lap.
    7. Skips private laps (403) automatically.
    8. Saves index to disk — resume after interruption with --skip-browser.
"""

import asyncio
import json
import re
import logging
import argparse
from pathlib import Path
from typing import Optional
import httpx
from tqdm import tqdm


#* ------------------------------- CONFIGURATION ------------------------------- #

TRACK_ID      = 53    # Imola — Autodromo Internazionale Enzo e Dino Ferrari
CAR_ID        = 155   # Ferrari 296 GT3

OUTPUT_DIR    = Path("./data/garage61_csvs")
MAX_SCROLLS   = 200   # 200 scrolls x ~20 laps/scroll = up to ~4000 laps
SCROLL_WAIT   = 2.5   # seconds between scrolls (don't go below 1.5)
DOWNLOAD_DELAY = 1.5  # seconds between CSV downloads

GARAGE61_BASE  = "https://garage61.net"
TOKEN_FILE     = Path(".garage61_token")
INDEX_FILE     = Path(".garage61_laps_index.json")
COOKIES_FILE   = Path(".garage61_cookies.json")

#* ------------------------------- LOGGING ------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("garage61.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


#* ------------------------------- NETWORK CAPTURE ------------------------------- #

class NetworkCapture:
    """
    Passively intercepts every Garage61 API response while we browse.
    Extracts:
      - Bearer token (needed to download CSVs later)
      - Full lap metadata for every lap the SPA loads
    """

    def __init__(self):
        self.bearer_token: Optional[str] = None
        self.laps: dict[str, dict] = {}   # lap_id (str) -> full metadata dict

    def on_request(self, request):
        """Grab the Bearer token from any authenticated API request."""
        if "garage61.net/api" not in request.url:
            return
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer ") and not self.bearer_token:
            self.bearer_token = auth[7:]
            log.info(f"Bearer token captured ({len(self.bearer_token)} chars)")

    async def on_response(self, response):
        """
        Extract lap IDs and metadata from any API response that returns laps.
        Wide net: catches /laps, /laps/search, /publicLaps, /leaderboard, etc.
        """
        url = response.url
        if "garage61.net/api" not in url or response.status != 200:
            return
        if not any(kw in url for kw in ("/laps", "/lap-records", "/publicLaps", "/leaderboard")):
            return

        try:
            body = await response.json()
        except Exception:
            return

        # Garage61 can return plain list OR {data: [...]} OR {laps: [...]}
        laps_list = []
        if isinstance(body, list):
            laps_list = body
        elif isinstance(body, dict):
            for key in ("data", "laps", "results", "items", "records"):
                val = body.get(key)
                if isinstance(val, list):
                    laps_list = val
                    break

        if not laps_list:
            return

        new = 0
        for lap in laps_list:
            lap_id = str(
                lap.get("id") or lap.get("lapId") or lap.get("lap_id") or ""
            )
            if lap_id and lap_id not in self.laps:
                self.laps[lap_id] = lap
                new += 1

        if new:
            log.info(
                f"+{new} new laps captured (total: {len(self.laps)}) "
                f"from ...{url.split('garage61.net')[1].split('?')[0]}"
            )


#* ------------------------------- BROWSER SESSION ------------------------------- #

async def run_browser(capture: NetworkCapture, max_scrolls: int) -> Optional[str]:
    """
    Opens Chromium, waits for manual login, navigates to the filtered laps
    page, and scrolls until all laps are loaded.
    Returns the Bearer token.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # Must be visible so you can log in
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )

        # Load saved cookies (skip re-login if session still valid)
        if COOKIES_FILE.exists():
            try:
                await context.add_cookies(json.loads(COOKIES_FILE.read_text()))
                log.info("Loaded saved cookies — you may not need to log in again.")
            except Exception as e:
                log.warning(f"Could not load cookies: {e}")

        page = await context.new_page()
        page.on("request",  capture.on_request)
        page.on("response", lambda r: asyncio.create_task(capture.on_response(r)))

        # ── Step 1: Open Garage61 and detect login ────────────────────────
        laps_url = f"{GARAGE61_BASE}/app/laps"
        log.info(f"Opening {laps_url}...")
        await page.goto(laps_url, wait_until="domcontentloaded", timeout=30_000)
        await asyncio.sleep(3)

        # Fast login check: if we're on a login/auth page, wait for manual login
        current = page.url
        needs_login = any(kw in current.lower() for kw in ("/login", "/sign-in", "/auth", "/signin"))

        if needs_login:
            print("\n" + "="*60)
            print("  LOGIN REQUIRED")
            print("="*60)
            print("  Log in to Garage61 in the browser window.")
            print("  The script continues automatically after login.")
            print("="*60 + "\n")

            for _ in range(300):
                url = page.url
                if "/app" in url and not any(kw in url.lower() for kw in ("/login", "/sign-in", "/auth", "/signin")):
                    log.info("Login successful!")
                    await asyncio.sleep(3)  # let post-login API calls settle
                    break
                await asyncio.sleep(1)
            else:
                log.error("Login timeout (5 min). Closing.")
                await browser.close()
                return None
        else:
            log.info("Already logged in (cookies valid).")

        # Save cookies for next run
        cookies = await context.cookies()
        COOKIES_FILE.write_text(json.dumps(cookies, indent=2))
        log.info(f"Cookies saved to {COOKIES_FILE}")

        # ── Step 2: Probe API to find correct filter format ───────────────
        # The SPA ignores URL query params for filtering, so we call the API
        # directly via fetch() to find the format that returns filtered laps.
        capture.laps.clear()
        log.info("Probing API for the correct filter format...")

        probe_js = """
            async ([trackId, carId]) => {
                function extractItems(data) {
                    if (Array.isArray(data)) return data;
                    if (typeof data === 'object' && data !== null) {
                        for (const key of ['data', 'laps', 'results', 'items', 'records']) {
                            if (Array.isArray(data[key])) return data[key];
                        }
                    }
                    return [];
                }

                const formats = [
                    `/api/internal/laps?trackId=${trackId}&carId=${carId}&pageSize=50`,
                    `/api/internal/laps?tracks=${trackId}&cars=${carId}&pageSize=50`,
                    `/api/internal/laps?tracks[]=${trackId}&cars[]=${carId}&pageSize=50`,
                    `/api/internal/laps?track=${trackId}&car=${carId}&pageSize=50`,
                    `/api/internal/publicLaps?trackId=${trackId}&carId=${carId}&pageSize=50`,
                    `/api/internal/public-laps?trackId=${trackId}&carId=${carId}&pageSize=50`,
                    `/api/internal/laps/search?trackId=${trackId}&carId=${carId}&pageSize=50`,
                    `/api/internal/lap-records?trackId=${trackId}&carId=${carId}&pageSize=50`,
                    `/api/v1/laps?trackId=${trackId}&carId=${carId}&limit=50`,
                    `/api/v1/publicLaps?trackId=${trackId}&carId=${carId}&limit=50`,
                    `/api/v1/public-laps?trackId=${trackId}&carId=${carId}&limit=50`,
                    `/api/v1/lap-records?trackId=${trackId}&carId=${carId}&limit=50`,
                    `/api/v1/laps?trackId=${trackId}&carId=${carId}&pageSize=50`,
                ];

                const tried = [];
                for (const url of formats) {
                    try {
                        const r = await fetch(url, {credentials: 'include'});
                        let text = '';
                        try { text = await r.text(); } catch {}
                        let data;
                        try { data = JSON.parse(text); } catch {
                            tried.push({url, status: r.status, count: 0, note: 'not JSON'});
                            continue;
                        }
                        const items = extractItems(data);
                        tried.push({url, status: r.status, count: items.length});

                        if (items.length > 4) {
                            return {success: true, url, count: items.length, firstBatch: items};
                        }
                    } catch (e) {
                        tried.push({url, count: 0, error: e.message});
                    }
                }
                return {success: false, tried};
            }
        """
        probe_result = await page.evaluate(probe_js, [TRACK_ID, CAR_ID])

        if probe_result.get("success"):
            working_url = probe_result["url"]
            first_batch = probe_result.get("firstBatch", [])
            first_count = probe_result["count"]
            log.info(f"Found working API: {working_url} ({first_count} laps in first page)")

            # Add first batch to capture
            for lap in first_batch:
                lap_id = str(lap.get("id") or lap.get("lapId") or lap.get("lap_id") or "")
                if lap_id:
                    capture.laps[lap_id] = lap

            # ── Step 2b: Paginate only if first batch looks like a full page ──
            # If API returned way more than pageSize (e.g. 1000 with pageSize=50),
            # it likely returned ALL results at once — no pagination needed.
            requested_size = 50
            if first_count <= requested_size:
                log.info("Paginating to collect all laps...")
                paginate_js = """
                    async ([baseUrl, existingIds, maxPages]) => {
                        function extractItems(data) {
                            if (Array.isArray(data)) return data;
                            if (typeof data === 'object' && data !== null) {
                                for (const key of ['data', 'laps', 'results', 'items', 'records']) {
                                    if (Array.isArray(data[key])) return data[key];
                                }
                            }
                            return [];
                        }
                        function getId(lap) {
                            return String(lap.id || lap.lapId || lap.lap_id || '');
                        }

                        const known = new Set(existingIds);
                        const newLaps = [];
                        const sep = baseUrl.includes('?') ? '&' : '?';

                        // Try page-based, offset-based, and skip-based pagination
                        const strategies = [
                            (p) => `${baseUrl}${sep}page=${p}`,
                            (p) => `${baseUrl}${sep}offset=${p * 50}`,
                            (p) => `${baseUrl}${sep}skip=${p * 50}`,
                        ];

                        for (const makeUrl of strategies) {
                            let emptyStreak = 0;
                            let foundAny = false;
                            for (let p = 1; p <= maxPages; p++) {
                                try {
                                    const r = await fetch(makeUrl(p), {credentials: 'include'});
                                    if (!r.ok) break;
                                    const data = await r.json();
                                    const items = extractItems(data);
                                    if (items.length === 0) break;

                                    let newInPage = 0;
                                    for (const lap of items) {
                                        const id = getId(lap);
                                        if (id && !known.has(id)) {
                                            known.add(id);
                                            newLaps.push(lap);
                                            newInPage++;
                                        }
                                    }
                                    if (newInPage > 0) {
                                        foundAny = true;
                                        emptyStreak = 0;
                                    } else {
                                        emptyStreak++;
                                        if (emptyStreak >= 2) break;
                                    }
                                } catch { break; }
                            }
                            if (foundAny) break;  // this strategy works, stop
                        }
                        return newLaps;
                    }
                """
                existing_ids = list(capture.laps.keys())
                more_laps = await page.evaluate(paginate_js, [working_url, existing_ids, 100])
                for lap in more_laps:
                    lap_id = str(lap.get("id") or lap.get("lapId") or lap.get("lap_id") or "")
                    if lap_id and lap_id not in capture.laps:
                        capture.laps[lap_id] = lap
                log.info(f"Pagination done. +{len(more_laps)} new laps.")
            else:
                log.info(
                    f"API returned {first_count} laps at once (requested {requested_size}). "
                    f"Likely all results — skipping pagination."
                )

            log.info(f"Total laps collected: {len(capture.laps)}")

        else:
            # Log what we tried for debugging
            log.warning("No API format returned filtered results (>4 laps).")
            for entry in probe_result.get("tried", []):
                log.info(f"  {entry.get('url', '?')}  ->  status={entry.get('status', '?')}  count={entry.get('count', 0)}  {entry.get('note', '')}  {entry.get('error', '')}")

            # ── Fallback: ask user to apply filters in the UI ─────────────
            print("\n" + "="*60)
            print("  MANUAL FILTERING REQUIRED")
            print("="*60)
            print("  The API probe did not find filtered results.")
            print("  In the browser, please:")
            print(f"    1. Navigate to the public laps / leaderboard")
            print(f"    2. Filter by Track: Imola  (ID {TRACK_ID})")
            print(f"    3. Filter by Car: Ferrari 296 GT3  (ID {CAR_ID})")
            print(f"    4. Press ENTER here when filtered laps are visible.")
            print("="*60)
            input()
            capture.laps.clear()
            await asyncio.sleep(3)

            # Now scroll to load all filtered laps
            log.info(f"Scrolling (max {max_scrolls} iterations)...")
            no_new_streak = 0
            for i in range(max_scrolls):
                prev = len(capture.laps)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(SCROLL_WAIT)
                try:
                    btn = page.get_by_role(
                        "button", name=re.compile(r"load more|show more|ver más", re.I)
                    )
                    if await btn.is_visible(timeout=400):
                        await btn.click()
                        await asyncio.sleep(SCROLL_WAIT)
                except Exception:
                    pass
                if len(capture.laps) == prev:
                    no_new_streak += 1
                    if no_new_streak >= 5:
                        log.info("No new laps in 5 consecutive scrolls — end of list.")
                        break
                else:
                    no_new_streak = 0
                    if (i + 1) % 10 == 0:
                        log.info(f"Scroll {i+1}: {len(capture.laps)} laps so far...")

        log.info(f"Browser phase done. Total laps captured: {len(capture.laps)}")

        # Extract cookies as dict for httpx downloads
        cookies_for_httpx = {c["name"]: c["value"] for c in cookies}

        # Try to get Bearer token (may not exist if auth is cookie-only)
        if not capture.bearer_token:
            try:
                token = await page.evaluate("""
                    () => {
                        for (let i = 0; i < localStorage.length; i++) {
                            const key = localStorage.key(i);
                            const val = localStorage.getItem(key);
                            if (val && val.length > 50 && /^eyJ/.test(val)) return val;
                        }
                        for (let i = 0; i < sessionStorage.length; i++) {
                            const key = sessionStorage.key(i);
                            const val = sessionStorage.getItem(key);
                            if (val && val.length > 50 && /^eyJ/.test(val)) return val;
                        }
                        return '';
                    }
                """)
                if token:
                    capture.bearer_token = token
                    log.info(f"Token from storage ({len(token)} chars)")
            except Exception:
                pass

        await browser.close()
        return {"token": capture.bearer_token, "cookies": cookies_for_httpx}


#* ------------------------------- CSV DOWNLOAD ------------------------------- #

# Car and track name maps (matching Garage61's display names)
CAR_NAMES = {
    155: "Ferrari 296 GT3",
}
TRACK_NAMES = {
    53: "Autodromo Internazionale Enzo e Dino Ferrari (Grand Prix)",
}


def build_filename(lap_id: str, meta: dict) -> str:
    """
    Match Garage61's own filename format:
    Garage 61 - {driver} - {car} - {track} - {lap_time} - {id}.csv
    """
    # Driver name
    driver = (
        meta.get("driver_name")
        or meta.get("driverName")
        or (meta.get("driver") or {}).get("name")
        or "Unknown"
    )
    # Build full name from driver sub-object if driver_name is missing
    if driver == "Unknown":
        d = meta.get("driver") or {}
        fn = d.get("firstname", "")
        ln = d.get("lastname", "")
        if fn or ln:
            driver = f"{fn} {ln}".strip()

    # Car name from ID
    car_id = meta.get("car_id") or meta.get("carId") or CAR_ID
    car_name = CAR_NAMES.get(car_id, f"Car {car_id}")

    # Track name from ID
    track_id = meta.get("track_id") or meta.get("trackId") or TRACK_ID
    track_name = TRACK_NAMES.get(track_id, f"Track {track_id}")

    # Lap time: API returns seconds (e.g. 99.500238529 → "01.39.500")
    t = meta.get("lap_time") or meta.get("lapTime") or meta.get("time") or 0
    if isinstance(t, (int, float)) and t > 0:
        total_ms = t * 1000 if t < 1000 else t  # handle both seconds and ms
        mins = int(total_ms // 60_000)
        secs = int((total_ms % 60_000) // 1000)
        millis = int(total_ms % 1000)
        time_s = f"{mins:02d}.{secs:02d}.{millis:03d}"
    elif isinstance(t, str):
        time_s = t
    else:
        time_s = "00.00.000"

    # Sanitize for filesystem (replace chars illegal in Windows filenames)
    safe = lambda s: re.sub(r'[<>:"/\\|?*]', '_', str(s))

    return f"Garage 61 - {safe(driver)} - {safe(car_name)} - {safe(track_name)} - {time_s} - {lap_id}.csv"


async def download_all_csvs(
    laps: dict[str, dict],
    output_dir: Path,
    token: Optional[str] = None,
    cookies: Optional[dict] = None,
) -> dict:
    """Download a CSV for every captured lap ID using Bearer token or cookies."""

    output_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "Accept": "text/csv, application/octet-stream, */*",
        "Referer": f"{GARAGE61_BASE}/app/laps",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
        ),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
        log.info("Using Bearer token for CSV downloads.")
    elif cookies:
        log.info("Using cookies for CSV downloads (no Bearer token).")
    else:
        log.warning("No token or cookies — downloads may fail.")

    stats = {"downloaded": 0, "private": 0, "no_csv": 0, "error": 0, "existed": 0}
    items = list(laps.items())
    log.info(f"Downloading CSVs for {len(items)} laps...")

    async with httpx.AsyncClient(
        headers=headers,
        cookies=cookies or {},
        timeout=60,
        follow_redirects=True,
    ) as client:
        # Probe which CSV endpoint works before downloading all laps
        csv_url_templates = [
            f"{GARAGE61_BASE}/api/internal/laps/{{lap_id}}/csv",
            f"{GARAGE61_BASE}/api/internal/laps/{{lap_id}}/export",
            f"{GARAGE61_BASE}/api/v1/laps/{{lap_id}}/csv",
        ]
        working_template = None
        probe_id = items[0][0] if items else None

        if probe_id:
            log.info(f"Probing CSV download endpoints with lap {probe_id}...")
            for tmpl in csv_url_templates:
                url = tmpl.format(lap_id=probe_id)
                try:
                    r = await client.get(url)
                    log.info(f"  {url.split('garage61.net')[1]}  ->  {r.status_code}  ({len(r.content)} bytes)")
                    if r.status_code == 200 and len(r.content) > 50 and b"\n" in r.content[:500]:
                        working_template = tmpl
                        log.info(f"Working CSV endpoint: {tmpl}")
                        break
                except Exception as e:
                    log.info(f"  {url.split('garage61.net')[1]}  ->  error: {e}")

        if not working_template:
            log.error("No working CSV download endpoint found. All returned auth errors or no data.")
            log.info("Try running without --skip-browser to get fresh auth cookies.")
            return stats

        for lap_id, meta in tqdm(items, desc="Downloading", unit="lap"):
            filepath = output_dir / build_filename(lap_id, meta)

            # Skip if already downloaded — safe to resume after interruption
            if filepath.exists() and filepath.stat().st_size > 100:
                stats["existed"] += 1
                continue

            await asyncio.sleep(DOWNLOAD_DELAY)

            try:
                r = await client.get(working_template.format(lap_id=lap_id))

                if r.status_code == 200:
                    # Verify it's real CSV, not a JSON error wrapped in a 200
                    if len(r.content) > 50 and b"\n" in r.content[:500]:
                        filepath.write_bytes(r.content)
                        log.debug(f"OK {filepath.name}")
                        stats["downloaded"] += 1
                    else:
                        log.debug(f"Lap {lap_id}: 200 but no valid CSV content")
                        stats["no_csv"] += 1

                elif r.status_code == 403:
                    log.debug(f"Lap {lap_id}: private (403)")
                    stats["private"] += 1

                elif r.status_code == 404:
                    log.debug(f"Lap {lap_id}: not found (404)")
                    stats["no_csv"] += 1

                elif r.status_code == 429:
                    wait = int(r.headers.get("retry-after", 60))
                    log.warning(f"Rate limited. Waiting {wait}s...")
                    await asyncio.sleep(wait)
                    r2 = await client.get(working_template.format(lap_id=lap_id))
                    if r2.status_code == 200 and len(r2.content) > 50:
                        filepath.write_bytes(r2.content)
                        stats["downloaded"] += 1
                    else:
                        stats["error"] += 1

                else:
                    log.warning(f"Lap {lap_id}: HTTP {r.status_code}")
                    stats["error"] += 1

            except Exception as e:
                log.error(f"Lap {lap_id}: {e}")
                stats["error"] += 1

    return stats



def save_index(laps: dict):
    INDEX_FILE.write_text(json.dumps(laps, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Lap index saved ({len(laps)} laps) -> {INDEX_FILE}")

def load_index() -> dict:
    if INDEX_FILE.exists():
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        log.info(f"Loaded {len(data)} laps from saved index.")
        return data
    return {}


#* -------------------------------- MAIN -------------------------------- #

async def main():
    parser = argparse.ArgumentParser(description="Garage61 public lap downloader")
    parser.add_argument("--max-scrolls", type=int, default=MAX_SCROLLS,
                        help=f"Max scroll iterations (default: {MAX_SCROLLS})")
    parser.add_argument("--output",      type=Path, default=OUTPUT_DIR,
                        help="Output directory for CSV files")
    parser.add_argument("--skip-browser", action="store_true",
                        help="Skip browser phase — use saved token + lap index")
    args = parser.parse_args()

    log.info("Garage61 Public Laps Downloader")
    log.info(f"  Track ID : {TRACK_ID}  (Imola)")
    log.info(f"  Car ID   : {CAR_ID}  (Ferrari 296 GT3)")
    log.info("="*55)

    capture = NetworkCapture()
    token   = None
    cookies = {}

    if args.skip_browser:
        # Load token if available
        if TOKEN_FILE.exists():
            token = TOKEN_FILE.read_text().strip()
        # Load cookies as fallback
        if COOKIES_FILE.exists():
            try:
                raw = json.loads(COOKIES_FILE.read_text())
                cookies = {c["name"]: c["value"] for c in raw}
            except Exception:
                pass
        if not token and not cookies:
            log.error("No saved token or cookies. Run without --skip-browser first.")
            return
        capture.laps = load_index()
        if not capture.laps:
            log.error("Saved lap index is empty. Run without --skip-browser first.")
            return
        log.info(f"Auth loaded. {len(capture.laps)} laps from index.")
    else:
        result = await run_browser(capture, max_scrolls=args.max_scrolls)
        if not result:
            log.error("Browser phase failed. Aborting.")
            return
        token = result.get("token")
        cookies = result.get("cookies", {})
        if token:
            TOKEN_FILE.write_text(token)
        save_index(capture.laps)

    if not capture.laps:
        log.warning("No laps captured. Check track/car IDs and try again.")
        return

    stats = await download_all_csvs(
        capture.laps, args.output, token=token, cookies=cookies
    )

    print("\n" + "="*55)
    print("  DOWNLOAD SUMMARY")
    print("="*55)
    print(f"  Downloaded  : {stats['downloaded']}")
    print(f"  Private     : {stats['private']}  (auto-skipped)")
    print(f"  No CSV      : {stats['no_csv']}  (no telemetry)")
    print(f"  Already had : {stats['existed']}")
    print(f"  Errors      : {stats['error']}")
    print(f"  Saved to    : {args.output.resolve()}")
    print("="*55)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved.")
        print(f"Resume with: python {__file__} --skip-browser")