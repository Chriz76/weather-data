import os
import sys
import requests
import shutil
import json
import time
from datetime import datetime, timezone, timedelta

# Funktioniert im Notebook/Colab und als Skript
script_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
sys.path.append(script_dir)

from process import AromeWindProcessor

# --- KONFIGURATION ---
OUTPUT_DIR = "./output"
TEMP_OM_DIR = "./om_temp"
API_VERSION = "1.1.0"

# Open-Meteo Modellname aus S3
MODEL_NAME = "meteofrance_arome_france0025_15min"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_OM_DIR, exist_ok=True)


def download_file(url, local_path):
    """
    Hilfsfunktion für Downloads.
    Gibt bei HTTP 404 sofort False zurück (kein unnötiges Warten/Retry).
    """
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=15, stream=True)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=16384):
                        f.write(chunk)
                return True
            elif response.status_code == 404:
                return False
            else:
                print(f"   ⚠️ Download-Versuch {attempt} fehlgeschlagen (Status: {response.status_code})")
        except Exception as e:
            print(f"   ⚠️ Download-Fehler bei Versuch {attempt}: {e}")

        if attempt < max_retries:
            wait_time = attempt * 2
            print(f"   ⏳ Warte {wait_time} Sekunden vor nächstem Versuch...")
            time.sleep(wait_time)

    return False


def run_arome_pipeline_15min():
    arome_run_env = os.environ.get("TARGET_RUN")
    
    if not arome_run_env:
        print("❌ FEHLER: Umgebungsvariable 'TARGET_RUN' fehlt!")
        sys.exit(1)

    # Erwartetes Format: "2026-07-27T17:00"
    target_time = datetime.strptime(arome_run_env, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    
    year = target_time.strftime("%Y")
    month = target_time.strftime("%m")
    day = target_time.strftime("%d")
    run_hour_z = target_time.strftime("%H") + "00Z"  # z. B. "1700Z"

    print(f"\n========================================================")
    print(f"START PIPELINE (15-Minuten) - Run: {target_time.strftime('%Y-%m-%dT%H:%M')}Z")
    print(f"========================================================")

    # Processor initialisieren
    processor = AromeWindProcessor(output_folder=OUTPUT_DIR, width=2000)

    # Start-Fokus für die index.json (z. B. 20260727_1700)
    detected_current_time_key = target_time.strftime('%Y%m%d_%H%M')
    processed_timestamps = []

    current_step_time = target_time
    consecutive_missing = 0

    # Solange 15-Minuten-Schritte herunterladen, bis Dateien auf S3 aufhören (404)
    while consecutive_missing < 2:
        # Format für Frontend-Grafiken: YYYYMMDD_HHMMZ.webp (z.B. 20260727_1715Z.webp)
        time_key = current_step_time.strftime('%Y%m%d_%H%M')
        webp_filename = f"{time_key}Z.webp"

        # Dateiname im S3-Bucket (z. B. 2026-07-27T1715.om)
        iso_file_name = f"{current_step_time.strftime('%Y-%m-%dT%H%M')}.om"
        
        url_om = f"https://openmeteo.s3.amazonaws.com/data_spatial/{MODEL_NAME}/{year}/{month}/{day}/{run_hour_z}/{iso_file_name}"
        om_path = os.path.join(TEMP_OM_DIR, iso_file_name)

        print(f"   -> Downloade 15m-Schritt: {iso_file_name}...")
        
        if download_file(url_om, om_path):
            consecutive_missing = 0
            success = processor.process_om_file(om_path, output_filename=webp_filename)

            if os.path.exists(om_path):
                os.remove(om_path)

            if success:
                processed_timestamps.append(time_key)
                print(f"      ✅ Erfolgreich prozessiert -> {webp_filename}")
        else:
            consecutive_missing += 1
            print(f"   ℹ️ Datei {iso_file_name} im Run {run_hour_z} nicht mehr vorhanden.")

        # Nächster 15-Minuten-Schritt
        current_step_time += timedelta(minutes=15)

    # =========================================================================
    # GENERIERUNG DER INDEX.JSON FÜR DAS FRONTEND
    # =========================================================================
    if processed_timestamps:
        index_path = os.path.join(OUTPUT_DIR, "index.json")
        sorted_timestamps = sorted(processed_timestamps)

        if detected_current_time_key in sorted_timestamps:
            current_hour = detected_current_time_key
        else:
            current_hour = sorted_timestamps[0]

        index_data = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "available_timestamps": sorted_timestamps,
            "current_hour": current_hour,
            "step_type": "15min",
            "api_version": API_VERSION
        }

        print(f"\n📝 Generiere {index_path}...")
        print(f"   -> Insgesamt generierte 15-Minuten-Schritte: {len(sorted_timestamps)}")
        print(f"   -> Standard-Fokus (current_hour): {current_hour}")

        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
        print("✅ index.json erfolgreich aktualisiert!")
    else:
        print("⚠️ Warnung: Keine Daten-Timestamps erzeugt.")

    # Aufräumen
    if os.path.exists(TEMP_OM_DIR):
        shutil.rmtree(TEMP_OM_DIR)
        print("🧹 Temporärer Ordner bereinigt!")

    print("\n🎉 15-MINUTEN PIPELINE SAUBER BEENDET!")


if __name__ == "__main__":
    if "TARGET_RUN" not in os.environ:
        os.environ["TARGET_RUN"] = "2026-07-27T17:00"

    run_arome_pipeline_15min()