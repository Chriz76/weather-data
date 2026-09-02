import glob
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
import requests

# Echten Processor aus der process.py Datei importieren
from process import AromeWindProcessor

# --- KONFIGURATION ---
OUTPUT_DIR = "./output"
TEMP_OM_DIR = "./om_temp"
API_VERSION = "1.1.0"

# Open-Meteo Modellnamen aus S3
MODEL_NAME_15MIN = "meteofrance_arome_france0025_15min"  # Wird stündlich berechnet (15-Min-Schritte)
MODEL_NAME_3H = "meteofrance_arome_france0025"         # Wird 3-stündlich berechnet (1-Std.-Schritte)
MAX_WEBP_COUNT = 200  # Maximal zu behaltende WebP-Dateien

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_OM_DIR, exist_ok=True)


def download_file(url, local_path):
    """Hilfsfunktion für Downloads.

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
                print(
                    f"   ⚠️ Download-Versuch {attempt} fehlgeschlagen (Status: {response.status_code})"
                )
        except Exception as e:
            print(f"   ⚠️ Download-Fehler bei Versuch {attempt}: {e}")

        if attempt < max_retries:
            wait_time = attempt * 2
            print(f"   ⏳ Warte {wait_time} Sekunden vor nächstem Versuch...")
            time.sleep(wait_time)

    return False

def process_3hourly_arome_run(
    base_target_time,
    processor,
    hours_to_check=12,
    start_offset_hours=7,
    download_hours=24,
):
    """Sucht ausgehend von base_target_time rückwärts nach dem neuesten verfügbaren
    3-stündlichen Arome-Run auf S3 und verarbeitet ihn.
    """
    end_offset_hours = start_offset_hours + download_hours

    print(f"\n========================================================")
    print(
        f"START 3-STÜNDLICHER CHECK & PROCESSOR (AB +{start_offset_hours}H BIS +{end_offset_hours}H)"
    )
    print(f"========================================================")

    ref_hour = base_target_time.replace(minute=0, second=0, microsecond=0)
    found_run_time = None

    # 1. RÜCKWÄRTSSUCHE NACH DEM NEUESTEN 3H-RUN AUF S3
    for i in range(hours_to_check):
        check_time = ref_hour - timedelta(hours=i)

        if check_time.hour % 3 != 0:
            continue

        year = check_time.strftime("%Y")
        month = check_time.strftime("%m")
        day = check_time.strftime("%d")
        run_hour_z = check_time.strftime("%H") + "00Z"

        # Check auf das tatsächliche Ende dieses konkreten S3-Runs
        check_step_time = check_time + timedelta(hours=end_offset_hours)
        iso_file = f"{check_step_time.strftime('%Y-%m-%dT%H%M')}.om"
        check_url = f"https://openmeteo.s3.amazonaws.com/data_spatial/{MODEL_NAME_3H}/{year}/{month}/{day}/{run_hour_z}/{iso_file}"

        print(
            f"   🔍 Prüfe S3 auf Vollständigkeit bis {check_step_time.strftime('%Y-%m-%dT%H:%M')}Z ({iso_file})..."
        )

        try:
            resp = requests.head(check_url, timeout=10)
            if resp.status_code == 200:
                print(
                    f"   🎯 Vollständigen 3h-Run auf S3 gefunden (Run: {check_time.strftime('%Y-%m-%dT%H:%M')}Z)"
                )
                found_run_time = check_time
                break
        except Exception as e:
            print(f"   ⚠️ Fehler bei HEAD-Anfrage: {e}")

    if not found_run_time:
        print("   ❌ Kein vollständiger 3h-Run auf S3 gefunden.")
        return []

    # 2. DEFINITION DES HERUNTERZULADENDEN BEREICHS
    # Start: Relativ zur aktuellen Ausführungszeit (+7h)
    start_time = base_target_time + timedelta(hours=start_offset_hours)
    # Ende: Das tatsächliche Ende des S3-Runs (Run + 31h)
    end_time = found_run_time + timedelta(hours=end_offset_hours)

    end_time_key = end_time.strftime("%Y%m%d_%H")
    end_webp_filename = f"{end_time_key}Z.webp"
    end_file_path = os.path.join(OUTPUT_DIR, end_webp_filename)

    timestamps_3h = []

    # 3. LOKALE PRÜFUNG: WURDE DIESER RUN BEREITS BIS ZUM ENDE VERARBEITET?
    if os.path.exists(end_file_path):
        print(
            f"   ℹ️ Run {found_run_time.strftime('%Y-%m-%dT%H:%M')}Z wurde bereits vollständig verarbeitet."
        )
        print(
            f"   -> Rekonsolidiere lokale WebPs ({start_time.strftime('%Y-%m-%d %H:00')} bis {end_time.strftime('%Y-%m-%d %H:00')})..."
        )

        curr = start_time
        while curr <= end_time:
            tk = curr.strftime("%Y%m%d_%H")
            if os.path.exists(os.path.join(OUTPUT_DIR, f"{tk}Z.webp")):
                timestamps_3h.append(tk)
            curr += timedelta(hours=1)

        return timestamps_3h

    # 4. DOWNLOAD & VERARBEITUNG (falls neu)
    print(
        f"   🚀 Neuer Run! Starte Download ab {start_time.strftime('%Y-%m-%dT%H:%M')}Z bis {end_time.strftime('%Y-%m-%dT%H:%M')}Z..."
    )

    year = found_run_time.strftime("%Y")
    month = found_run_time.strftime("%m")
    day = found_run_time.strftime("%d")
    run_hour_z = found_run_time.strftime("%H") + "00Z"

    current_step_time = start_time
    consecutive_missing = 0

    while current_step_time <= end_time and consecutive_missing < 2:
        time_key = current_step_time.strftime("%Y%m%d_%H")
        webp_filename = f"{time_key}Z.webp"
        iso_file_name = f"{current_step_time.strftime('%Y-%m-%dT%H%M')}.om"

        url_om = f"https://openmeteo.s3.amazonaws.com/data_spatial/{MODEL_NAME_3H}/{year}/{month}/{day}/{run_hour_z}/{iso_file_name}"
        om_path = os.path.join(TEMP_OM_DIR, iso_file_name)

        print(f"   -> Downloade 3h-Schritt (1h Auflösung): {iso_file_name}...")

        if download_file(url_om, om_path):
            consecutive_missing = 0
            success = processor.process_om_file(
                om_path, output_filename=webp_filename
            )

            if os.path.exists(om_path):
                os.remove(om_path)

            if success:
                timestamps_3h.append(time_key)
                print(f"      ✅ Erfolgreich prozessiert -> {webp_filename}")
        else:
            consecutive_missing += 1
            print(
                f"   ℹ️ Datei {iso_file_name} im 3h-Run nicht vorhanden (404)."
            )

        current_step_time += timedelta(hours=1)

    return timestamps_3h

def cleanup_old_webps(output_dir, max_keep=200):
    """Löscht ältere WebP-Dateien im Ausgabeordner basierend auf dem Dateinamen."""
    webp_files = glob.glob(os.path.join(output_dir, "*.webp"))
    
    if len(webp_files) > max_keep:
        print(f"\n🧹 Bereinige alte WebPs ({len(webp_files)} vorhanden, maximal {max_keep} erlaubt)...")
        # Alphabethische Sortierung nach Dateinamen (älteste Timestamps stehen vorne)
        webp_files.sort()
        
        # Alle Dateien bis auf die letzten max_keep (die neuesten) löschen
        files_to_delete = webp_files[:-max_keep]
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"   🗑️ Gelöscht: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"   ⚠️ Fehler beim Löschen von {file_path}: {e}")
        print(f"✅ Bereinigung abgeschlossen. Es verbleiben {max_keep} WebP-Dateien.")
        

def run_arome_pipeline_15min():
    arome_run_env = os.environ.get("TARGET_RUN")

    if not arome_run_env:
        # Fallback: Falls keine Umgebungsvariable gesetzt ist, nutzen wir vorangehende UTC-Stunde
        fallback_time = datetime.now(timezone.utc) - timedelta(hours=2)
        arome_run_env = fallback_time.strftime("%Y-%m-%dT%H:00")
        print(f"ℹ️ Keine 'TARGET_RUN' Variable gesetzt. Verwende automatischen Fallback-Run: {arome_run_env}")

    target_time = datetime.strptime(
        arome_run_env, "%Y-%m-%dT%H:%M"
    ).replace(tzinfo=timezone.utc)

    year = target_time.strftime("%Y")
    month = target_time.strftime("%m")
    day = target_time.strftime("%d")
    run_hour_z = target_time.strftime("%H") + "00Z"

    print(f"\n========================================================")
    print(
        f"START PIPELINE (15-Minuten) - Run: {target_time.strftime('%Y-%m-%dT%H:%M')}Z"
    )
    print(f"========================================================")

    # Initialisiere den echten AromeWindProcessor
    processor = AromeWindProcessor(output_folder=OUTPUT_DIR, width=2000)

    # Key im HHMM-Format für den 15m Run
    detected_current_time_key = target_time.strftime("%Y%m%d_%H%M")
    processed_timestamps_15m = []

    current_step_time = target_time
    consecutive_missing = 0

    # 1. 15-MINUTEN RUN VERARBEITEN
    while consecutive_missing < 2:
        time_key = current_step_time.strftime("%Y%m%d_%H%M")  # HHMM-Format
        webp_filename = f"{time_key}Z.webp"
        iso_file_name = f"{current_step_time.strftime('%Y-%m-%dT%H%M')}.om"

        url_om = f"https://openmeteo.s3.amazonaws.com/data_spatial/{MODEL_NAME_15MIN}/{year}/{month}/{day}/{run_hour_z}/{iso_file_name}"
        om_path = os.path.join(TEMP_OM_DIR, iso_file_name)

        print(f"   -> Downloade 15m-Schritt: {iso_file_name}...")

        if download_file(url_om, om_path):
            consecutive_missing = 0
            success = processor.process_om_file(
                om_path, output_filename=webp_filename
            )

            if os.path.exists(om_path):
                os.remove(om_path)

            if success:
                processed_timestamps_15m.append(time_key)
                print(f"      ✅ Erfolgreich prozessiert -> {webp_filename}")
        else:
            consecutive_missing += 1
            print(
                f"   ℹ️ Datei {iso_file_name} im Run {run_hour_z} nicht mehr vorhanden."
            )

        current_step_time += timedelta(minutes=15)

    # 2. 3-STÜNDLICHEN RUN VERARBEITEN
    timestamps_3h = process_3hourly_arome_run(
        base_target_time=target_time, processor=processor
    )

    # 3. BEIDE TIMESTAMPS VERBINDEN & INDEX.JSON EINMALIG GENERIEREN
    all_timestamps = sorted(
        list(set(processed_timestamps_15m + timestamps_3h))
    )

    if all_timestamps:
        index_path = os.path.join(OUTPUT_DIR, "index.json")

        current_hour = (
            detected_current_time_key
            if detected_current_time_key in all_timestamps
            else all_timestamps[0]
        )

        index_data = {
            "generated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "available_timestamps": all_timestamps,
            "current_hour": current_hour,
            "step_type": "15min",
            "api_version": API_VERSION,
        }

        print(f"\n📝 Generiere {index_path}...")
        print(f"   -> 15-Min-Schritte (HHMM): {len(processed_timestamps_15m)}")
        print(f"   -> 3h-Schritte (HH):       {len(timestamps_3h)}")
        print(f"   -> Gesamt kombiniert:      {len(all_timestamps)}")
        print(f"   -> Standard-Fokus (current_hour): {current_hour}")

        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
        print("✅ index.json erfolgreich erstellt!")
    else:
        print("⚠️ Warnung: Keine Daten-Timestamps erzeugt.")

    # Alte WebP-Dateien nach Dateinamen-Sortierung bereinigen
    cleanup_old_webps(OUTPUT_DIR, max_keep=MAX_WEBP_COUNT)
    
    # Aufräumen
    if os.path.exists(TEMP_OM_DIR):
        shutil.rmtree(TEMP_OM_DIR)
        print("🧹 Temporärer Ordner bereinigt!")

    print("\n🎉 PIPELINE SAUBER BEENDET!")


if __name__ == "__main__":
    run_arome_pipeline_15min()
