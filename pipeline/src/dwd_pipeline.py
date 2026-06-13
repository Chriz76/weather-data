import os
import sys
import requests
import shutil
import json
import time  
from datetime import datetime, timezone, timedelta

# Pfad-Erweiterung für den Import
sys.path.append(os.path.dirname(__file__))
from process import WindProcessor

# --- KONFIGURATION ---
ROOT_FOLDER = "."  
OUTPUT_DIR = "./output" 
TEMP_GRIB_DIR = "./grib_temp"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_GRIB_DIR, exist_ok=True)

def download_file(url, local_path):
    """Hilfsfunktion für echte Downloads mit Stream-Pufferung und 3 Retries"""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=15, stream=True)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                print(f"   ⚠️ Download-Versuch {attempt} fehlgeschlagen (Status: {response.status_code})")
        except Exception as e:
            print(f"   ⚠️ Download-Fehler bei Versuch {attempt}: {e}")
        
        if attempt < max_retries:
            wait_time = attempt * 2  # Steigendes Intervall: 2s, 4s...
            print(f"   ⏳ Warte {wait_time} Sekunden vor nächstem Versuch...")
            time.sleep(wait_time)
            
    return False

def run_ruc_pipeline():
    utc_now = datetime.now(timezone.utc)
    print(f"\n========================================================")
    print(f"START PIPELINE - Aktuelle UTC Zeit: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}Z")
    print(f"========================================================")

    # Erstelle den Prozessor
    processor = WindProcessor(root_folder=ROOT_FOLDER, output_folder=OUTPUT_DIR)
    newest_run_processed = False
    detected_current_hour = None

    # Bereich von 14 auf 5 Stunden reduziert für weniger Rückblick
    for hour_offset in range(5):
        target_time = utc_now - timedelta(hours=hour_offset)
        target_time = target_time.replace(minute=0, second=0, microsecond=0)
        dwd_run_folder = target_time.strftime("%Y-%m-%dT%H:00")

        # Pfad für die finale +14h PNG-Datei dieses Laufs
        final_valid_time = target_time + timedelta(hours=14)
        final_output_filename = f"{final_valid_time.strftime('%Y%m%d_%H')}Z.png"
        final_output_path = os.path.join(OUTPUT_DIR, final_output_filename)

        should_process = False
        missing_hours = []

        if not newest_run_processed:
            if os.path.exists(final_output_path):
                print(f"💾 [Lauf {dwd_run_folder}Z] Bereits komplett vorhanden. (Wechsle in Historie-Modus)")
                newest_run_processed = True
                detected_current_hour = target_time.strftime('%Y%m%d_%H')
            else:
                print(f"📡 [Lauf {dwd_run_folder}Z] Fehlt lokal. Prüfe DWD Server...")
                should_process = True
                missing_hours = list(range(15))  # Alle 15 Schritte von PT000H bis PT014H

        if newest_run_processed:
            # Modus B: Historie nach Lücken scannen
            for f_hour in range(15):
                valid_time = target_time + timedelta(hours=f_hour)
                output_filename = f"{valid_time.strftime('%Y%m%d_%H')}Z.png"
                if not os.path.exists(os.path.join(OUTPUT_DIR, output_filename)):
                    missing_hours.append(f_hour)
            
            if missing_hours:
                print(f"📚 [Lauf {dwd_run_folder}Z] {len(missing_hours)} Lücke(n) entdeckt. Prüfe Server...")
                should_process = True

        # Wenn etwas verarbeitet werden muss, fragen wir den Server per HEAD
        if should_process and missing_hours:
            test_url = f"https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/V_10M/r/{target_time.strftime('%Y-%m-%dT%H%%3A00')}/s/PT014H00M.grib2"
            
            head_success = False
            response_code = None
            
            # --- NEU: INTELLIGENTE WARTESCHLEIFE FÜR DEN AKTUELLSTEN LAUF (hour_offset == 0) ---
            max_wait_attempts = 3 if hour_offset == 0 else 1
            wait_delay_seconds = 300  # 5 Minuten Splitting-Intervall
            
            for wait_attempt in range(1, max_wait_attempts + 1):
                if wait_attempt > 1:
                    print(f"   ⏰ [Retry-Schleife] Versuche es erneut in {wait_delay_seconds // 60} Minuten... (Versuch {wait_attempt}/{max_wait_attempts})")
                    time.sleep(wait_delay_seconds)
                
                # HEAD-Check innerhalb des aktuellen Versuchs (mit schnellen 3 internen Retries bei Netzwerk-Schluckauf)
                max_head_retries = 3
                for attempt in range(1, max_head_retries + 1):
                    try:
                        response = requests.head(test_url, timeout=8) # Leicht erhöhtes Timeout
                        response_code = response.status_code
                        if response_code == 200:
                            head_success = True
                            break
                        else:
                            print(f"   ⚠️ HEAD-Check {attempt} ergab Statuscode: {response_code}")
                    except Exception as e:
                        print(f"   ⚠️ HEAD-Verbindungsfehler bei Check {attempt}: {e}")
                    
                    if attempt < max_head_retries:
                        time.sleep(3) # Kurze Atempause bei direktem Verbindungsfehler
                
                # Wenn der Server bereit ist (200 OK), brechen wir die 5-Minuten-Warteschleife sofort ab!
                if head_success:
                    break
                elif hour_offset == 0 and wait_attempt < max_wait_attempts:
                    print(f"   ❌ [Lauf {dwd_run_folder}Z] Noch nicht auf dem DWD-Server verfügbar (Status: {response_code or 'Timeout'}).")

            if head_success:
                print(f"👑 [Lauf {dwd_run_folder}Z] Server bereit! Starte jetzt die Verarbeitung von {len(missing_hours)} Schritten...")
                
                if not newest_run_processed:
                    detected_current_hour = target_time.strftime('%Y%m%d_%H')

                for f_hour in missing_hours:
                    valid_time = target_time + timedelta(hours=f_hour)
                    time_key = valid_time.strftime('%Y%m%d_%H')
                    png_filename = f"{time_key}Z.png"
                    
                    url_u = f"https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/U_10M/r/{target_time.strftime('%Y-%m-%dT%H%%3A00')}/s/PT{f_hour:03d}H00M.grib2"
                    url_v = f"https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/V_10M/r/{target_time.strftime('%Y-%m-%dT%H%%3A00')}/s/PT{f_hour:03d}H00M.grib2"
                    
                    u_path = os.path.join(TEMP_GRIB_DIR, f"u_{time_key}.grib2")
                    v_path = os.path.join(TEMP_GRIB_DIR, f"v_{time_key}.grib2")
                    
                    print(f"   -> Downloade Schritt +{f_hour}h...")
                    if download_file(url_u, u_path) and download_file(url_v, v_path):
                        success = processor.process_step(u_path, v_path, time_key, png_filename)
                        
                        if os.path.exists(u_path): os.remove(u_path)
                        if os.path.exists(v_path): os.remove(v_path)
                        
                        if success:
                            print(f"      ✅ Schritt +{f_hour}h erfolgreich prozessiert.")
                    else:
                        print(f"   ❌ Fehler beim Download von Schritt +{f_hour}h.")
                
                if not newest_run_processed:
                    newest_run_processed = True
            else:
                print(f"❌ [Lauf {dwd_run_folder}Z] Daten final nicht verfügbar (Letzter Status: {response_code}). Wechsle permanent in Historie-Modus.")

    # =========================================================================
    # FINALES SPEICHERN & DYNAMISCHE ERSTELLUNG DER INDEX.JSON
    # =========================================================================
    processor.flush_json_to_disk()

    all_timestamps = []
    if processor.cluster_memory:
        first_cluster = next(iter(processor.cluster_memory.values()))
        if "timeline" in first_cluster:
            all_timestamps = list(first_cluster["timeline"].keys())

    if all_timestamps:
        index_path = os.path.join(OUTPUT_DIR, "index.json")
        sorted_timestamps = sorted(all_timestamps)
        
        if detected_current_hour and detected_current_hour in sorted_timestamps:
            current_hour = detected_current_hour
        else:
            if len(sorted_timestamps) >= 14:
                current_hour = sorted_timestamps[len(sorted_timestamps) // 2]
            else:
                current_hour = sorted_timestamps[-1]

        index_data = {
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "available_timestamps": sorted_timestamps,
            "current_hour": current_hour
        }

        print(f"\n📝 [Hauptprogramm] Generiere {index_path}...")
        print(f"   -> Verfügbare Schritte im Frontend: {len(sorted_timestamps)}")
        print(f"   -> Exakt detektierter Standard-Fokus (current_hour): {current_hour}")
        
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
        print("✅ index.json erfolgreich aktualisiert!")
    else:
        print("⚠️ Warnung: Keine Daten-Timestamps für die index.json gefunden.")

    if os.path.exists(TEMP_GRIB_DIR):
        shutil.rmtree(TEMP_GRIB_DIR)
        print("🧹 Temporärer Download-Ordner wurde vollständig bereinigt!")

    print("\n🎉 PIPELINE ERFOLGREICH BEENDET!")

if __name__ == "__main__":
    run_ruc_pipeline()
