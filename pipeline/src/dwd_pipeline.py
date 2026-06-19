import os
import sys
import requests
import shutil
import json
import time
from datetime import datetime, timezone, timedelta
# import pickle # Removed for loading index_payload

# Add the current script's directory to sys.path to find process.py
sys.path.append(os.path.dirname(__file__))
from process import WindProcessor

# --- CONFIGURATION ---
# OUTPUT_DIR will be where the gh-pages branch is checked out, usually './output'
OUTPUT_DIR = "./output"
TEMP_GRIB_DIR = "./grib_temp"

# Path to the warmed-up index.
# This part is removed as WindProcessor now handles geometry loading internally.
# INDEX_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "warmed_static_weather_indices_dense.pkl")

# ROOT_FOLDER now points to where clat.grib2 and clon.grib2 are expected
ROOT_FOLDER = "/content/"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_GRIB_DIR, exist_ok=True)

# Load the warmed-up static weather indices
# This block is removed as WindProcessor now handles this internally.
# print(f"--- Loading warmed-up static weather indices from {INDEX_DATA_PATH} ---")
# with open(INDEX_DATA_PATH, "rb") as f:
#     index_payload = pickle.load(f)
# print("✅ Warmed-up static weather indices loaded.")


def download_file(url, local_path):
    """Helper function for real downloads with stream buffering and 3 retries"""
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
                print(f"   ⚠️ Download attempt {attempt} failed (Status: {response.status_code})")
        except Exception as e:
            print(f"   ⚠️ Download error on attempt {attempt}: {e}")

        if attempt < max_retries:
            wait_time = attempt * 2  # Increasing interval: 2s, 4s...
            print(f"   ⏳ Waiting {wait_time} seconds before next attempt...")
            time.sleep(wait_time)

    return False

def run_ruc_pipeline():
    utc_now = datetime.now(timezone.utc)
    print(f"\n========================================================")
    print(f"START PIPELINE - Current UTC Time: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}Z")
    print(f"========================================================")

    # Initialize WindProcessor, passing the root_folder where clat.grib2 and clon.grib2 are located
    processor = WindProcessor(root_folder=ROOT_FOLDER, output_folder=OUTPUT_DIR)
    # The cluster_output_folder for the processor needs to be relative to OUTPUT_DIR
    processor.cluster_output_folder = os.path.join(OUTPUT_DIR, "grid_cluster")
    os.makedirs(processor.cluster_output_folder, exist_ok=True) # Ensure it exists

    newest_run_processed = False
    detected_current_hour = None

    # Range reduced from 14 to 5 hours for less lookback
    for hour_offset in range(5):
        target_time = utc_now - timedelta(hours=hour_offset)
        target_time = target_time.replace(minute=0, second=0, microsecond=0)
        dwd_run_folder = target_time.strftime("%Y-%m-%dT%H:00")

        # Path for the final +14h PNG file of this run
        final_valid_time = target_time + timedelta(hours=14)
        final_output_filename = f"{final_valid_time.strftime('%Y%m%d_%H')}Z.png"
        final_output_path = os.path.join(OUTPUT_DIR, final_output_filename)

        should_process = False
        missing_hours = []

        if not newest_run_processed:
            if os.path.exists(final_output_path):
                print(f"💾 [Run {dwd_run_folder}Z] Already fully present. (Switching to history mode)")
                newest_run_processed = True
                detected_current_hour = target_time.strftime('%Y%m%d_%H')
            else:
                print(f"📡 [Run {dwd_run_folder}Z] Missing locally. Checking DWD Server...")
                should_process = True
                missing_hours = list(range(15))  # All 15 steps from PT000H to PT014H

        if newest_run_processed:
            # Mode B: Scan history for gaps
            for f_hour in range(15):
                valid_time = target_time + timedelta(hours=f_hour)
                output_filename = f"{valid_time.strftime('%Y%m%d_%H')}Z.png"
                if not os.path.exists(os.path.join(OUTPUT_DIR, output_filename)):
                    missing_hours.append(f_hour)

            if missing_hours:
                print(f"📚 [Run {dwd_run_folder}Z] {len(missing_hours)} gap(s) detected. Checking Server...")
                should_process = True

        # If something needs to be processed, we query the server via HEAD
        if should_process and missing_hours:
            test_url = f"https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/V_10M/r/{target_time.strftime('%Y-%m-%dT%H%%3A00')}/s/PT014H00M.grib2"

            head_success = False
            response_code = None

            # --- NEW: INTELLIGENT WAITING LOOP FOR THE LATEST RUN (hour_offset == 0) ---
            max_wait_attempts = 3 if hour_offset == 0 else 1
            wait_delay_seconds = 300  # 5 minutes splitting interval

            for wait_attempt in range(1, max_wait_attempts + 1):
                if wait_attempt > 1:
                    print(f"   ⏰ [Retry loop] Trying again in {wait_delay_seconds // 60} minutes... (Attempt {wait_attempt}/{max_wait_attempts})")
                    time.sleep(wait_delay_seconds)

                # HEAD check within the current attempt (with quick 3 internal retries for network hiccups)
                max_head_retries = 3
                for attempt in range(1, max_head_retries + 1):
                    try:
                        response = requests.head(test_url, timeout=8) # Slightly increased timeout
                        response_code = response.status_code
                        if response_code == 200:
                            head_success = True
                            break
                        else:
                            print(f"   ⚠️ HEAD-Check {attempt} returned Status: {response_code}")
                    except Exception as e:
                        print(f"   ⚠️ HEAD-Connection error on Check {attempt}: {e}")

                    if attempt < max_head_retries:
                        time.sleep(3) # Short break for direct connection errors

                # If the server is ready (200 OK), break the 5-minute waiting loop immediately!
                if head_success:
                    break
                elif hour_offset == 0 and wait_attempt < max_wait_attempts:
                    print(f"   ❌ [Run {dwd_run_folder}Z] Not yet available on DWD server (Status: {response_code or 'Timeout'}).")

            if head_success:
                print(f"👑 [Run {dwd_run_folder}Z] Server ready! Starting processing of {len(missing_hours)} steps...")

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

                    print(f"   -> Downloading step +{f_hour}h...")
                    if download_file(url_u, u_path) and download_file(url_v, v_path):
                        success = processor.process_step(u_path, v_path, time_key, png_filename)

                        if os.path.exists(u_path): os.remove(u_path)
                        if os.path.exists(v_path): os.remove(v_path)

                        if success:
                            print(f"      ✅ Step +{f_hour}h processed successfully.")
                    else:
                        print(f"   ❌ Error downloading step +{f_hour}h.")

                if not newest_run_processed:
                    newest_run_processed = True
            else:
                print(f"❌ [Run {dwd_run_folder}Z] Data finally not available (Last Status: {response_code}). Switching permanently to history mode.")

    # =========================================================================
    # FINAL SAVING & DYNAMIC CREATION OF INDEX.JSON
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

        print(f"\n📝 [Main Program] Generating {index_path}...")
        print(f"   -> Available steps in frontend: {len(sorted_timestamps)}")
        print(f"   -> Exactly detected standard focus (current_hour): {current_hour}")

        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)
        print("✅ index.json successfully updated!")
    else:
        print("⚠️ Warning: No data timestamps found for index.json.")

    if os.path.exists(TEMP_GRIB_DIR):
        shutil.rmtree(TEMP_GRIB_DIR)
        print("🧹 Temporary download folder has been completely cleaned up!")

    print("\n🎉 PIPELINE SUCCESSFULLY COMPLETED!")

if __name__ == "__main__":
    run_ruc_pipeline()
