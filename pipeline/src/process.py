import os
import orjson
import pickle
import xarray as xr
import numpy as np
from PIL import Image
import eccodes as ecc
from scipy.interpolate import LinearNDInterpolator
import time


class WindProcessor:
    def __init__(self, root_folder,timeLineLength, output_folder="./wind_tiles_simulation"):
        init_start_time = time.perf_counter()
        self.output_folder = output_folder
        self.cluster_output_folder = os.path.join(output_folder, "grid_cluster")
        os.makedirs(self.cluster_output_folder, exist_ok=True)
        self.root_folder = root_folder
        self.timeLineLength = timeLineLength
        
        self.cluster_memory = {}

        clat_path = os.path.join(self.root_folder, "clat.grib2")
        clon_path = os.path.join(self.root_folder, "clon.grib2")

        if not os.path.exists(clat_path) or not os.path.exists(clon_path):
            raise FileNotFoundError(f"Geometry files missing in folder: {self.root_folder}")

        with open(clat_path, "rb") as f:
            gid = ecc.codes_grib_new_from_file(f)
            raw_lat = ecc.codes_get_array(gid, "values")
            ecc.codes_release(gid)

        with open(clon_path, "rb") as f:
            gid = ecc.codes_grib_new_from_file(f)
            raw_lon = ecc.codes_get_array(gid, "values")
            ecc.codes_release(gid)

        if max(abs(raw_lat)) < 7:
            lat_deg = raw_lat * (180.0 / np.pi)
            lon_deg = raw_lon * (180.0 / np.pi)
        else:
            lat_deg = raw_lat
            lon_deg = raw_lon
        lon_deg = np.where(lon_deg > 180, lon_deg - 360, lon_deg)

        lon_min, lon_max = -4.1616, 20.5444
        lat_min, lat_max = 43.0440, 58.1647

        lat_pts_rad = np.radians(lat_deg)
        y_pts_merc = np.degrees(np.log(np.tan(np.pi/4.0 + lat_pts_rad/2.0)))
        x_pts_merc = lon_deg

        y_min_merc = np.degrees(np.log(np.tan(np.pi/4.0 + np.radians(lat_min)/2.0)))
        y_max_merc = np.degrees(np.log(np.tan(np.pi/4.0 + np.radians(lat_max)/2.0)))

        self.width = 2000
        self.height = int(self.width * (y_max_merc - y_min_merc) / (lon_max - lon_min))

        grid_x_linear = np.linspace(lon_min, lon_max, self.width)
        grid_y_merc = np.linspace(y_max_merc, y_min_merc, self.height)
        self.grid_x, self.grid_y = np.meshgrid(grid_x_linear, grid_y_merc)

        points_merc = np.vstack((x_pts_merc, y_pts_merc)).T.astype(np.float64)
        self.total_dwd_points = len(points_merc)

        self.interpolator = LinearNDInterpolator(points_merc, np.zeros(self.total_dwd_points, dtype=np.float64))

        cluster_cols = np.floor((lon_deg - lon_min) / 1.0).astype(np.int32)
        cluster_rows = np.floor((lat_deg - lat_min) / 1.0).astype(np.int32)

        unique_clusters = np.unique(np.column_stack((cluster_cols, cluster_rows)), axis=0)

        self.cluster_mapping = {}
        for col, row in unique_clusters:
            point_indices = np.where((cluster_cols == col) & (cluster_rows == row))[0]
            self.cluster_mapping[(int(col), int(row))] = {
                "indices": point_indices,
                "lats": np.round(lat_deg[point_indices], 4).tolist(),
                "lons": np.round(lon_deg[point_indices], 4).tolist()
            }

        scipy_warmup_start_init = time.perf_counter()
        dummy_input_warmup = np.zeros((self.total_dwd_points, 1), dtype=np.float64)
        self.interpolator.values[:, 0] = dummy_input_warmup[:, 0]
        _ = self.interpolator(self.grid_x, self.grid_y)
        scipy_warmup_duration_init = time.perf_counter() - scipy_warmup_start_init
        print(f"    ⏱️ SciPy Interpolator Warmup (in init): {scipy_warmup_duration_init:.4f}s")

        init_duration = time.perf_counter() - init_start_time
        print(f"✅ [Processor] Gitter reaktiviert: {self.width}x{self.height} Pixel | {len(self.cluster_mapping)} Cluster bereit. (Init time: {init_duration:.4f}s)")

    def process_step(self, u_path, v_path, gust_path, time_key, png_filename):
        step_start_time = time.perf_counter()

        if not os.path.exists(u_path) or not os.path.exists(v_path) or not os.path.exists(gust_path):
            print(f"⚠️ GRIB2-Dateien für {png_filename} unvollständig. Überspringe Verarbeitung.")
            return False

        print(f"-> Berechne Wind und interpoliere für {png_filename}...")

        grib_load_start = time.perf_counter()
        with open(u_path, 'rb') as f:
            gid_u = ecc.codes_grib_new_from_file(f)
            u_values = ecc.codes_get_array(gid_u, 'values')
            try:
                u_missing_value = ecc.codes_get(gid_u, 'missingValue')
                u_values[u_values == u_missing_value] = np.nan
            except ecc.CodesInternalError:
                pass
            ecc.codes_release(gid_u)

        with open(v_path, 'rb') as f:
            gid_v = ecc.codes_grib_new_from_file(f)
            v_values = ecc.codes_get_array(gid_v, 'values')
            try:
                v_missing_value = ecc.codes_get(gid_v, 'missingValue')
                v_values[v_values == v_missing_value] = np.nan
            except ecc.CodesInternalError:
                pass
            ecc.codes_release(gid_v)

        # Read gust-component values
        with open(gust_path, 'rb') as f:
            gid_gust = ecc.codes_grib_new_from_file(f)
            gust_values = ecc.codes_get_array(gid_gust, 'values')
            try:
                gust_missing_value = ecc.codes_get(gid_gust, 'missingValue')
                gust_values[gust_values == gust_missing_value] = np.nan
            except ecc.CodesInternalError:
                pass
            ecc.codes_release(gid_gust)

        grib_load_duration = time.perf_counter() - grib_load_start
        print(f"    ⏱️ GRIB-Ladezeit: {grib_load_duration:.4f}s")

        wind_calc_start = time.perf_counter()
        wind_pts = np.sqrt(u_values**2 + v_values**2) * 1.94384
        min_len = min(self.total_dwd_points, len(wind_pts))
        current_wind_pts = wind_pts[:min_len].astype(np.float64)

        u_slice = u_values[:min_len].astype(np.float64)
        v_slice = v_values[:min_len].astype(np.float64)
        raw_dir_deg = 270.0 - np.degrees(np.arctan2(v_slice, u_slice))
        wind_dir_pts = np.mod(raw_dir_deg, 360.0)
        current_gust_pts = gust_values[:min_len].astype(np.float64) * 1.94384 # Convert gust to knots

        wind_calc_duration = time.perf_counter() - wind_calc_start
        print(f"    ⏱️ Windberechnung (Knots, Richtung, Böen): {wind_calc_duration:.4f}s")

        rounded_wind_pts = np.round(current_wind_pts, 1)
        rounded_wind_dir = np.round(wind_dir_pts, 0)
        rounded_gust_pts = np.round(current_gust_pts, 0)

        # ---------------------------------------------------------------------
        # TEIL A: PNG GENERIERUNG (Basiert weiterhin nur auf Geschwindigkeit)
        # ---------------------------------------------------------------------
        interp_start = time.perf_counter()
        self.interpolator.values[:, 0] = current_wind_pts
        grid_data = self.interpolator(self.grid_x, self.grid_y)
        interp_duration = time.perf_counter() - interp_start
        print(f"    ⏱️ Interpolation: {interp_duration:.4f}s")

        color_map_start = time.perf_counter()
        conditions = [
            grid_data < 3, grid_data < 5, grid_data < 6, grid_data < 7,
            grid_data < 8, grid_data < 9, grid_data < 10, grid_data < 12,
            grid_data < 15, grid_data < 20, grid_data < 25, grid_data >= 25
        ]

        color_palette = np.array([
            [0, 0, 0, 0], [230, 255, 255, 255], [0, 191, 255, 255],
            [0, 255, 204, 255], [0, 204, 0, 255], [153, 255, 0, 255],
            [255, 255, 0, 255], [209, 158, 0, 255], [255, 85, 0, 255],
            [255, 0, 0, 255], [255, 51, 153, 255], [153, 0, 204, 255],
            [0, 0, 255, 255]
        ], dtype=np.uint8)

        choices_indices = np.arange(1, len(conditions) + 1)
        selected_color_indices = np.select(conditions, choices_indices, default=0)
        img_array = color_palette[selected_color_indices]

        color_map_duration = time.perf_counter() - color_map_start
        print(f"    ⏱️ Color Mapping (np.select): {color_map_duration:.4f}s")

        png_save_start = time.perf_counter()
        img = Image.fromarray(img_array, 'RGBA')
        output_image_path = os.path.join(self.output_folder, png_filename)
        img.save(output_image_path, compress_level=6)
        png_save_duration = time.perf_counter() - png_save_start
        print(f"    ⏱️ PNG Speichern (compress_level=6): {png_save_duration:.4f}s")

        # ---------------------------------------------------------------------
        # TEIL B: EFFIZIENTES JSON-UPDATE IM RAM (OHNE REDUNDANZEN)
        # ---------------------------------------------------------------------
        json_agg_start = time.perf_counter()
        for (col, row), meta in self.cluster_mapping.items():
            cluster_key = (col, row)

            # Nur von Festplatte laden, wenn es noch nicht im RAM-Cache liegt
            if cluster_key not in self.cluster_memory:
                cluster_filename = os.path.join(self.cluster_output_folder, f"cluster_{col}_{row}.json")
                if os.path.exists(cluster_filename):
                    with open(cluster_filename, "rb") as jf:
                        try:
                            # Da das Format garantiert stimmt, laden wir direkt die bestehende Struktur
                            self.cluster_memory[cluster_key] = orjson.loads(jf.read())
                        except Exception:
                            # Fallback für unerwartete Ladefehler
                            self.cluster_memory[cluster_key] = {
                                "col": col, "row": row,
                                "lats": meta["lats"], "lons": meta["lons"],
                                "timeline": {}
                            }
                else:
                    self.cluster_memory[cluster_key] = {
                        "col": col, "row": row,
                        "lats": meta["lats"], "lons": meta["lons"],
                        "timeline": {}
                    }

            idx = meta["indices"]

            # Speicher- & Parsingeffiziente Zuweisung flacher Arrays
            self.cluster_memory[cluster_key]["timeline"][time_key] = {
                "speeds": rounded_wind_pts[idx],
                "dirs": rounded_wind_dir[idx],
                "gusts": rounded_gust_pts[idx]
            }

            # Hard-Rotation auf maximal x Stunden im RAM abfangen
            if len(self.cluster_memory[cluster_key]["timeline"]) > self.timeLineLength:
                sorted_keys = sorted(self.cluster_memory[cluster_key]["timeline"].keys())
                del self.cluster_memory[cluster_key]["timeline"][sorted_keys[0]]

        json_agg_duration = time.perf_counter() - json_agg_start
        print(f"    ⏱️ JSON RAM Aggregation (Flache Parallel-Arrays): {json_agg_duration:.4f}s")

        total_step_duration = time.perf_counter() - step_start_time
        print(f"    ⏱️ Total process_step duration: {total_step_duration:.4f}s")

        return True

    def flush_json_to_disk(self):
        flush_start_time = time.perf_counter()
        if not self.cluster_memory:
            print("💾 [Processor] Kein Cluster-Speicher zum Schreiben vorhanden.")
            return
        print(f"\n💾 [Processor] Schreibe {len(self.cluster_memory)} optimierte JSON-Cluster gesammelt in den lokalen Output-Ordner...")

        total_serialization_time = 0.0
        total_write_time = 0.0

        for (col, row), cluster_data in self.cluster_memory.items():
            cluster_filename = os.path.join(self.cluster_output_folder, f"cluster_{col}_{row}.json")

            serialization_start = time.perf_counter()
            # orjson liest die flachen NumPy-Arrays blitzschnell aus
            json_bytes = orjson.dumps(cluster_data, option=orjson.OPT_SERIALIZE_NUMPY)
            serialization_duration = time.perf_counter() - serialization_start
            total_serialization_time += serialization_duration

            write_start = time.perf_counter()
            with open(cluster_filename, "wb") as json_file:
                json_file.write(json_bytes)
            write_duration = time.perf_counter() - write_start
            total_write_time += write_duration

        flush_duration = time.perf_counter() - flush_start_time
        print(f"✅ Alle JSON-Files im lokalen Output-Ordner gespeichert! (Flush time: {flush_duration:.4f}s)")
        print(f"    ⏱️ Summe JSON Serialisierung (orjson): {total_serialization_time:.4f}s")
        print(f"    ⏱️ Summe File Writing (file.write): {total_write_time:.4f}s")
