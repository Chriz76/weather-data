import os
import json
import pickle
import xarray as xr
import numpy as np
from PIL import Image
import eccodes as ecc
from scipy.interpolate import LinearNDInterpolator
import time # Import time for detailed logging


class WindProcessor:
    # Modified __init__ to accept root_folder directly
    def __init__(self, root_folder, output_folder="./wind_tiles_simulation"):
        init_start_time = time.perf_counter()
        self.output_folder = output_folder
        self.cluster_output_folder = os.path.join(output_folder, "grid_cluster")
        os.makedirs(self.cluster_output_folder, exist_ok=True);
        self.root_folder = root_folder # Store root_folder

        self.cluster_memory = {}

        # --- Geometry and Interpolator Generation (moved from pre-computation) ---
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

        self.width = 2000 # Assign to self
        self.height = int(self.width * (y_max_merc - y_min_merc) / (lon_max - lon_min)) # Assign to self

        grid_x_linear = np.linspace(lon_min, lon_max, self.width)
        grid_y_merc = np.linspace(y_max_merc, y_min_merc, self.height)
        self.grid_x, self.grid_y = np.meshgrid(grid_x_linear, grid_y_merc) # Assign to self

        points_merc = np.vstack((x_pts_merc, y_pts_merc)).T.astype(np.float64)
        self.total_dwd_points = len(points_merc) # Assign to self

        self.interpolator = LinearNDInterpolator(points_merc, np.zeros(self.total_dwd_points, dtype=np.float64)) # Assign to self

        # Cluster mapping logic
        cluster_cols = np.floor((lon_deg - lon_min) / 1.0).astype(np.int32)
        cluster_rows = np.floor((lat_deg - lat_min) / 1.0).astype(np.int32)

        unique_clusters = np.unique(np.column_stack((cluster_cols, cluster_rows)), axis=0)

        self.cluster_mapping = {} # Assign to self
        for col, row in unique_clusters:
            point_indices = np.where((cluster_cols == col) & (cluster_rows == row))[0]
            self.cluster_mapping[(int(col), int(row))] = {
                "indices": point_indices,
                "lats": np.round(lat_deg[point_indices], 4).tolist(),
                "lons": np.round(lon_deg[point_indices], 4).tolist()
            }
        # --- End of Geometry and Interpolator Generation ---

        # Perform SciPy warmup after interpolator is created
        scipy_warmup_start_init = time.perf_counter()
        dummy_input_warmup = np.zeros((self.total_dwd_points, 1), dtype=np.float64)
        self.interpolator.values = dummy_input_warmup
        _ = self.interpolator(self.grid_x, self.grid_y)
        scipy_warmup_duration_init = time.perf_counter() - scipy_warmup_start_init
        print(f"   ⏱️ SciPy Interpolator Warmup (in init): {scipy_warmup_duration_init:.4f}s")

        init_duration = time.perf_counter() - init_start_time;
        print(f"✅ [Processor] Gitter reaktiviert: {self.width}x{self.height} Pixel | {len(self.cluster_mapping)} Cluster bereit. (Init time: {init_duration:.4f}s)")

    def process_step(self, u_path, v_path, time_key, png_filename):
        """
        Verarbeitet einen einzelnen Zeitschritt (U- und V-GRIB2-Datei).
        Generiert das PNG direkt und puffert das JSON-Update im RAM.
        """
        step_start_time = time.perf_counter()

        if not os.path.exists(u_path) or not os.path.exists(v_path):
            print(f"⚠️ GRIB2-Dateien für {png_filename} unvollständig. Überspringe Verarbeitung.")
            return False

        print(f"-> Berechne Wind & interpoliere für {png_filename}...")

        # 1. GRIB-Daten einlesen
        grib_load_start = time.perf_counter()

        # Read u-component values
        with open(u_path, 'rb') as f:
            gid_u = ecc.codes_grib_new_from_file(f)
            u_values = ecc.codes_get_array(gid_u, 'values')
            try:
                # Get missingValue from GRIB message and replace in array
                u_missing_value = ecc.codes_get(gid_u, 'missingValue')
                u_values[u_values == u_missing_value] = np.nan
            except ecc.CodesInternalError:
                # missingValue key might not exist, or values are already clean
                pass
            ecc.codes_release(gid_u)

        # Read v-component values
        with open(v_path, 'rb') as f:
            gid_v = ecc.codes_grib_new_from_file(f)
            v_values = ecc.codes_get_array(gid_v, 'values')
            try:
                # Get missingValue from GRIB message and replace in array
                v_missing_value = ecc.codes_get(gid_v, 'missingValue')
                v_values[v_values == v_missing_value] = np.nan
            except ecc.CodesInternalError:
                # missingValue key might not exist, or values are already clean
                pass
            ecc.codes_release(gid_v)

        grib_load_duration = time.perf_counter() - grib_load_start;
        print(f"   ⏱️ GRIB-Ladezeit: {grib_load_duration:.4f}s")

        # Windgeschwindigkeit in Knoten berechnen
        wind_calc_start = time.perf_counter()
        wind_pts = np.sqrt(u_values**2 + v_values**2) * 1.94384
        min_len = min(self.total_dwd_points, len(wind_pts));
        current_wind_pts = wind_pts[:min_len].astype(np.float64);
        wind_calc_duration = time.perf_counter() - wind_calc_start;
        print(f"   ⏱️ Windberechnung (Knots): {wind_calc_duration:.4f}s")

        # ---------------------------------------------------------------------
        # TEIL A: PNG GENERIERUNG
        # ---------------------------------------------------------------------
        interp_start = time.perf_counter();
        self.interpolator.values[:, 0] = current_wind_pts;
        grid_data = self.interpolator(self.grid_x, self.grid_y);
        interp_duration = time.perf_counter() - interp_start;
        print(f"   ⏱️ Interpolation: {interp_duration:.4f}s")

        # Farb-Kategorisierung mit np.select für Effizienz
        color_map_start = time.perf_counter();

        # Define conditions (order matters for overlapping conditions)
        conditions = [
            grid_data < 3,   # Lightest color for weakest winds
            grid_data < 5,
            grid_data < 6,
            grid_data < 7,
            grid_data < 8,
            grid_data < 9,
            grid_data < 10,
            grid_data < 12,
            grid_data < 15,
            grid_data < 20,
            grid_data < 25,
            grid_data >= 25  # Dark blue for strong winds
        ]

        # Define the actual RGBA color palette (transparent as first entry for default/NaNs)
        color_palette = np.array([
            [0, 0, 0, 0],        # Index 0: Transparent (for NaNs and default)
            [230, 255, 255, 255], # Index 1: < 3
            [0, 191, 255, 255],  # Index 2: < 5
            [0, 255, 204, 255],  # Index 3: < 6
            [0, 204, 0, 255],    # Index 4: < 7
            [153, 255, 0, 255],  # Index 5: < 8
            [255, 255, 0, 255],  # Index 6: < 9
            [209, 158, 0, 255],  # Index 7: < 10
            [255, 85, 0, 255],   # Index 8: < 12
            [255, 0, 0, 255],    # Index 9: < 15
            [255, 51, 153, 255], # Index 10: < 20
            [153, 0, 204, 255],  # Index 11: < 25
            [0, 0, 255, 255]     # Index 12: >= 25 (This is the dark blue you want to keep for strong winds)
        ], dtype=np.uint8)

        # The choices for np.select map to the color_palette indices (1 to 12 for actual wind colors).
        choices_indices = np.arange(1, len(conditions) + 1);

        # Create an array of integer indices using np.select.
        # default=0 means NaNs and any other values not matching a condition will get the transparent color (color_palette[0]).
        selected_color_indices = np.select(conditions, choices_indices, default=0);

        # Map these indices to the actual RGBA values
        img_array = color_palette[selected_color_indices];

        color_map_duration = time.perf_counter() - color_map_start;
        print(f"   ⏱️ Color Mapping (np.select): {color_map_duration:.4f}s")

        png_save_start = time.perf_counter();
        img = Image.fromarray(img_array, 'RGBA');
        output_image_path = os.path.join(self.output_folder, png_filename);
        img.save(output_image_path, compress_level=6);
        png_save_duration = time.perf_counter() - png_save_start;
        print(f"   ⏱️ PNG Speichern (compress_level=6): {png_save_duration:.4f}s")

        # ---------------------------------------------------------------------
        # TEIL B: JSON-UPDATE IM RAM PUFFERN (Keine Festplattenlast!)
        # ---------------------------------------------------------------------
        json_agg_start = time.perf_counter();
        for (col, row), meta in self.cluster_mapping.items():
            cluster_key = (col, row) # Use tuple as key for consistency

            # Wenn das Cluster noch nicht im RAM-Puffer ist, initialisieren
            if cluster_key not in self.cluster_memory:
                cluster_filename = os.path.join(self.cluster_output_folder, f"cluster_{col}_{row}.json");
                if os.path.exists(cluster_filename):
                    with open(cluster_filename, "r") as jf:
                        self.cluster_memory[cluster_key] = json.load(jf);
                else:
                    self.cluster_memory[cluster_key] = {
                        "col": col, "row": row,
                        "lats": meta["lats"], "lons": meta["lons"],
                        "timeline": {}
                    };

            idx = meta["indices"];
            cluster_winds = [
                None if np.isnan(w) else round(float(w), 1)
                for w in current_wind_pts[idx]
            ];

            # Im RAM erweitern
            self.cluster_memory[cluster_key][time_key] = cluster_winds;

            # Hard-Rotation auf 19 Stunden direkt im RAM
            if len(self.cluster_memory[cluster_key]["timeline"]) > 19:
                sorted_keys = sorted(self.cluster_memory[cluster_key]["timeline"].keys());
                del self.cluster_memory[cluster_key]["timeline"][sorted_keys[0]];
        json_agg_duration = time.perf_counter() - json_agg_start;
        print(f"   ⏱️ JSON RAM Aggregation: {json_agg_duration:.4f}s")

        total_step_duration = time.perf_counter() - step_start_time;
        print(f"   ⏱️ Total process_step duration: {total_step_duration:.4f}s")

        return True
    def flush_json_to_disk(self):
        """
        Schreibt alle im RAM modifizierten Cluster-Daten in einem Rutsch auf die Festplatte.
        """
        flush_start_time = time.perf_counter();
        if not self.cluster_memory:
            print("💾 [Processor] Kein Cluster-Speicher zum Schreiben vorhanden.");
            return;
        print(f"\n💾 [Processor] Schreibe {len(self.cluster_memory)} JSON-Cluster gesammelt in den lokalen Output-Ordner...");

        total_serialization_time = 0.0;
        total_write_time = 0.0;

        # Iterate using (col, row) as keys
        for (col, row), cluster_data in self.cluster_memory.items():
            cluster_filename = os.path.join(self.cluster_output_folder, f"cluster_{col}_{row}.json");

            serialization_start = time.perf_counter();
            json_string = json.dumps(cluster_data);
            serialization_duration = time.perf_counter() - serialization_start;
            total_serialization_time += serialization_duration;

            write_start = time.perf_counter();
            with open(cluster_filename, "w") as json_file:
                json_file.write(json_string);
            write_duration = time.perf_counter() - write_start;
            total_write_time += write_duration;

        flush_duration = time.perf_counter() - flush_start_time;
        print(f"✅ Alle JSON-Files im lokalen Output-Ordner gespeichert! (Flush time: {flush_duration:.4f}s)");
        print(f"   ⏱️ Summe JSON Serialisierung (json.dumps): {total_serialization_time:.4f}s");
        print(f"   ⏱️ Summe File Writing (file.write): {total_write_time:.4f}s");
