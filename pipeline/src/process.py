import os
import json
import pickle
import xarray as xr
import numpy as np
from PIL import Image
import eccodes as ecc
from scipy.interpolate import LinearNDInterpolator


class WindProcessor:
    def __init__(self, root_folder=".", output_folder="./wind_tiles_simulation"):
        self.root_folder = root_folder
        self.output_folder = output_folder
        self.cluster_output_folder = os.path.join(output_folder, "grid_cluster")
        os.makedirs(self.cluster_output_folder, exist_ok=True)
        
        self.index_input_path = os.path.join(root_folder, "static_weather_indices.pkl")
        
        # Zentraler In-Memory-Speicher, um I/O-Zugriffe auf das Drive zu minimieren
        self.cluster_memory = {}

        # =========================================================================

        # =========================================================================
        # PHASE 1: STATISCHE INDIZES LIVE RECHNEN
        # =========================================================================
        print("--- 🌍 [Processor] Berechne statische Gitter-Indizes live im RAM ---")
        
        current_script_dir = os.path.dirname(__file__)
        clat_path = os.path.join(current_script_dir, "clat.grib2")
        clon_path = os.path.join(current_script_dir, "clon.grib2")

        if not os.path.exists(clat_path) or not os.path.exists(clon_path):
            raise FileNotFoundError(f"Geometrie-Dateien fehlen im Ordner: {current_script_dir}")

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

        cluster_size = 1.0
        cluster_cols = np.floor((lon_deg - lon_min) / cluster_size).astype(np.int32)
        cluster_rows = np.floor((lat_deg - lat_min) / cluster_size).astype(np.int32)
        unique_clusters = np.unique(np.column_stack((cluster_cols, cluster_rows)), axis=0)

        self.cluster_mapping = {}
        for col, row in unique_clusters:
            point_indices = np.where((cluster_cols == col) & (cluster_rows == row))[0]
            self.cluster_mapping[(int(col), int(row))] = {
                "indices": point_indices,
                "lats": np.round(lat_deg[point_indices], 4).tolist(),
                "lons": np.round(lon_deg[point_indices], 4).tolist()
            }
        print(f"✅ Gitter reaktiviert: {self.width}x{self.height} Pixel | {len(self.cluster_mapping)} Cluster bereit.")
    
    def process_step(self, u_path, v_path, time_key, png_filename):
        """
        Verarbeitet einen einzelnen Zeitschritt (U- und V-GRIB2-Datei).
        Generiert das PNG direkt und puffert das JSON-Update im RAM.
        """
        if not os.path.exists(u_path) or not os.path.exists(v_path):
            print(f"⚠️ GRIB2-Dateien für {png_filename} unvollständig. Überspringe Verarbeitung.")
            return False

        print(f"-> Berechne Wind & interpoliere für {png_filename}...")
        
        # 1. GRIB-Daten einlesen
        ds_u = xr.open_dataset(u_path, engine="cfgrib")
        u_var = list(ds_u.data_vars)[0]
        u_values = ds_u[u_var].values.flatten()
        ds_u.close()

        ds_v = xr.open_dataset(v_path, engine="cfgrib")
        v_var = list(ds_v.data_vars)[0]
        v_values = ds_v[v_var].values.flatten()
        ds_v.close()

        # Windgeschwindigkeit in Knoten berechnen
        wind_pts = np.sqrt(u_values**2 + v_values**2) * 1.94384
        wind_dir_rad = np.arctan2(u_values, v_values)
        wind_dir_deg = (np.degrees(wind_dir_rad) + 180.0) % 360.0
        min_len = min(self.total_dwd_points, len(wind_pts))
        current_wind_pts = wind_pts[:min_len].astype(np.float64)
        current_wind_dirs = wind_dir_deg[:min_len].astype(np.float64)

        # ---------------------------------------------------------------------
        # TEIL A: PNG GENERIERUNG
        # ---------------------------------------------------------------------
        self.interpolator.values[:, 0] = current_wind_pts
        grid_data = self.interpolator(self.grid_x, self.grid_y)
        nan_mask = np.isnan(grid_data)

        # Farb-Kategorisierung (dein originaler Farbcode)
        img_array = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        img_array[:] = [0, 0, 255, 255]
        img_array[grid_data >= 25] = [0, 0, 255, 255]
        img_array[grid_data < 25]  = [153, 0, 204, 255]
        img_array[grid_data < 20]  = [255, 51, 153, 255]
        img_array[grid_data < 15]  = [255, 0, 0, 255]
        img_array[grid_data < 12]  = [255, 85, 0, 255]
        img_array[grid_data < 10]  = [209, 158, 0, 255]
        img_array[grid_data < 9]   = [255, 255, 0, 255]
        img_array[grid_data < 8]   = [153, 255, 0, 255]
        img_array[grid_data < 7]   = [0, 204, 0, 255]
        img_array[grid_data < 6]   = [0, 255, 204, 255]
        img_array[grid_data < 5]   = [0, 191, 255, 255]
        img_array[grid_data < 3]   = [230, 255, 255, 255]
        img_array[nan_mask] = [0, 0, 0, 0]

        img = Image.fromarray(img_array, 'RGBA')
        output_image_path = os.path.join(self.output_folder, png_filename)
        img.save(output_image_path)

        # ---------------------------------------------------------------------
        # TEIL B: JSON-UPDATE IM RAM PUFFERN (Optimiert & Vektoriell!)
        # ---------------------------------------------------------------------
        # 1. Globale Vorverarbeitung aller Punkte mit schnellem NumPy
        # Ersetzt NaN durch None direkt beim Erstellen der Listen (über einen Kniff)
        # Oder wir runden alles vorab:
        winds_rounded = np.where(np.isnan(current_wind_pts), -999, np.round(current_wind_pts, 1))
        dirs_rounded = np.where(np.isnan(current_wind_dirs), -999, np.round(current_wind_dirs))

        # 2. Schleife über die Cluster (jetzt ohne schwere Rechenlast)
        for (col, row), meta in self.cluster_mapping.items():
            cluster_key = f"{col}_{row}"
            
            # Puffer-Initialisierung (Nutzt den RAM-Cache)
            if cluster_key not in self.cluster_memory:
                cluster_filename = os.path.join(self.cluster_output_folder, f"cluster_{cluster_key}.json")
                if os.path.exists(cluster_filename):
                    with open(cluster_filename, "r") as jf:
                        self.cluster_memory[cluster_key] = json.load(jf)
                        if "directions" not in self.cluster_memory[cluster_key]:
                            self.cluster_memory[cluster_key]["directions"] = {}
                else:
                    self.cluster_memory[cluster_key] = {
                        "col": col, "row": row,
                        "lats": meta["lats"], "lons": meta["lons"],
                        "timeline": {}, "directions": {}
                    }

            idx = meta["indices"]
            
            # Slicing der bereits fertig berechneten NumPy-Arrays (Blitzschnell)
            c_winds = winds_rounded[idx]
            c_dirs = dirs_rounded[idx]

            # In native Python-Typen konvertieren und -999 zurück zu None wandeln
            cluster_winds = [None if w == -999 else float(w) for w in c_winds]
            cluster_dirs = [None if d == -999 else int(d) for d in c_dirs]
            
            # Daten im RAM zuweisen
            self.cluster_memory[cluster_key]["timeline"][time_key] = cluster_winds
            self.cluster_memory[cluster_key]["directions"][time_key] = cluster_dirs

            # Optimierte Rotation: Da wir chronologisch einfügen, 
            # ist der älteste Eintrag immer der erste nach dem Sortieren der Keys.
            # Wenn wir wissen, dass wir > 19 sind, löschen wir einfach das älteste.
            timeline_dict = self.cluster_memory[cluster_key]["timeline"]
            if len(timeline_dict) > 19:
                oldest_key = min(timeline_dict.keys()) # min() ist schneller als sorted()
                del timeline_dict[oldest_key]
                if oldest_key in self.cluster_memory[cluster_key]["directions"]:
                    del self.cluster_memory[cluster_key]["directions"][oldest_key]
                
        return True
    def flush_json_to_disk(self):
        """
        Schreibt alle im RAM modifizierten Cluster-Daten in einem Rutsch auf die Festplatte.
        """
        if not self.cluster_memory:
            return
        print(f"\n💾 [Processor] Schreibe {len(self.cluster_memory)} JSON-Cluster gesammelt auf Google Drive...")
        for cluster_key, cluster_data in self.cluster_memory.items():
            cluster_filename = os.path.join(self.cluster_output_folder, f"cluster_{cluster_key}.json")
            with open(cluster_filename, "w") as json_file:
                json.dump(cluster_data, json_file)
        print("✅ Alle JSON-Files mit Google Drive synchronisiert!")

