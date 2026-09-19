import os
import math
import time
import numpy as np
from PIL import Image
from omfiles import OmFileReader
from scipy.ndimage import map_coordinates
import io
from pmtiles.writer import Writer
from pmtiles.tile import TileType, Compression, zxy_to_tileid
import requests
import gzip
from pmtiles.tile import TileType, Compression, zxy_to_tileid


# --- 1. GEO-HILFSFUNKTIONEN ---

def tile_bounds_wgs84(z, x, y):
    """Berechnet die exakte Bounding-Box (lon_min, lat_min, lon_max, lat_max)
    einer Web-Mercator Kachel in WGS84 Grad.
    """
    n = 2.0 ** z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0

    lat_rad_max = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat_rad_min = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n)))

    return lon_min, math.degrees(lat_rad_min), lon_max, math.degrees(lat_rad_max)


class AromeWindProcessor:
    def __init__(self, output_folder="./wind_tiles_arome", width=2000):
        init_start_time = time.perf_counter()
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.width = width

        # AROME Bounding-Box aus WGS84
        self.lat_min, self.lat_max = 37.5, 55.4
        self.lon_min, self.lon_max = -12.0, 16.0

        # Quell-Rasterdimensionen (AROME 0.025°)
        self.src_lat_shape = 717
        self.src_lon_shape = 1121

        # Mercator-Y Berechnungen
        y_min_merc = np.degrees(np.log(np.tan(np.pi / 4.0 + np.radians(self.lat_min) / 2.0)))
        y_max_merc = np.degrees(np.log(np.tan(np.pi / 4.0 + np.radians(self.lat_max) / 2.0)))

        # Höhe proportional zur Mercator-Verzerrung
        self.height = int(self.width * (y_max_merc - y_min_merc) / (self.lon_max - self.lon_min))

        # 1. Ziel-Grid in Web-Mercator definieren (Pixel 0 oben = Nord = y_max_merc)
        grid_x_linear = np.linspace(self.lon_min, self.lon_max, self.width)
        grid_y_merc = np.linspace(y_max_merc, y_min_merc, self.height)

        grid_x, grid_y = np.meshgrid(grid_x_linear, grid_y_merc)

        # 2. Rücktransformation der Mercator-Y-Pixel in echte WGS84-Latitudes
        lat_source = np.degrees(2 * np.arctan(np.exp(np.radians(grid_y))) - np.pi / 2.0)
        lon_source = grid_x

        # 3. Indizes für das AROME-Quellraster
        row_indices = (lat_source - self.lat_min) / (self.lat_max - self.lat_min) * (self.src_lat_shape - 1)
        col_indices = (lon_source - self.lon_min) / (self.lon_max - self.lon_min) * (self.src_lon_shape - 1)

        # Fertige Lookup-Matrix
        self.interp_coords = np.array([row_indices, col_indices])

        # Farbschema
        self.color_palette = np.array([
            [0, 0, 0, 0],         # 0: Out of bounds / NaN
            [230, 255, 255, 255], # 1: < 3 Knots
            [0, 191, 255, 255],   # 2: < 5
            [0, 255, 204, 255],   # 3: < 6
            [0, 204, 0, 255],     # 4: < 7
            [153, 255, 0, 255],   # 5: < 8
            [255, 255, 0, 255],   # 6: < 9
            [209, 158, 0, 255],   # 7: < 10
            [255, 85, 0, 255],    # 8: < 12
            [255, 0, 0, 255],     # 9: < 15
            [255, 51, 153, 255],  # 10: < 20
            [153, 0, 204, 255],   # 11: < 25
            [0, 0, 255, 255]      # 12: >= 25
        ], dtype=np.uint8)

        init_duration = time.perf_counter() - init_start_time
        print(f"✅ [AromeProcessor] Ziel-Gitter initialisiert: {self.width}x{self.height} Pixel (Init: {init_duration:.4f}s)")

    def _create_wind_direction_pmtiles(self, u_raw, v_raw, output_pmtiles_path, min_zoom=0, max_zoom=8):
        """Erzeugt binäre Float32 PMTiles mit Gzip-Komprimierung und 24-Byte Header."""

        # 1D Koordinatenvektoren (North-Up im Ursprungskontext)
        lats = np.linspace(self.lat_min, self.lat_max, self.src_lat_shape)
        lons = np.linspace(self.lon_min, self.lon_max, self.src_lon_shape)

        tiles_dict = {}

        # Schleife über Web-Mercator Kacheln je Zoomstufe
        for z in range(min_zoom, max_zoom + 1):
            stride = 2 ** (max_zoom - z)

            # Subsampling je nach Zoom-Level
            u_lod = u_raw[::stride, ::stride]
            v_lod = v_raw[::stride, ::stride]
            lats_lod = lats[::stride]
            lons_lod = lons[::stride]

            n = 2 ** z
            for x in range(n):
                for y in range(n):
                    t_lon_min, t_lat_min, t_lon_max, t_lat_max = tile_bounds_wgs84(z, x, y)

                    if (t_lon_max < self.lon_min or t_lon_min > self.lon_max or
                        t_lat_max < self.lat_min or t_lat_min > self.lat_max):
                        continue

                    col_indices = np.where((lons_lod >= t_lon_min) & (lons_lod <= t_lon_max))[0]
                    row_indices = np.where((lats_lod >= t_lat_min) & (lats_lod <= t_lat_max))[0]

                    if len(col_indices) == 0 or len(row_indices) == 0:
                        continue

                    c_start = col_indices[0]
                    c_end = col_indices[-1] + 1

                    # Lat-Indizes für North-Up Ausrichtung
                    r_start_idx = row_indices[-1]
                    r_end_idx = row_indices[0]

                    # Teilbereich ausschneiden
                    sub_u = u_lod[r_end_idx:r_start_idx + 1, c_start:c_end]
                    sub_v = v_lod[r_end_idx:r_start_idx + 1, c_start:c_end]

                    # Zeilen umkehren für North-Up (nördlichste Breite = Zeile 0)
                    sub_u = np.flipud(sub_u)
                    sub_v = np.flipud(sub_v)

                    rows, cols = sub_u.shape

                    if rows == 0 or cols == 0:
                        continue

                    # Bounding-Check auf valide Vektoren
                    valid_mask = ~np.isnan(sub_u) & ~np.isnan(sub_v)
                    if not np.any(valid_mask):
                        continue

                    # 1. HEADER (6x Float32 = 24 Bytes)
                    origin_lng = float(lons_lod[c_start])
                    origin_lat = float(lats_lod[r_start_idx]) # Nördlichste Breite

                    delta_lng = float(lons_lod[1] - lons_lod[0]) if len(lons_lod) > 1 else 0.025 * stride
                    delta_lat = float(lats_lod[1] - lats_lod[0]) if len(lats_lod) > 1 else 0.025 * stride

                    header_meta = np.array([
                        origin_lng,
                        origin_lat,
                        delta_lng,
                        delta_lat,
                        float(rows),
                        float(cols)
                    ], dtype=np.float32)

                    # 2. PAYLOAD (u, v verschachtelt als Float32 Array)
                    # Form: [u0, v0, u1, v1, u2, v2, ...]
                    uv_interleaved = np.empty((rows, cols, 2), dtype=np.float32)
                    uv_interleaved[:, :, 0] = sub_u
                    uv_interleaved[:, :, 1] = sub_v

                    raw_tile_bytes = header_meta.tobytes() + uv_interleaved.tobytes()

                    # 3. GZIP KOMPRESSION ANWENDEN
                    compressed_tile_bytes = gzip.compress(raw_tile_bytes)

                    tiles_dict[zxy_to_tileid(z, x, y)] = compressed_tile_bytes

        # PMTiles schreiben
        with open(output_pmtiles_path, "wb") as f:
            writer = Writer(f)

            for tile_id in sorted(tiles_dict.keys()):
                writer.write_tile(tile_id, tiles_dict[tile_id])

            header = {
                "tile_type": TileType.UNKNOWN,
                "tile_compression": Compression.GZIP,
                "min_zoom": min_zoom,
                "max_zoom": max_zoom,
                "min_lon": self.lon_min,
                "min_lat": self.lat_min,
                "max_lon": self.lon_max,
                "max_lat": self.lat_max,
                "center_zoom": 5,
                "center_lon": (self.lon_min + self.lon_max) / 2.0,
                "center_lat": (self.lat_min + self.lat_max) / 2.0
            }

            metadata = {
                "name": "AROME Wind Vector Binary PMTiles (Gzip)",
                "format": "binary",
                "description": "24 Byte Float32 Header + Interleaved Float32 (U, V) Payload mit Gzip-Kompression."
            }

            writer.finalize(header, metadata)

        file_size_kb = os.path.getsize(output_pmtiles_path) / 1024.0
        print(f"✅ PMTiles Container erfolgreich erstellt: {output_pmtiles_path}")
        print(f"💾 Gesamtgröße: {file_size_kb:.2f} KB")
        return True

    def process_om_file(self, om_path, output_filename=None):
        step_start_time = time.perf_counter()

        if not os.path.exists(om_path):
            print(f"⚠️ Datei {om_path} nicht gefunden.")
            return False

        if output_filename is None:
            base_name = os.path.splitext(os.path.basename(om_path))[0]
            output_filename = f"{base_name}.webp"

        print(f"-> Verarbeite: {os.path.basename(om_path)} -> {output_filename}...")

        with OmFileReader(om_path) as root:
            u_node = root.get_child_by_name("wind_u_component_10m")
            v_node = root.get_child_by_name("wind_v_component_10m")

            u_raw = u_node.read_array(...)
            v_raw = v_node.read_array(...)

        # Process for WEBP (existing logic)
        u_clean = np.nan_to_num(u_raw, nan=0.0)
        v_clean = np.nan_to_num(v_raw, nan=0.0)

        wind_speed_knots = np.sqrt(u_clean**2 + v_clean**2) * 1.94384
        valid_mask = ~np.isnan(u_raw)

        grid_data = map_coordinates(wind_speed_knots, self.interp_coords, order=1, mode='nearest')
        grid_valid = map_coordinates(valid_mask.astype(float), self.interp_coords, order=0, mode='nearest')

        conditions = [
            (grid_data < 3) & (grid_valid > 0.5),
            (grid_data < 5) & (grid_valid > 0.5),
            (grid_data < 6) & (grid_valid > 0.5),
            (grid_data < 7) & (grid_valid > 0.5),
            (grid_data < 8) & (grid_valid > 0.5),
            (grid_data < 9) & (grid_valid > 0.5),
            (grid_data < 10) & (grid_valid > 0.5),
            (grid_data < 12) & (grid_valid > 0.5),
            (grid_data < 15) & (grid_valid > 0.5),
            (grid_data < 20) & (grid_valid > 0.5),
            (grid_data < 25) & (grid_valid > 0.5),
            (grid_data >= 25) & (grid_valid > 0.5)
        ]

        choices_indices = np.arange(1, len(conditions) + 1)
        selected_color_indices = np.select(conditions, choices_indices, default=0)
        img_array = self.color_palette[selected_color_indices]

        img = Image.fromarray(img_array, 'RGBA')
        output_webp_path = os.path.join(self.output_folder, output_filename)
        img.save(output_webp_path, format="WEBP", lossless=True, method=4)

        success_webp = True # Assuming webp processing is always successful for now

        # Process for PMTiles (new logic)
        pmtiles_filename = output_filename.replace('.webp', '_dir.pmtiles')
        output_pmtiles_path = os.path.join(self.output_folder, pmtiles_filename)
        success_pmtiles = self._create_wind_direction_pmtiles(u_raw, v_raw, output_pmtiles_path)

        if os.path.exists(om_path):
            os.remove(om_path)

        step_duration = time.perf_counter() - step_start_time
        print(f"    ⏱️ Dauer: {step_duration:.3f}s")
        return success_webp and success_pmtiles
