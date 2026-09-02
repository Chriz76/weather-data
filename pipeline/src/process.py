import os
import time
import numpy as np
from PIL import Image
from omfiles import OmFileReader
from scipy.ndimage import map_coordinates


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

        grid_x_linear = np.linspace(self.lon_min, self.lon_max, self.width)
        grid_y_merc = np.linspace(y_min_merc, y_max_merc, self.height)

        grid_x, grid_y = np.meshgrid(grid_x_linear, grid_y_merc)

        # 2. Rücktransformation der Mercator-Y-Pixel in echte WGS84-Latitudes
        lat_source = np.degrees(2 * np.arctan(np.exp(np.radians(grid_y))) - np.pi / 2.0)
        lon_source = grid_x

        # 3. Indizes für das AROME-Quellraster
        row_indices = (self.lat_max - lat_source) / (self.lat_max - self.lat_min) * (self.src_lat_shape - 1)
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

        step_duration = time.perf_counter() - step_start_time
        print(f"    ⏱️ Dauer: {step_duration:.3f}s")
        return True
