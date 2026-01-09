"""
Semantic-Aware Rain Simulation for Mining Site Data Augmentation
=================================================================
Uses DeepLabV3 semantic segmentation to detect roads, vehicles, and vegetation.
Applies appropriate rain effects to each region.
"""

import cv2
import numpy as np
import os
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


# Global model cache to avoid reloading for each image
_SEGMENTATION_MODEL = None
_DEVICE = None


def get_segmentation_model():
    """Load DeepLabV3 model (cached for efficiency)."""
    global _SEGMENTATION_MODEL, _DEVICE
    
    if _SEGMENTATION_MODEL is None:
        print("Loading DeepLabV3 segmentation model...")
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {_DEVICE}")
        
        # Load pre-trained DeepLabV3 with ResNet50 backbone
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        _SEGMENTATION_MODEL = deeplabv3_resnet50(weights=weights)
        _SEGMENTATION_MODEL.to(_DEVICE)
        _SEGMENTATION_MODEL.eval()
        print("Model loaded successfully!")
    
    return _SEGMENTATION_MODEL, _DEVICE


class RegionDetector:
    """
    Detects semantic regions using DeepLabV3 deep learning model.
    
    MANUAL ROAD MASK: Place 'road_mask.png' in the script directory.
    - White (255) = Road area
    - Black (0) = Not road
    If provided, automatic road detection is skipped.
    """
    
    # COCO class indices (used by torchvision DeepLabV3)
    VEHICLE_CLASSES = [2, 7, 3, 8, 4, 6, 1]  # bicycle, car, motorcycle, truck, etc.
    VEGETATION_CLASSES = [9]  # potted plant (limited in COCO)
    
    # Manual road mask path (set once, used for all images)
    _manual_road_mask = None
    _manual_mask_loaded = False
    
    @classmethod
    def load_manual_road_mask(cls, script_dir):
        """Load manual road mask if it exists."""
        if cls._manual_mask_loaded:
            return
        
        mask_path = os.path.join(script_dir, "road_mask.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                cls._manual_road_mask = mask
                print(f"Loaded manual road mask: {mask_path}")
            else:
                print(f"Warning: Could not read road_mask.png")
        else:
            print("No road_mask.png found - using automatic detection")
        
        cls._manual_mask_loaded = True
    
    def __init__(self, img):
        self.img = img
        self.height, self.width = img.shape[:2]
        self.hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Run segmentation
        self._segmentation_mask = self._run_segmentation()
    
    def _run_segmentation(self):
        """Run DeepLabV3 segmentation on the image."""
        model, device = get_segmentation_model()
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
        # Preprocessing
        preprocess = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = preprocess(img_rgb).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        
        # Get class predictions
        predictions = output.argmax(0).cpu().numpy()
        
        return predictions
    
    def detect_road(self):
        """
        Detect road/ground areas.
        Uses manual mask if available, otherwise automatic detection.
        """
        # Check for manual road mask first
        if RegionDetector._manual_road_mask is not None:
            mask = RegionDetector._manual_road_mask
            # Resize if dimensions don't match
            if mask.shape[:2] != (self.height, self.width):
                mask = cv2.resize(mask, (self.width, self.height))
            return cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Automatic detection for mining roads
        return self._detect_road_auto()
    
    def _detect_road_auto(self):
        """
        Automatic road detection for MINING SITES.
        Mining roads are unpaved - made of sand, soil, gravel.
        Uses wider color ranges for earthy tones.
        """
        h, s, v = cv2.split(self.hsv)
        
        # === MINING ROAD COLORS ===
        # Sand/Beige: low hue (yellow-brown), moderate saturation
        sand_hue = ((h >= 10) & (h <= 35)).astype(np.uint8)
        sand_sat = ((s >= 20) & (s <= 150)).astype(np.uint8)
        sand_mask = (sand_hue & sand_sat).astype(np.uint8) * 255
        
        # Soil/Brown: similar hue range, higher saturation
        soil_hue = ((h >= 5) & (h <= 25)).astype(np.uint8)
        soil_sat = ((s >= 30) & (s <= 180)).astype(np.uint8)
        soil_mask = (soil_hue & soil_sat).astype(np.uint8) * 255
        
        # Grey/Dusty roads: very low saturation (neutral colors)
        grey_mask = (s < 50).astype(np.uint8) * 255
        
        # Combine all road colors (OR logic)
        road_colors = cv2.bitwise_or(sand_mask, soil_mask)
        road_colors = cv2.bitwise_or(road_colors, grey_mask)
        
        # Brightness filter (not too dark, not too bright)
        brightness_mask = ((v > 40) & (v < 240)).astype(np.uint8) * 255
        road_mask = cv2.bitwise_and(road_colors, brightness_mask)
        
        # Exclude detected vehicles and vegetation
        vehicle_mask = self.detect_vehicles()
        vegetation_mask = self.detect_vegetation()
        
        road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(vehicle_mask))
        road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(vegetation_mask))
        
        # Exclude sky region
        sky_mask = self.detect_sky()
        road_mask = cv2.bitwise_and(road_mask, cv2.bitwise_not(sky_mask))
        
        # Morphological cleanup - close gaps, remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Keep larger regions only (roads are continuous)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask)
        final_mask = np.zeros_like(road_mask)
        min_area = self.height * self.width * 0.01  # Lower threshold for mining roads
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                final_mask[labels == i] = 255
        
        # Fallback if nothing detected
        if np.sum(final_mask) < min_area:
            final_mask = road_mask
        
        return cv2.GaussianBlur(final_mask, (21, 21), 0)
    
    def detect_vehicles(self):
        """Detect vehicles using DeepLabV3 segmentation."""
        vehicle_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for cls in self.VEHICLE_CLASSES:
            vehicle_mask[self._segmentation_mask == cls] = 255
        
        # Dilate slightly to capture full vehicle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        vehicle_mask = cv2.dilate(vehicle_mask, kernel)
        
        return cv2.GaussianBlur(vehicle_mask, (15, 15), 0)
    
    def detect_vegetation(self):
        """
        Detect grass/vegetation using green color range.
        """
        # Green color range in HSV
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([90, 255, 200])
        green_mask = cv2.inRange(self.hsv, lower_green, upper_green)
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.GaussianBlur(green_mask, (11, 11), 0)
        
        return green_mask
    
    def detect_sky(self):
        """
        Detect sky region (top area with blue/white/grey colors).
        """
        # Position mask: top 40% of image
        position_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        end_row = int(self.height * 0.4)
        position_mask[:end_row, :] = 255
        
        # Sky colors: high value, low-medium saturation
        lower_sky = np.array([0, 0, 120])
        upper_sky = np.array([180, 100, 255])
        color_mask = cv2.inRange(self.hsv, lower_sky, upper_sky)
        
        sky_mask = cv2.bitwise_and(color_mask, position_mask)
        sky_mask = cv2.GaussianBlur(sky_mask, (31, 31), 0)
        
        return sky_mask
    
    def estimate_depth(self):
        """
        Estimate depth based on vertical position.
        Top of image = far (high depth value), bottom = near (low depth).
        Returns normalized depth map (0-1).
        """
        depth_map = np.linspace(1.0, 0.0, self.height).reshape(-1, 1)
        depth_map = np.tile(depth_map, (1, self.width))
        return depth_map.astype(np.float32)
    
    def get_all_masks(self):
        """Return all detected region masks."""
        return {
            'road': self.detect_road(),
            'vegetation': self.detect_vegetation(),
            'sky': self.detect_sky(),
            'vehicle': self.detect_vehicles(),
            'depth': self.estimate_depth()
        }


class RainEffects:
    """
    Apply region-specific rain effects.
    """
    
    @staticmethod
    def wet_road(img, road_mask):
        """
        Make road appear wet: strong darkening + specular highlights + wet sheen.
        """
        img_float = img.astype(np.float32)
        mask_norm = road_mask.astype(np.float32) / 255.0
        
        # Strong darkening (wet surfaces absorb light significantly)
        darkening = 0.72
        for c in range(3):
            img_float[:, :, c] = img_float[:, :, c] * (1 - mask_norm * (1 - darkening))
        
        # Add wet sheen - subtle glossy effect
        # Create smooth gradient for consistent wet look
        rows = img.shape[0]
        wet_gradient = np.linspace(0.0, 0.12, rows).reshape(-1, 1)
        wet_gradient = np.tile(wet_gradient, (1, img.shape[1]))
        wet_sheen = wet_gradient * mask_norm
        
        for c in range(3):
            img_float[:, :, c] = np.clip(img_float[:, :, c] + wet_sheen * 40, 0, 255)
        
        # Add specular highlights (wet shine spots)
        noise = np.random.rand(img.shape[0], img.shape[1]).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (7, 7), 0)
        specular_mask = (noise > 0.82).astype(np.float32) * mask_norm * 0.20
        
        for c in range(3):
            img_float[:, :, c] = np.clip(img_float[:, :, c] + specular_mask * 70, 0, 255)
        
        return img_float.astype(np.uint8)
    
    @staticmethod
    def add_water_puddles(img, road_mask, puddle_density=0.555):
        """
        Add water puddles/potholes on the road with reflections.
        
        Args:
            img: Input image
            road_mask: Road region mask
            puddle_density: 0.0-1.0, how much of road area has puddles
        """
        rows, cols = img.shape[:2]
        img_float = img.astype(np.float32)
        mask_norm = road_mask.astype(np.float32) / 255.0
        
        # Generate irregular puddle shapes using noise
        puddle_noise = np.random.rand(rows, cols).astype(np.float32)
        
        # Blur to create blob shapes
        puddle_noise = cv2.GaussianBlur(puddle_noise, (45, 45), 0)
        
        # Threshold to create puddle regions
        puddle_threshold = 1.0 - puddle_density
        puddle_mask = (puddle_noise > puddle_threshold).astype(np.float32)
        
        # Apply only within road
        puddle_mask = puddle_mask * mask_norm
        
        # Smooth edges
        puddle_mask = cv2.GaussianBlur(puddle_mask, (15, 15), 0)
        
        # === Create reflection effect ===
        # Flip image vertically for reflection
        reflection = cv2.flip(img, 0).astype(np.float32)
        
        # Blur reflection (water surface distortion)
        reflection = cv2.GaussianBlur(reflection, (9, 9), 0)
        
        # Reduce reflection brightness (water absorbs some light)
        reflection = reflection * 0.45
        
        # Add water tint (slight blue)
        reflection[:, :, 0] = np.clip(reflection[:, :, 0] + 15, 0, 255)  # Blue
        reflection[:, :, 2] = np.clip(reflection[:, :, 2] - 10, 0, 255)  # Red
        
        # Darken puddle areas (water is darker than road)
        puddle_darken = img_float * 0.65
        
        # Blend: puddle base + reflection
        puddle_blend = puddle_darken * 0.6 + reflection * 0.4
        
        # Apply puddle effect
        puddle_mask_3ch = np.stack([puddle_mask] * 3, axis=-1)
        result = img_float * (1 - puddle_mask_3ch) + puddle_blend * puddle_mask_3ch
        
        # Add subtle specular highlights on puddles (sky reflection)
        specular_noise = np.random.rand(rows, cols).astype(np.float32)
        specular_noise = cv2.GaussianBlur(specular_noise, (5, 5), 0)
        specular = (specular_noise > 0.92).astype(np.float32) * puddle_mask * 0.3
        
        for c in range(3):
            result[:, :, c] = np.clip(result[:, :, c] + specular * 80, 0, 255)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def wet_vegetation(img, veg_mask):
        """
        Make vegetation appear wet: slight darkening, no shine.
        """
        img_float = img.astype(np.float32)
        mask_norm = veg_mask.astype(np.float32) / 255.0
        
        # Slight darkening
        for c in range(3):
            img_float[:, :, c] = img_float[:, :, c] * (1 - mask_norm * 0.1)
        
        return img_float.astype(np.uint8)
    
    @staticmethod
    def apply_depth_haze(img, depth_map, intensity=0.35):
        """
        Apply fog/haze based on depth. Distant objects (high depth) get more haze.
        """
        img_float = img.astype(np.float32) / 255.0
        
        # Fog color (cool grey-blue)
        fog_color = np.array([0.78, 0.76, 0.73])  # BGR
        
        # Haze intensity based on depth
        haze_amount = depth_map * intensity
        haze_amount = np.stack([haze_amount] * 3, axis=-1)
        
        # Blend: lerp between original and fog color
        img_hazed = img_float * (1 - haze_amount) + fog_color * haze_amount
        
        return np.clip(img_hazed * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def reduce_distant_visibility(img, depth_map, vehicle_mask):
        """
        Reduce visibility of distant vehicles specifically.
        """
        img_float = img.astype(np.float32)
        
        # Combine depth and vehicle mask
        vehicle_norm = vehicle_mask.astype(np.float32) / 255.0
        combined = depth_map * vehicle_norm
        
        # Blur distant vehicles more
        blurred = cv2.GaussianBlur(img, (15, 15), 0).astype(np.float32)
        
        combined_3ch = np.stack([combined] * 3, axis=-1)
        img_float = img_float * (1 - combined_3ch * 0.5) + blurred * (combined_3ch * 0.5)
        
        return img_float.astype(np.uint8)
    
    @staticmethod
    def generate_rain_streaks(shape, density=0.006):
        """
        Generate multi-layer rain streaks with depth variation.
        """
        rows, cols = shape[:2]
        rain_layer = np.zeros((rows, cols), dtype=np.float32)
        
        # Layer 1: Background rain (distant, small, faint)
        for _ in range(int(density * rows * cols * 0.5)):
            x = np.random.randint(0, cols)
            y = np.random.randint(0, rows)
            length = np.random.randint(4, 10)
            cv2.line(rain_layer, (x, y), (x + 1, y + length), 0.25, 1, cv2.LINE_AA)
        
        # Layer 2: Midground rain
        for _ in range(int(density * rows * cols * 0.35)):
            x = np.random.randint(0, cols)
            y = np.random.randint(0, rows)
            length = np.random.randint(12, 22)
            cv2.line(rain_layer, (x, y), (x + 2, y + length), 0.45, 1, cv2.LINE_AA)
        
        # Layer 3: Foreground rain (reduced size for more realism)
        for _ in range(int(density * rows * cols * 0.15)):
            x = np.random.randint(0, cols)
            y = np.random.randint(0, rows)
            length = np.random.randint(15, 25)  # Reduced from 25-40
            cv2.line(rain_layer, (x, y), (x + 2, y + length), 0.55, 1, cv2.LINE_AA)  # Thinner
        
        # Motion blur
        kernel_size = 7
        kernel = np.zeros((kernel_size, kernel_size))
        np.fill_diagonal(kernel, 1)
        kernel /= kernel_size
        rain_layer = cv2.filter2D(rain_layer, -1, kernel)
        
        return rain_layer
    
    @staticmethod
    def apply_rain_overlay(img, rain_layer, sky_mask):
        """
        Overlay rain streaks on image. Rain is more visible against sky.
        """
        img_float = img.astype(np.float32) / 255.0
        sky_norm = sky_mask.astype(np.float32) / 255.0
        
        # Rain visibility boost in sky area
        rain_boost = 1.0 + sky_norm * 0.3
        rain_adjusted = rain_layer * rain_boost
        
        rain_3ch = np.stack([rain_adjusted] * 3, axis=-1)
        
        # Screen blend
        result = 1 - (1 - img_float) * (1 - rain_3ch * 0.7)
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_atmosphere(img):
        """
        Global atmospheric adjustments: cool tone, reduced contrast, desaturation.
        """
        # Reduce contrast
        img_float = img.astype(np.float32) / 255.0
        img_float = (img_float - 0.5) * 0.88 + 0.5
        img_uint8 = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        
        # Desaturate in HSV
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.78
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
        
        # Cool tone
        img_out[:, :, 0] = np.clip(img_out[:, :, 0] + 6, 0, 255)  # Blue
        img_out[:, :, 2] = np.clip(img_out[:, :, 2] - 4, 0, 255)  # Red
        
        return img_out.astype(np.uint8)


def apply_rain(img):
    """
    Main function: Apply complete semantic-aware rain simulation.
    
    Pipeline:
    1. Detect regions (road, vegetation, sky, vehicles, depth)
    2. Apply global atmospheric effects
    3. Apply region-specific wetness
    4. Add water puddles on road
    5. Apply depth-based haze
    6. Reduce distant vehicle visibility
    7. Add rain streaks
    """
    # Step 1: Detect regions
    detector = RegionDetector(img)
    masks = detector.get_all_masks()
    
    # Step 2: Global atmosphere
    result = RainEffects.apply_atmosphere(img)
    
    # Step 3: Wet surfaces
    result = RainEffects.wet_road(result, masks['road'])
    result = RainEffects.wet_vegetation(result, masks['vegetation'])
    
    # Step 4: Add water puddles on road
    result = RainEffects.add_water_puddles(result, masks['road'], puddle_density=0.52)
    
    # Step 5: Depth-based haze
    result = RainEffects.apply_depth_haze(result, masks['depth'], intensity=0.30)
    
    # Step 6: Reduce distant vehicle visibility
    result = RainEffects.reduce_distant_visibility(result, masks['depth'], masks['vehicle'])
    
    # Step 7: Add rain streaks
    rain = RainEffects.generate_rain_streaks(img.shape)
    result = RainEffects.apply_rain_overlay(result, rain, masks['sky'])
    
    return result


def debug_visualize_masks(img):
    """
    Debug function: Visualize detected regions with color overlay.
    - Road: Blue
    - Vegetation: Green  
    - Sky: Cyan
    - Vehicle: Red
    """
    detector = RegionDetector(img)
    masks = detector.get_all_masks()
    
    # Create overlay
    overlay = img.copy().astype(np.float32)
    
    # Road mask - Blue
    road_norm = masks['road'].astype(np.float32) / 255.0
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + road_norm * 100, 0, 255)  # Blue
    
    # Vegetation mask - Green
    veg_norm = masks['vegetation'].astype(np.float32) / 255.0
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + veg_norm * 100, 0, 255)  # Green
    
    # Sky mask - Cyan
    sky_norm = masks['sky'].astype(np.float32) / 255.0
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + sky_norm * 80, 0, 255)  # Blue
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + sky_norm * 80, 0, 255)  # Green
    
    # Vehicle mask - Red
    veh_norm = masks['vehicle'].astype(np.float32) / 255.0
    overlay[:, :, 2] = np.clip(overlay[:, :, 2] + veh_norm * 100, 0, 255)  # Red
    
    return overlay.astype(np.uint8)


def process_batch_debug(input_dir, output_dir):
    """Process images with debug visualization to check region detection."""
    # Load manual road mask if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    RegionDetector.load_manual_road_mask(script_dir)
    
    debug_dir = os.path.join(output_dir, "debug_masks")
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(extensions)]
    
    print(f"Found {len(files)} images. Creating debug visualizations...")
    print("Color legend: Road=Blue, Vegetation=Green, Sky=Cyan, Vehicle=Red")
    
    for i, filename in enumerate(files, 1):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        debug_img = debug_visualize_masks(img)
        out_path = os.path.join(debug_dir, f"debug_{filename}")
        cv2.imwrite(out_path, debug_img)
        print(f"[{i}/{len(files)}] Debug: {filename}")
    
    print(f"Done! Check '{debug_dir}' folder.")


def process_batch(input_dir, output_dir):
    """Process all images in input directory."""
    # Load manual road mask if available
    script_dir = os.path.dirname(os.path.abspath(__file__))
    RegionDetector.load_manual_road_mask(script_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(extensions)]
    
    print(f"Found {len(files)} images. Processing with semantic rain simulation...")
    
    for i, filename in enumerate(files, 1):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {filename}")
            continue
        
        result = apply_rain(img)
        
        out_path = os.path.join(output_dir, f"rain_{filename}")
        cv2.imwrite(out_path, result)
        print(f"[{i}/{len(files)}] Processed: {filename}")
    
    print("Done!")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_PATH = os.path.join(SCRIPT_DIR, "input_files")
    OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output_files")
    
    if not os.path.exists(INPUT_PATH):
        os.makedirs(INPUT_PATH)
        print(f"Created: {INPUT_PATH}")
        print("Add your images to input_files/ and run again.")
    else:
        process_batch(INPUT_PATH, OUTPUT_PATH)
