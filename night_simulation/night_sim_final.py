#!/usr/bin/env python3
"""
Final Night Simulation - Simplified and Reliable
Creates realistic night scenes with proper headlight placement
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Dict
import argparse
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class NightSimFinal:
    """Simple, reliable night simulation with custom YOLO model."""
    
    # Your custom model's vehicle classes (adjust if different)
    VEHICLE_CLASSES = {0: 'vehicle', 1: 'truck', 2: 'dumper', 3: 'tipper'}  # Adjust based on your model
    
    def __init__(self, config: dict, model_path: str = 'site_2_yolov11n_v1+v2.pt'):
        self.config = config
        self.night_darkness = config.get('night_darkness', 0.40)  # Twilight
        self.headlight_brightness = config.get('headlight_brightness', 0.65)  # REDUCED
        self.streetlight_brightness = config.get('streetlight_brightness', 1.15)  # INCREASED
        self.streetlight_spread = config.get('streetlight_spread', 100)
        
        # Load YOUR custom YOLO model
        self.yolo = None
        if YOLO_AVAILABLE:
            try:
                # Try to load your custom model
                if os.path.exists(model_path):
                    self.yolo = YOLO(model_path)
                    print(f"✓ Loaded custom YOLO model: {model_path}")
                else:
                    # Fallback to default
                    print(f"Custom model not found at {model_path}")
                    print("Trying default YOLOv8...")
                    self.yolo = YOLO('yolov8n.pt')
                    # Use standard COCO classes for fallback
                    self.VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
            except Exception as e:
                print(f"Error loading YOLO: {e}")
                print("Using fallback detection")
        else:
            print("ultralytics not installed - using fallback detection")
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """Detect vehicles using custom YOLO model."""
        vehicles = []
        h, w = image.shape[:2]
        
        if self.yolo:
            results = self.yolo(image, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Accept all vehicle classes with confidence > 0.25
                    # Your custom model should detect vehicles accurately
                    if conf > 0.25:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vehicles.append({
                            'bbox': (x1, y1, x2, y2),
                            'center_x': (x1 + x2) // 2,
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'class': cls_id,
                            'confidence': conf
                        })
                        print(f"  Detected: class={cls_id}, conf={conf:.2f}, bbox=({x1},{y1},{x2},{y2})")
        else:
            # Fallback: simple edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if w * h * 0.01 < area < w * h * 0.4:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    if 0.5 < bw / bh < 4.0:
                        vehicles.append({
                            'bbox': (x, y, x + bw, y + bh),
                            'center_x': x + bw // 2,
                            'width': bw,
                            'height': bh
                        })
        
        return vehicles
    
    def get_headlight_positions(self, vehicle: Dict, image_width: int) -> List[Tuple[int, int]]:
        """
        Get headlight positions based on vehicle location in frame.
        Vehicles on left face right, vehicles on right face left, center faces camera.
        """
        x1, y1, x2, y2 = vehicle['bbox']
        cx = vehicle['center_x']
        vw = vehicle['width']
        vh = vehicle['height']
        
        # Headlights are in bottom 20% of vehicle
        hy = y2 - int(vh * 0.15)
        
        # Determine which way vehicle is facing based on position
        if cx < image_width * 0.35:
            # Left side - facing right, headlights on RIGHT edge
            return [(x2 - int(vw * 0.15), hy)]
        elif cx > image_width * 0.65:
            # Right side - facing left, headlights on LEFT edge  
            return [(x1 + int(vw * 0.15), hy)]
        else:
            # Center - facing camera, both headlights visible
            left_x = x1 + int(vw * 0.25)
            right_x = x1 + int(vw * 0.75)
            return [(left_x, hy), (right_x, hy)]
    
    def darken_image(self, image: np.ndarray) -> np.ndarray:
        """Apply twilight darkening (less dark than night)."""
        img_float = image.astype(np.float32) / 255.0
        
        # Moderate darkening for twilight
        darkened = np.power(img_float, 1.8) * self.night_darkness  # Less gamma
        
        # Slight blue twilight tint
        darkened[:, :, 0] *= 1.1  # Slight blue
        darkened[:, :, 2] *= 0.9  # Slight reduce red
        
        # Gentle sky darkening
        h = image.shape[0]
        gradient = np.linspace(0.7, 1.0, h)[:, np.newaxis, np.newaxis]  # Less aggressive
        darkened = darkened * gradient
        
        return (np.clip(darkened, 0, 1) * 255).astype(np.uint8)
    
    def add_headlight(self, image: np.ndarray, pos: Tuple[int, int], 
                     direction: str = 'forward') -> None:
        """Add bright headlight with beam."""
        x, y = pos
        h, w = image.shape[:2]
        overlay = image.astype(np.float32)
        
        # Bright white core
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        
        # Small bright core
        core = np.exp(-(dist**2) / (2 * 10**2))
        core = np.clip(core * self.headlight_brightness * 255, 0, 255)
        
        # Glow
        glow = np.exp(-(dist**2) / (2 * 30**2))
        glow = np.clip(glow * self.headlight_brightness * 180, 0, 255)
        
        light = np.maximum(core, glow)
        
        # White LED color
        for i in range(3):
            overlay[:, :, i] += light
        
        # Add beam
        if direction == 'forward':
            dx, dy = 0, 1
        elif direction == 'right':
            dx, dy = 0.6, 0.8
        elif direction == 'left':
            dx, dy = -0.6, 0.8
        else:
            dx, dy = 0, 1
        
        # Beam projection
        beam_len = int(h * 0.45)
        beam_width = int(w * 0.08)
        
        px = X - x
        py = Y - y
        proj = px * dx + py * dy
        proj = np.clip(proj, 0, beam_len)
        perp = np.abs(px * (-dy) + py * dx)
        
        cone_w = 5 + (proj / beam_len) * beam_width
        
        beam = np.where(
            (proj > 0) & (proj < beam_len) & (perp < cone_w),
            1.0, 0.0
        )
        
        falloff = (1 - proj / beam_len) * (1 - perp / (cone_w + 1))
        beam = beam * falloff * self.headlight_brightness * 200
        
        for i in range(3):
            overlay[:, :, i] += beam
        
        np.copyto(image, np.clip(overlay, 0, 255).astype(np.uint8))
    
    def add_streetlights_right_side(self, image: np.ndarray) -> None:
        """Add bright golden street lights on RIGHT SIDE only."""
        h, w = image.shape[:2]
        overlay = image.astype(np.float32)
        
        # RIGHT side only: 2 lights at 75%, 90% of width
        right_positions = [
            int(w * 0.75),
            int(w * 0.90)
        ]
        
        ly = int(h * 0.1)  # Light height
        
        # Golden-orange color
        color = np.array([40, 150, 240])  # BGR
        
        for lx in right_positions:
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - lx)**2 + (Y - ly)**2)
            
            # LARGE spread with HIGHER intensity
            core = np.exp(-(dist**2) / (2 * 35**2))  # Larger core
            core = np.clip(core * self.streetlight_brightness * 1.2, 0, 1)  # Brighter
            
            glow = np.exp(-(dist**2) / (2 * self.streetlight_spread**2))
            glow = np.clip(glow * self.streetlight_brightness * 0.7, 0, 1)  # Brighter glow
            
            light = np.maximum(core, glow)
            
            for j in range(3):
                overlay[:, :, j] += light * color[j]
            
            # Wide downward cone with higher intensity
            cone_len = int(h * 0.65)
            cone_w = int(w * 0.18)
            
            proj = Y - ly
            proj = np.clip(proj, 0, cone_len)
            perp = np.abs(X - lx)
            
            cone_width = 25 + (proj / cone_len) * cone_w
            
            cone = np.where(
                (proj > 0) & (proj < cone_len) & (perp < cone_width),
                1.0, 0.0
            )
            
            falloff = (1 - proj / cone_len) * (1 - perp / (cone_width + 1))
            cone = cone * falloff * self.streetlight_brightness * 0.5  # Increased from 0.4
            
            for j in range(3):
                overlay[:, :, j] += cone * color[j]
        
        np.copyto(image, np.clip(overlay, 0, 255).astype(np.uint8))
    
    def simulate(self, image: np.ndarray) -> np.ndarray:
        """Complete night simulation."""
        original = image.copy()
        
        # 1. Detect vehicles
        vehicles = self.detect_vehicles(original)
        
        # 2. Darken image (twilight)
        result = self.darken_image(image)
        
        # 3. Add street lights on right side only (brighter)
        self.add_streetlights_right_side(result)
        
        # 4. Add headlights
        for vehicle in vehicles:
            positions = self.get_headlight_positions(vehicle, image.shape[1])
            
            # Determine beam direction
            cx = vehicle['center_x']
            w = image.shape[1]
            
            if cx < w * 0.35:
                direction = 'right'
            elif cx > w * 0.65:
                direction = 'left'
            else:
                direction = 'forward'
            
            for pos in positions:
                self.add_headlight(result, pos, direction)
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Night simulation with custom YOLO model')
    parser.add_argument('-i', '--input', default='input_files', help='Input directory')
    parser.add_argument('-o', '--output', default='output_files', help='Output directory')
    parser.add_argument('-c', '--config', help='Config JSON file')
    parser.add_argument('-m', '--model', default='site_2_yolov11n_v1+v2.pt', 
                       help='Path to custom YOLO model (default: site_2_yolov11n_v1+v2.pt)')
    args = parser.parse_args()
    
    # Load config
    config = {
        'night_darkness': 0.40,  # Twilight
        'headlight_brightness': 0.35,  # Reduced so vehicles visible
        'streetlight_brightness': 1.15,  # Increased brightness
        'streetlight_spread': 100
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config.update(json.load(f))
    
    # Create output dir
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize simulator with custom model
    sim = NightSimFinal(config, model_path=args.model)
    
    # Process images
    image_files = list(Path(args.input).glob('*.jpg')) + \
                  list(Path(args.input).glob('*.png')) + \
                  list(Path(args.input).glob('*.jpeg'))
    
    print(f"\nProcessing {len(image_files)} images...\n")
    
    for img_file in image_files:
        print(f"  Processing: {img_file.name}")
        img = cv2.imread(str(img_file))
        if img is not None:
            result = sim.simulate(img)
            output_path = Path(args.output) / f"night_{img_file.name}"
            cv2.imwrite(str(output_path), result)
            print(f"    ✓ Saved: {output_path.name}")
    
    print(f"\n✓ Done! Processed {len(image_files)} images")


if __name__ == '__main__':
    main()
