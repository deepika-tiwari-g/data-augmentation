#!/usr/bin/env python3
"""
Road Dust Simulation for Data Augmentation
Generates realistic dust clouds behind moving vehicles for mining/industrial site images.

Features:
- Particle-based dust simulation
- Speed-dependent dust intensity
- Natural dust color matching
- Batch processing support
- No external noise library required (uses built-in numpy)

Author: Data Augmentation Pipeline
Date: 2026-01-09
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from typing import Tuple, List
import random

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")


class DustParticle:
    """Represents a single dust particle with physics properties."""
    
    def __init__(self, x: float, y: float, age: int = 0):
        self.x = x
        self.y = y
        self.age = age
        self.lifespan = random.randint(15, 30)
        self.vx = random.uniform(-2, 2)  # Horizontal drift
        self.vy = random.uniform(-3, -1)  # Upward movement
        self.size = random.uniform(3, 8)
        
    def update(self, wind_vector: Tuple[float, float]):
        """Update particle position with Brownian motion and wind."""
        # Add wind
        self.vx += wind_vector[0] * 0.1
        self.vy += wind_vector[1] * 0.1
        
        # Brownian motion (random walk)
        self.vx += random.uniform(-0.5, 0.5)
        self.vy += random.uniform(-0.3, 0.3)
        
        # Apply velocity
        self.x += self.vx
        self.y += self.vy
        
        # Age the particle
        self.age += 1
        
        # Damping (friction)
        self.vx *= 0.95
        self.vy *= 0.95
    
    def is_alive(self) -> bool:
        """Check if particle should still be rendered."""
        return self.age < self.lifespan
    
    def get_opacity(self) -> float:
        """Calculate current opacity based on age (fade out over time)."""
        return max(0, 1 - (self.age / self.lifespan))


class SimplifiedNoiseGenerator:
    """Generates smooth noise for volumetric cloud effects (simplified Perlin-like)."""
    
    def __init__(self, scale: float = 50.0):
        self.scale = scale
        self.offset_x = random.uniform(0, 1000)
        self.offset_y = random.uniform(0, 1000)
    
    def get_noise(self, x: float, y: float) -> float:
        """Get smooth noise value at given coordinates using sine waves."""
        # Use multiple octaves of sine waves for smooth organic noise
        nx = (x + self.offset_x) / self.scale
        ny = (y + self.offset_y) / self.scale
        
        # Combine multiple frequencies
        value = 0.0
        value += np.sin(nx * 2 * np.pi) * np.cos(ny * 2 * np.pi) * 0.5
        value += np.sin(nx * 4 * np.pi + 1.5) * np.cos(ny * 3 * np.pi + 0.7) * 0.3
        value += np.sin(nx * 8 * np.pi + 2.1) * np.cos(ny * 6 * np.pi + 1.3) * 0.2
        
        # Normalize to 0-1 range
        return (value + 1.0) / 2.0


class DustCloudGenerator:
    """Generates realistic dust clouds with volumetric effects."""
    
    def __init__(self, image_shape: Tuple[int, int]):
        self.height, self.width = image_shape[:2]
        self.particles: List[DustParticle] = []
        self.wind_vector = (random.uniform(-1, 1), random.uniform(-0.5, 0.5))
        self.noise_gen = SimplifiedNoiseGenerator()
    
    def add_particles_at_emitter(self, x: float, y: float, intensity: int):
        """Add new particles at tire emitter location."""
        for _ in range(intensity):
            px = x + random.uniform(-10, 10)
            py = y + random.uniform(-5, 5)
            self.particles.append(DustParticle(px, py))
    
    def update_particles(self):
        """Update all particles and remove dead ones."""
        for particle in self.particles:
            particle.update(self.wind_vector)
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive()]
    
    def render_to_layer(self, dust_color: Tuple[int, int, int]) -> np.ndarray:
        """Render all particles to a transparent layer."""
        # Create RGBA layer
        layer = np.zeros((self.height, self.width, 4), dtype=np.float32)
        
        for particle in self.particles:
            px, py = int(particle.x), int(particle.y)
            
            # Skip if out of bounds
            if px < 0 or px >= self.width or py < 0 or py >= self.height:
                continue
            
            # Get particle opacity
            opacity = particle.get_opacity()
            size = int(particle.size)
            
            # Add volumetric effect using noise
            noise_val = self.noise_gen.get_noise(particle.x, particle.y)
            opacity *= (0.5 + noise_val * 0.5)  # Modulate opacity with noise
            
            # Draw particle as a soft circle
            y1 = max(0, py - size)
            y2 = min(self.height, py + size)
            x1 = max(0, px - size)
            x2 = min(self.width, px + size)
            
            for y in range(y1, y2):
                for x in range(x1, x2):
                    # Calculate distance from particle center
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    
                    if dist <= size:
                        # Gaussian falloff
                        falloff = np.exp(-(dist**2) / (2 * (size/2)**2))
                        alpha = opacity * falloff * 0.6
                        
                        # Add to layer (additive blending)
                        layer[y, x, 0] += dust_color[0] * alpha  # B
                        layer[y, x, 1] += dust_color[1] * alpha  # G
                        layer[y, x, 2] += dust_color[2] * alpha  # R
                        layer[y, x, 3] += alpha * 255
        
        # Apply blur for smooth volumetric appearance
        layer[:, :, :3] = cv2.GaussianBlur(layer[:, :, :3], (15, 15), 0)
        layer[:, :, 3] = cv2.GaussianBlur(layer[:, :, 3], (21, 21), 0)
        
        # Clip values
        layer = np.clip(layer, 0, 255)
        
        return layer.astype(np.uint8)


def extract_road_color(image: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """
    Extract average road color from the area around the vehicle.
    
    Args:
        image: Input image (BGR format)
        vehicle_bbox: (x1, y1, x2, y2) bounding box of vehicle
    
    Returns:
        Average road color as (B, G, R) tuple
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = vehicle_bbox
    
    # Sample road area below vehicle
    road_y1 = min(h - 1, y2)
    road_y2 = min(h, y2 + 50)
    road_x1 = max(0, x1 - 20)
    road_x2 = min(w, x2 + 20)
    
    if road_y2 > road_y1 and road_x2 > road_x1:
        road_sample = image[road_y1:road_y2, road_x1:road_x2]
        avg_color = cv2.mean(road_sample)[:3]
        
        # Make it slightly lighter and more desaturated (dust color)
        dust_color = tuple(min(255, int(c * 1.3)) for c in avg_color)
        return dust_color
    else:
        # Default brownish-gray dust color
        return (160, 150, 140)


def add_dust_wake(
    image: np.ndarray,
    vehicle_bbox: Tuple[int, int, int, int],
    vehicle_speed: float = 1.0,
    num_frames: int = 25
) -> np.ndarray:
    """
    Add realistic dust wake behind a moving vehicle.
    
    Args:
        image: Input image (BGR format)
        vehicle_bbox: (x1, y1, x2, y2) bounding box of vehicle
        vehicle_speed: Speed multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast)
        num_frames: Number of simulation frames to generate dust
    
    Returns:
        Image with dust wake added
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = vehicle_bbox
    
    # Calculate tire emitter positions (rear wheels)
    vehicle_width = x2 - x1
    vehicle_height = y2 - y1
    
    # Rear wheels are at bottom, 20% inset from edges
    rear_y = y2 - int(vehicle_height * 0.05)
    left_tire_x = x1 + int(vehicle_width * 0.20)
    right_tire_x = x2 - int(vehicle_width * 0.20)
    
    # Extract road color for realistic dust
    dust_color = extract_road_color(image, vehicle_bbox)
    
    # Create dust generator
    dust_gen = DustCloudGenerator(image.shape)
    
    # Particle emission intensity based on speed
    base_intensity = int(3 * vehicle_speed)
    
    # Simulate dust generation over multiple frames
    for frame in range(num_frames):
        # Reduce emission over time (initial burst, then decay)
        intensity = max(1, int(base_intensity * (1 - frame / num_frames * 0.5)))
        
        # Emit particles from both tires
        dust_gen.add_particles_at_emitter(left_tire_x, rear_y, intensity)
        dust_gen.add_particles_at_emitter(right_tire_x, rear_y, intensity)
        
        # Update particle physics
        dust_gen.update_particles()
    
    # Render dust to layer
    dust_layer = dust_gen.render_to_layer(dust_color)
    
    # Composite dust onto image using alpha blending
    result = image.copy()
    alpha = dust_layer[:, :, 3:4].astype(np.float32) / 255.0
    dust_rgb = dust_layer[:, :, :3].astype(np.float32)
    
    # Only apply dust where alpha > 0
    mask = alpha > 0.01
    result_float = result.astype(np.float32)
    
    # Alpha blend: result = image * (1 - alpha) + dust * alpha
    result_float = result_float * (1 - alpha) + dust_rgb * alpha
    
    result = np.clip(result_float, 0, 255).astype(np.uint8)
    
    return result


def add_simple_dust_wake(
    image: np.ndarray,
    vehicle_bbox: Tuple[int, int, int, int],
    vehicle_speed: float = 1.0
) -> np.ndarray:
    """
    Simplified version using static dust clouds (faster, no particle simulation).
    
    Args:
        image: Input image (BGR format)
        vehicle_bbox: (x1, y1, x2, y2) bounding box of vehicle
        vehicle_speed: Speed multiplier
    
    Returns:
        Image with dust wake added
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = vehicle_bbox
    
    # Calculate dust region
    vehicle_width = x2 - x1
    vehicle_height = y2 - y1
    
    # Dust extends behind and around vehicle
    dust_width = int(vehicle_width * 1.5)
    dust_height = int(vehicle_height * 1.2 * vehicle_speed)
    
    # Center behind vehicle
    dust_x = x1 + vehicle_width // 2
    dust_y = y2
    
    # Extract road color
    dust_color = extract_road_color(image, vehicle_bbox)
    
    # Create dust mask
    dust_mask = np.zeros((h, w), dtype=np.float32)
    
    # Draw multiple overlapping ellipses for cloud effect
    for i in range(int(3 + vehicle_speed * 2)):
        offset_y = i * int(dust_height / 4)
        offset_x = random.randint(-int(dust_width / 4), int(dust_width / 4))
        
        center = (dust_x + offset_x, min(h - 1, dust_y + offset_y))
        axes = (dust_width // 2 + random.randint(-10, 10), 
                dust_height // 3 + random.randint(-5, 5))
        
        cv2.ellipse(dust_mask, center, axes, 0, 0, 360, 
                   0.3 * vehicle_speed / (i + 1), -1)
    
    # Blur for soft cloud effect
    dust_mask = cv2.GaussianBlur(dust_mask, (51, 51), 0)
    
    # Apply dust color with alpha blending
    result = image.copy().astype(np.float32)
    dust_mask_3ch = np.stack([dust_mask] * 3, axis=-1)
    dust_color_array = np.array(dust_color, dtype=np.float32)
    
    result = result * (1 - dust_mask_3ch) + dust_color_array * dust_mask_3ch
    
    return np.clip(result, 0, 255).astype(np.uint8)


def process_batch(
    input_dir: str = "input_files",
    output_dir: str = "output_files",
    vehicle_speed: float = 1.0,
    num_variants: int = 1,
    use_simple: bool = False,
    model_path: str = "site_2_yolov11n_v1+v2.pt"
):
    """
    Process all images in input directory with YOLO vehicle detection.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed images
        vehicle_speed: Base speed multiplier for dust generation
        num_variants: Number of dust variants to generate per image
        use_simple: Use simplified dust generation (faster)
        model_path: Path to YOLO model for vehicle detection
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO model for vehicle detection
    yolo = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_model_path = os.path.join(script_dir, model_path)
    
    if YOLO_AVAILABLE and os.path.exists(full_model_path):
        print(f"Loading YOLO model: {model_path}")
        yolo = YOLO(full_model_path)
        print("✓ Model loaded successfully\n")
    elif YOLO_AVAILABLE and not os.path.exists(full_model_path):
        print(f"Warning: YOLO model not found at {full_model_path}")
        print("Proceeding without vehicle detection (synthetic bounding box mode)\n")
    else:
        print("Warning: YOLO not available")
        print("Proceeding without vehicle detection (synthetic bounding box mode)\n")
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Base speed: {vehicle_speed}x")
    print(f"Variants per image: {num_variants}")
    print(f"Output directory: {output_dir}")
    print(f"Method: {'Simple' if use_simple else 'Particle-based'}")
    print(f"Detection: {'YOLO' if yolo else 'Synthetic'}\n")
    
    total_processed = 0
    
    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_file.name}")
        
        try:
            # Read image
            image = cv2.imread(str(img_file))
            
            if image is None:
                print(f"  ✗ Could not read image, skipping...")
                continue
            
            h, w = image.shape[:2]
            
            # Detect vehicles using YOLO or use synthetic bbox
            vehicle_bboxes = []
            
            if yolo:
                # Run YOLO detection
                results = yolo(image, verbose=False)
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            if conf > 0.25:  # Confidence threshold
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                vehicle_bboxes.append((x1, y1, x2, y2))
                
                if vehicle_bboxes:
                    print(f"  Detected {len(vehicle_bboxes)} vehicle(s)")
                else:
                    print(f"  No vehicles detected, skipping...")
                    continue
            else:
                # Fallback: Create synthetic vehicle bbox in center-bottom
                bbox_width = int(w * 0.25)
                bbox_height = int(h * 0.30)
                x1 = (w - bbox_width) // 2
                y1 = h - bbox_height - int(h * 0.05)
                x2 = x1 + bbox_width
                y2 = y1 + bbox_height
                vehicle_bboxes = [(x1, y1, x2, y2)]
                print(f"  Using synthetic vehicle bbox")
            
            # Generate variants with different speed multipliers
            for variant_idx in range(num_variants):
                if num_variants == 1:
                    speed = vehicle_speed
                    suffix = ""
                else:
                    # Generate variants: light, medium, heavy
                    speeds = {
                        0: (0.6, "_light"),
                        1: (1.0, "_medium"),
                        2: (1.5, "_heavy")
                    }
                    speed, suffix = speeds.get(variant_idx, (1.0, f"_v{variant_idx}"))
                
                # Start with original image
                dusty_image = image.copy()
                
                # Apply dust to each detected vehicle
                for bbox in vehicle_bboxes:
                    if use_simple:
                        dusty_image = add_simple_dust_wake(dusty_image, bbox, speed)
                    else:
                        dusty_image = add_dust_wake(dusty_image, bbox, speed, num_frames=25)
                
                # Save result
                output_name = f"{img_file.stem}_dust{suffix}{img_file.suffix}"
                output_file = output_path / output_name
                
                cv2.imwrite(str(output_file), dusty_image)
                total_processed += 1
                
                print(f"  ✓ Generated: {output_name} (speed: {speed}x)")
        
        except Exception as e:
            print(f"  ✗ Error processing {img_file.name}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total images generated: {total_processed}")
    print(f"Output location: {output_dir}")
    print(f"{'='*60}")


def main():
    """Main function to run the road dust simulation."""
    parser = argparse.ArgumentParser(
        description='Road Dust Simulation for Mining/Industrial Site Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default settings)
  python road_dust_augmentation.py
  
  # Custom speed for fast-moving vehicles
  python road_dust_augmentation.py --speed 1.5
  
  # Generate 3 variants (light, medium, heavy)
  python road_dust_augmentation.py --variants 3
  
  # Use simplified fast method
  python road_dust_augmentation.py --simple
  
  # Full options
  python road_dust_augmentation.py --input-dir input_files --output-dir output_files --speed 1.2 --variants 1
        """
    )
    
    parser.add_argument(
        '--input-dir',
        default='input_files',
        help='Input directory containing images (default: input_files)'
    )
    parser.add_argument(
        '--output-dir',
        default='output_files',
        help='Output directory for processed images (default: output_files)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Vehicle speed multiplier (0.5=slow, 1.0=normal, 2.0=fast)'
    )
    parser.add_argument(
        '--variants',
        type=int,
        default=1,
        help='Number of dust variants to generate per image (default: 1)'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simplified dust generation (faster, less realistic)'
    )
    
    args =parser.parse_args()
    
    # Process images
    process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vehicle_speed=args.speed,
        num_variants=args.variants,
        use_simple=args.simple,
        model_path="site_2_yolov11n_v1+v2.pt"
    )


if __name__ == '__main__':
    main()
