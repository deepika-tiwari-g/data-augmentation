"""
Realistic Road Dust Simulation Module
Generates volumetric dust clouds with particle physics for data augmentation
"""

import cv2
import numpy as np
from noise import pnoise2
from typing import Tuple, List
import random


class DustParticle:
    """Represents a single dust particle with physics properties"""
    
    def __init__(self, x: float, y: float, age: int = 0):
        self.x = x
        self.y = y
        self.age = age
        self.max_age = random.randint(20, 40)  # Frames the particle lives (increased)
        self.vx = random.uniform(-4, 4)  # Horizontal velocity (wider spread)
        self.vy = random.uniform(-5, -2)  # Upward velocity (stronger upward movement)
        self.size = random.uniform(30, 80)  # Particle size (larger for visibility)
        self.opacity = random.uniform(0.6, 0.9)  # Initial opacity (stronger)
        
    def update(self, wind_vector: Tuple[float, float]):
        """Update particle position with Brownian motion and wind"""
        # Add wind drift
        self.vx += wind_vector[0] * 0.3
        self.vy += wind_vector[1] * 0.3
        
        # Brownian motion (random walk)
        self.vx += random.uniform(-0.5, 0.5)
        self.vy += random.uniform(-0.3, 0.1)
        
        # Apply velocity
        self.x += self.vx
        self.y += self.vy
        
        # Age the particle
        self.age += 1
        
    def is_alive(self) -> bool:
        """Check if particle should still be rendered"""
        return self.age < self.max_age
    
    def get_opacity(self) -> float:
        """Calculate current opacity based on age (fade out over time)"""
        life_ratio = self.age / self.max_age
        return self.opacity * (1 - life_ratio)


class PerlinNoiseGenerator:
    """Generates Perlin noise for volumetric cloud effects"""
    
    def __init__(self, scale: float = 100.0):
        self.scale = scale
        self.offset_x = random.uniform(0, 1000)
        self.offset_y = random.uniform(0, 1000)
        
    def get_noise(self, x: float, y: float) -> float:
        """Get Perlin noise value at given coordinates"""
        try:
            noise_val = pnoise2(
                (x + self.offset_x) / self.scale,
                (y + self.offset_y) / self.scale,
                octaves=3,
                persistence=0.5,
                lacunarity=2.0
            )
            # Normalize to 0-1 range
            return (noise_val + 1) / 2
        except:
            return 0.5


class DustCloudGenerator:
    """Generates realistic dust clouds with volumetric effects"""
    
    def __init__(self, image_shape: Tuple[int, int]):
        self.height, self.width = image_shape[:2]
        self.particles: List[DustParticle] = []
        self.noise_gen = PerlinNoiseGenerator(scale=80.0)
        self.wind_vector = (random.uniform(-1, 1), random.uniform(-0.5, 0.5))
        
    def add_particles_at_emitter(self, x: float, y: float, intensity: int):
        """Add new particles at tire emitter location"""
        for _ in range(intensity):
            # Add wider spread around the emitter point for better coverage
            px = x + random.uniform(-30, 30)
            py = y + random.uniform(-15, 15)
            self.particles.append(DustParticle(px, py))
    
    def update_particles(self):
        """Update all particles and remove dead ones"""
        for particle in self.particles:
            particle.update(self.wind_vector)
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive()]
    
    def render_to_layer(self, dust_color: Tuple[int, int, int]) -> np.ndarray:
        """Render all particles to a transparent layer"""
        # Create RGBA layer
        layer = np.zeros((self.height, self.width, 4), dtype=np.float32)
        
        for particle in self.particles:
            x, y = int(particle.x), int(particle.y)
            
            # Skip if out of bounds
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            
            # Get base opacity
            opacity = particle.get_opacity()
            
            # Create Gaussian blob for this particle
            size = int(particle.size)
            half_size = size // 2
            
            # Define region of interest
            x1 = max(0, x - half_size)
            x2 = min(self.width, x + half_size)
            y1 = max(0, y - half_size)
            y2 = min(self.height, y + half_size)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Create Gaussian kernel
            for dy in range(y1, y2):
                for dx in range(x1, x2):
                    # Distance from particle center
                    dist = np.sqrt((dx - x)**2 + (dy - y)**2)
                    
                    # Gaussian falloff
                    if dist < half_size:
                        # Add Perlin noise for volumetric effect
                        noise_val = self.noise_gen.get_noise(dx, dy)
                        
                        # Gaussian with noise modulation
                        gaussian = np.exp(-(dist**2) / (2 * (half_size/2)**2))
                        alpha = opacity * gaussian * (0.5 + 0.5 * noise_val)
                        
                        # Blend with existing alpha (additive) - increased strength
                        current_alpha = layer[dy, dx, 3]
                        new_alpha = min(1.0, current_alpha + alpha * 0.8)
                        
                        # Set color and alpha
                        layer[dy, dx, 0] = dust_color[0]
                        layer[dy, dx, 1] = dust_color[1]
                        layer[dy, dx, 2] = dust_color[2]
                        layer[dy, dx, 3] = new_alpha
        
        return layer


def extract_road_color(image: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """
    Extract average road color from the area around the vehicle
    
    Args:
        image: Input image (BGR format)
        vehicle_bbox: (x1, y1, x2, y2) bounding box of vehicle
    
    Returns:
        Average road color as (B, G, R) tuple
    """
    x1, y1, x2, y2 = vehicle_bbox
    h, w = image.shape[:2]
    
    # Sample road area below the vehicle
    road_y1 = min(h - 1, y2)
    road_y2 = min(h, y2 + 50)
    road_x1 = max(0, x1 - 20)
    road_x2 = min(w, x2 + 20)
    
    if road_y2 <= road_y1 or road_x2 <= road_x1:
        # Default brownish-grey if sampling fails
        return (100, 110, 120)
    
    road_region = image[road_y1:road_y2, road_x1:road_x2]
    
    # Calculate average color
    avg_color = np.mean(road_region, axis=(0, 1))
    
    # Make it much lighter and more visible for dust
    dust_color = avg_color * 1.5 + 40  # Lighter and brighter
    dust_color = np.clip(dust_color, 0, 255).astype(np.uint8)
    
    return tuple(dust_color.tolist())


def add_dust_wake(
    image: np.ndarray,
    vehicle_bbox: Tuple[int, int, int, int],
    vehicle_speed: float = 1.0,
    num_frames: int = 25
) -> np.ndarray:
    """
    Add realistic dust wake behind a moving vehicle with billowing cloud effect
    
    Args:
        image: Input image (BGR format)
        vehicle_bbox: (x1, y1, x2, y2) bounding box of vehicle
        vehicle_speed: Speed multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast)
        num_frames: Number of simulation frames to generate dust
    
    Returns:
        Image with dust wake added
    """
    x1, y1, x2, y2 = vehicle_bbox
    h, w = image.shape[:2]
    
    # Extract road color for dust
    dust_color = extract_road_color(image, vehicle_bbox)
    
    # Initialize dust cloud generator
    dust_gen = DustCloudGenerator((h, w))
    
    # Calculate vehicle dimensions
    vehicle_width = x2 - x1
    vehicle_height = y2 - y1
    vehicle_center_x = (x1 + x2) / 2
    
    # Create multiple emitter points for billowing cloud effect
    emitter_points = []
    
    # Rear emitters (bottom edge) - 5 points across rear
    for i in range(5):
        ratio = i / 4.0  # 0, 0.25, 0.5, 0.75, 1.0
        x = x1 + vehicle_width * ratio
        y = y2  # Bottom of vehicle
        emitter_points.append((x, y, 'rear'))
    
    # Left side emitters (3 points along left edge)
    for i in range(3):
        ratio = 0.4 + (i / 2.0) * 0.6  # Start from 40% down the side
        x = x1  # Left edge
        y = y1 + vehicle_height * ratio
        emitter_points.append((x, y, 'left_side'))
    
    # Right side emitters (3 points along right edge)
    for i in range(3):
        ratio = 0.4 + (i / 2.0) * 0.6  # Start from 40% down the side
        x = x2  # Right edge
        y = y1 + vehicle_height * ratio
        emitter_points.append((x, y, 'right_side'))
    
    # Calculate particle emission intensity based on speed
    # Much higher for billowing effect
    base_intensity = int(50 * vehicle_speed)  # 50-100 particles per frame
    
    # Simulate dust generation over multiple frames
    for frame in range(num_frames):
        # Gradually reduce intensity over time (front-loaded emission)
        if frame < num_frames * 0.4:
            intensity_multiplier = 1.0
        elif frame < num_frames * 0.7:
            intensity_multiplier = 0.7
        else:
            intensity_multiplier = 0.4
        
        # Emit particles from all emitter points
        for emitter_x, emitter_y, emitter_type in emitter_points:
            # Adjust intensity based on emitter type
            if emitter_type == 'rear':
                # Rear emitters produce most dust
                point_intensity = int(base_intensity * intensity_multiplier)
            else:
                # Side emitters produce less dust
                point_intensity = int(base_intensity * 0.5 * intensity_multiplier)
            
            # Add particles with position-specific velocity bias
            for _ in range(point_intensity):
                # Wide spread around emitter
                px = emitter_x + random.uniform(-150, 150)
                py = emitter_y + random.uniform(-30, 30)
                
                # Create particle
                particle = DustParticle(px, py)
                
                # Add lateral expansion based on position
                if emitter_type == 'left_side':
                    # Push dust to the left
                    particle.vx += random.uniform(-6, -2)
                elif emitter_type == 'right_side':
                    # Push dust to the right
                    particle.vx += random.uniform(2, 6)
                elif emitter_type == 'rear':
                    # Rear dust expands outward from center
                    if px < vehicle_center_x:
                        particle.vx += random.uniform(-4, -1)
                    else:
                        particle.vx += random.uniform(1, 4)
                
                # Add extra upward velocity for billowing effect
                particle.vy += random.uniform(-2, -1)
                
                dust_gen.particles.append(particle)
        
        # Update particle physics
        dust_gen.update_particles()
    
    # Render dust to transparent layer
    dust_layer = dust_gen.render_to_layer(dust_color)
    
    # Convert image to float for blending
    image_float = image.astype(np.float32) / 255.0
    
    # Extract RGB and alpha channels
    dust_rgb = dust_layer[:, :, :3] / 255.0
    dust_alpha = dust_layer[:, :, 3:4]
    
    # Apply very gentle density gradient (keep dust visible throughout)
    y_coords = np.linspace(1.0, 0.7, h).reshape(-1, 1, 1)
    dust_alpha = dust_alpha * y_coords
    
    # Alpha blending with atmospheric haze effect
    blended = image_float * (1 - dust_alpha) + dust_rgb * dust_alpha
    
    # Add stronger atmospheric haze in dust regions for billowing effect
    haze_color = np.array(dust_color, dtype=np.float32) / 255.0
    haze_alpha = dust_alpha * 0.3  # Stronger haze
    blended = blended * (1 - haze_alpha) + haze_color * haze_alpha
    
    # Convert back to uint8
    result = np.clip(blended * 255, 0, 255).astype(np.uint8)
    
    return result



def add_dust_wake_simple(
    image: np.ndarray,
    vehicle_bbox: Tuple[int, int, int, int],
    vehicle_speed: float = 1.0
) -> np.ndarray:
    """
    Simplified version that generates dust in a single pass
    
    Args:
        image: Input image (BGR format)
        vehicle_bbox: (x1, y1, x2, y2) bounding box of vehicle
        vehicle_speed: Speed multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast)
    
    Returns:
        Image with dust wake added
    """
    return add_dust_wake(image, vehicle_bbox, vehicle_speed, num_frames=25)
