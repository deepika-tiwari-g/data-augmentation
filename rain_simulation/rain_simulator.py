"""
Realistic Heavy Rainfall Simulator for Mining Site Data Augmentation

This script simulates realistic heavy rainfall conditions at mining sites with:
- Volumetric rain streaks with varying opacity and motion blur
- Wet asphalt and dark soil with high specularity and reflections
- Scattered water puddles with subtle ripple patterns
- Atmospheric hazy mist and fog for depth
- Cool color temperature and desaturated tones
- Water droplets on vehicle surfaces
- High-definition cinematic weather effects
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, List, Dict
import random


class RegionDetector:
    """Detects semantic regions in mining site images for region-specific rain effects"""
    
    def detect_regions(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect semantic regions in the image
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Dictionary with float masks [0.0-1.0] for: 'road', 'sky', 'vehicles', 'vegetation'
        """
        h, w = image.shape[:2]
        
        # Detect each region type
        road_mask = self._detect_road(image)
        sky_mask = self._detect_sky(image)
        vehicles_mask = self._detect_vehicles(image)
        vegetation_mask = self._detect_vegetation(image)
        
        # Ensure masks don't overlap (priority: vehicles > vegetation > sky > road)
        # Vehicles have highest priority
        sky_mask = sky_mask * (1 - vehicles_mask)
        vegetation_mask = vegetation_mask * (1 - vehicles_mask) * (1 - sky_mask)
        road_mask = road_mask * (1 - vehicles_mask) * (1 - sky_mask) * (1 - vegetation_mask)
        
        return {
            'road': road_mask,
            'sky': sky_mask,
            'vehicles': vehicles_mask,
            'vegetation': vegetation_mask
        }
    
    def _detect_road(self, image: np.ndarray) -> np.ndarray:
        """
        Detect road/ground areas using color and position
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Road mask [0.0-1.0]
        """
        h, w = image.shape[:2]
        
        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Road characteristics: low saturation, low-mid value (grey/black asphalt)
        # HSV ranges for road: H=any, S=0-80, V=20-150
        lower_road = np.array([0, 0, 20])
        upper_road = np.array([180, 80, 150])
        
        color_mask = cv2.inRange(hsv, lower_road, upper_road).astype(np.float32) / 255.0
        
        # Position-based: roads typically in lower 70% of image
        position_mask = np.zeros((h, w), dtype=np.float32)
        road_start = int(h * 0.3)  # Start from 30% down
        position_mask[road_start:, :] = 1.0
        
        # Combine color and position
        road_mask = color_mask * position_mask
        
        # Smooth the mask
        road_mask = cv2.GaussianBlur(road_mask, (21, 21), 0)
        
        # Threshold to create cleaner mask
        road_mask = np.clip(road_mask * 1.5, 0.0, 1.0)
        
        return road_mask
    
    def _detect_sky(self, image: np.ndarray) -> np.ndarray:
        """
        Detect sky areas using brightness and position
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Sky mask [0.0-1.0]
        """
        h, w = image.shape[:2]
        
        # Convert to grayscale for brightness detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Sky is typically bright
        brightness_mask = np.clip((gray - 0.4) * 2.5, 0.0, 1.0)
        
        # Position-based: sky typically in upper 50% of image
        position_mask = np.zeros((h, w), dtype=np.float32)
        sky_end = int(h * 0.5)
        
        # Gradient from top (1.0) to middle (0.0)
        gradient = np.linspace(1.0, 0.0, sky_end)
        position_mask[:sky_end, :] = np.tile(gradient.reshape(-1, 1), (1, w))
        
        # Combine brightness and position
        sky_mask = brightness_mask * position_mask
        
        # Smooth the mask
        sky_mask = cv2.GaussianBlur(sky_mask, (31, 31), 0)
        
        return sky_mask
    
    def _detect_vehicles(self, image: np.ndarray) -> np.ndarray:
        """
        Detect vehicle areas using edges and color
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Vehicle mask [0.0-1.0]
        """
        h, w = image.shape[:2]
        
        # Edge detection for vehicle contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        
        # Dilate edges to create vehicle regions
        kernel = np.ones((15, 15), np.uint8)
        vehicle_regions = cv2.dilate(edges, kernel, iterations=2)
        
        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Vehicles are often distinct colors (white, yellow, red trucks)
        # Exclude very low saturation (road) and very high saturation (vegetation)
        # Focus on mid-saturation with various hues
        color_mask = np.zeros((h, w), dtype=np.float32)
        
        # White vehicles (low saturation, high value)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
        
        # Colored vehicles (moderate saturation, moderate-high value)
        colored_mask = cv2.inRange(hsv, np.array([0, 50, 100]), np.array([180, 200, 255]))
        
        color_mask = ((white_mask + colored_mask) > 0).astype(np.float32)
        
        # Combine edge-based and color-based detection
        vehicle_mask = vehicle_regions * color_mask
        
        # Smooth and enhance
        vehicle_mask = cv2.GaussianBlur(vehicle_mask, (11, 11), 0)
        vehicle_mask = np.clip(vehicle_mask * 2.0, 0.0, 1.0)
        
        # Position constraint: vehicles typically in middle 60% of image (not top sky, not bottom edge)
        position_mask = np.zeros((h, w), dtype=np.float32)
        vehicle_start = int(h * 0.2)
        vehicle_end = int(h * 0.8)
        position_mask[vehicle_start:vehicle_end, :] = 1.0
        
        vehicle_mask = vehicle_mask * position_mask
        
        return vehicle_mask
    
    def _detect_vegetation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect vegetation areas using green color
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Vegetation mask [0.0-1.0]
        """
        h, w = image.shape[:2]
        
        # Convert to HSV for green detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Green hue range: 35-85 degrees, moderate-high saturation
        lower_green = np.array([35, 40, 30])
        upper_green = np.array([85, 255, 255])
        
        vegetation_mask = cv2.inRange(hsv, lower_green, upper_green).astype(np.float32) / 255.0
        
        # Vegetation typically on sides and edges, not in center (where road/vehicles are)
        # Create a mask that favors edges
        center_x = w // 2
        x_coords = np.arange(w)
        edge_preference = np.abs(x_coords - center_x) / (w / 2)  # 0 at center, 1 at edges
        edge_mask = np.tile(edge_preference, (h, 1))
        
        # Apply edge preference
        vegetation_mask = vegetation_mask * (0.3 + 0.7 * edge_mask)
        
        # Smooth the mask
        vegetation_mask = cv2.GaussianBlur(vegetation_mask, (21, 21), 0)
        
        return vegetation_mask


class RainSimulator:
    """Simulates realistic heavy rainfall effects on images"""
    
    def __init__(self, 
                 rain_intensity: float = 0.8,
                 puddle_density: float = 0.6,
                 wetness_level: float = 0.7,
                 atmospheric_depth: float = 0.5,
                 color_temperature: float = 0.6,
                 semantic_aware: bool = True):
        """
        Initialize the rain simulator
        
        Args:
            rain_intensity: Intensity of rain streaks (0.0-1.0)
            puddle_density: Density of water puddles (0.0-1.0)
            wetness_level: Level of surface wetness/specularity (0.0-1.0)
            atmospheric_depth: Strength of atmospheric fog/mist (0.0-1.0)
            color_temperature: Cool color temperature (0.0=neutral, 1.0=very cool)
            semantic_aware: Enable semantic-aware region-specific effects (default: True)
        """
        self.rain_intensity = rain_intensity
        self.puddle_density = puddle_density
        self.wetness_level = wetness_level
        self.atmospheric_depth = atmospheric_depth
        self.color_temperature = color_temperature
        self.semantic_aware = semantic_aware
        
        # Initialize region detector for semantic-aware mode
        if self.semantic_aware:
            self.region_detector = RegionDetector()
        
    def generate_rain_streaks(self, image: np.ndarray) -> np.ndarray:
        """
        Generate realistic rain streaks with physics-based motion blur
        
        Args:
            image: Input image
            
        Returns:
            Rain streak mask (0.0-1.0 range for alpha blending)
        """
        h, w = image.shape[:2]
        rain_layer = np.zeros((h, w), dtype=np.float32)
        
        # Calculate number of rain streaks based on intensity
        num_streaks = int(self.rain_intensity * 3000 * (h * w) / (1920 * 1080))
        
        for _ in range(num_streaks):
            # Random position
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            
            # Physics-based rain streak parameters
            # Elongated streaks (30-60 pixels for realistic falling velocity)
            length = random.randint(30, 60)
            thickness = random.uniform(0.8, 1.5)
            # Angle variation 70-80 degrees from horizontal (nearly vertical)
            angle = random.uniform(75, 85)
            # Low opacity for semi-transparent effect
            opacity = random.uniform(0.08, 0.15)
            
            # Calculate end point (falling downward)
            end_x = int(x + length * np.cos(np.radians(angle)))
            end_y = int(y + length * np.sin(np.radians(angle)))
            
            # Ensure points are within bounds
            end_x = max(0, min(w - 1, end_x))
            end_y = max(0, min(h - 1, end_y))
            
            # Ensure thickness is a positive integer
            line_thickness = max(1, int(thickness))
            
            # Draw rain streak with opacity value
            cv2.line(rain_layer, (x, y), (end_x, end_y), opacity, 
                    line_thickness, cv2.LINE_AA)
        
        # Apply directional motion blur (70-80 degree angle)
        kernel_size = 21  # Larger kernel for more pronounced blur
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        # Create vertical line kernel
        kernel[:, kernel_size // 2] = 1.0 / kernel_size
        
        # Rotate kernel to match rain angle (75-80 degrees)
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, -10, 1.0)  # -10 for downward angle
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        # Apply motion blur
        rain_layer = cv2.filter2D(rain_layer, -1, kernel)
        
        # Clip to valid range
        rain_layer = np.clip(rain_layer, 0.0, 1.0)
        
        return rain_layer
    
    def create_wet_surface_effect(self, image: np.ndarray) -> np.ndarray:
        """
        Create wet surface effect with enhanced ground darkening and specularity
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Image with wet surface effect (uint8)
        """
        # Convert to float [0.0, 1.0]
        img_float = image.astype(np.float32) / 255.0
        
        # ENHANCED: Darken ground significantly (wet asphalt/soil looks deep grey/black)
        # Reduce albedo more aggressively for realistic wet look
        img_float = img_float * (1.0 - self.wetness_level * 0.35)
        
        # ENHANCED: Increase contrast more for wet look
        contrast = 1.35  # Increased from 1.2
        img_float = (img_float - 0.5) * contrast + 0.5
        img_float = np.clip(img_float, 0.0, 1.0)
        
        # ENHANCED: Stronger specular highlights for wet surfaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Detect bright areas and edges (specular highlights on wet surfaces)
        _, bright_mask = cv2.threshold((gray * 255).astype(np.uint8), 120, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.GaussianBlur(bright_mask, (11, 11), 0)
        bright_mask = bright_mask.astype(np.float32) / 255.0
        
        # ENHANCED: Stronger specular highlights (wet surfaces reflect sky/light)
        specular = np.stack([bright_mask] * 3, axis=-1) * self.wetness_level * 0.25
        img_float = np.clip(img_float + specular, 0.0, 1.0)
        
        # ENHANCED: Stronger reflections on wet ground
        h, w = image.shape[:2]
        reflection_strength = self.wetness_level * 0.18  # Increased from 0.12
        
        # Create reflection of upper portion (sky, vehicles)
        reflection_height = h // 2  # Larger reflection area
        reflection = np.flipud(img_float[:reflection_height, :])
        
        # Darken reflections (wet surfaces show darker reflections)
        reflection = reflection * 0.7
        
        # Create fade mask
        fade = np.linspace(reflection_strength, 0, reflection_height)
        fade = np.tile(fade.reshape(-1, 1, 1), (1, w, 3))
        
        # Blend reflection into lower portion (ground area)
        blend_start = h - reflection_height
        img_float[blend_start:, :] = np.clip(
            img_float[blend_start:, :] * (1 - fade) + reflection * fade,
            0.0, 1.0
        )
        
        # Convert back to uint8
        return (img_float * 255).astype(np.uint8)
    
    def generate_puddles(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate scattered water puddles with subtle ripple patterns
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (puddle mask, puddle effect layer)
        """
        h, w = image.shape[:2]
        puddle_mask = np.zeros((h, w), dtype=np.float32)
        
        # Generate random puddles in lower portion of image
        num_puddles = int(self.puddle_density * 15)
        
        for _ in range(num_puddles):
            # Puddles more likely in lower half
            center_x = random.randint(0, w - 1)
            center_y = random.randint(h // 2, h - 1)
            
            # Random puddle size
            radius_x = random.randint(30, 100)
            radius_y = random.randint(20, 60)
            
            # Create elliptical puddle
            y_coords, x_coords = np.ogrid[:h, :w]
            ellipse = (((x_coords - center_x) / radius_x) ** 2 + 
                      ((y_coords - center_y) / radius_y) ** 2 <= 1)
            
            # Add to puddle mask with smooth edges
            puddle_mask[ellipse] = np.maximum(
                puddle_mask[ellipse], 
                random.uniform(0.5, 1.0)
            )
        
        # Smooth puddle edges
        puddle_mask = cv2.GaussianBlur(puddle_mask, (31, 31), 0)
        
        # Create ripple effect
        ripple_layer = self._create_ripples(image, puddle_mask)
        
        return puddle_mask, ripple_layer
    
    def _create_ripples(self, image: np.ndarray, puddle_mask: np.ndarray) -> np.ndarray:
        """
        Create subtle ripple patterns in puddles
        
        Args:
            image: Input image
            puddle_mask: Mask indicating puddle locations
            
        Returns:
            Ripple effect layer
        """
        h, w = image.shape[:2]
        
        # Create ripple displacement map
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # Multiple ripple sources
        num_ripples = 8
        displacement_x = np.zeros((h, w), dtype=np.float32)
        displacement_y = np.zeros((h, w), dtype=np.float32)
        
        for _ in range(num_ripples):
            # Random ripple center
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            
            # Distance from center
            dist = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)
            
            # Ripple parameters
            frequency = random.uniform(0.05, 0.1)
            amplitude = random.uniform(1.5, 3.0)
            phase = random.uniform(0, 2 * np.pi)
            
            # Create ripple wave
            ripple = amplitude * np.sin(dist * frequency + phase) * np.exp(-dist / 100)
            
            # Add to displacement
            displacement_x += ripple * (x_coords - cx) / (dist + 1e-6)
            displacement_y += ripple * (y_coords - cy) / (dist + 1e-6)
        
        # Apply displacement only in puddle areas
        displacement_x *= puddle_mask
        displacement_y *= puddle_mask
        
        # Create distorted image
        map_x = (x_coords + displacement_x).astype(np.float32)
        map_y = (y_coords + displacement_y).astype(np.float32)
        
        ripple_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        return ripple_image
    
    def add_atmospheric_effects(self, image: np.ndarray) -> np.ndarray:
        """
        Add atmospheric hazy mist and fog for depth
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Image with atmospheric effects (uint8)
        """
        h, w = image.shape[:2]
        
        # Convert to float [0.0, 1.0]
        img_float = image.astype(np.float32) / 255.0
        
        # Create depth-based fog (stronger at top/distance)
        depth_gradient = np.linspace(self.atmospheric_depth, 0, h)
        depth_gradient = np.tile(depth_gradient.reshape(-1, 1, 1), (1, w, 3))
        
        # Fog color (cool gray) - normalized to [0.0, 1.0]
        fog_color = np.array([0.71, 0.73, 0.71], dtype=np.float32)  # ~180/255, 185/255, 180/255
        
        # Blend fog with proper normalization
        img_float = (
            img_float * (1 - depth_gradient) + 
            fog_color * depth_gradient
        )
        img_float = np.clip(img_float, 0.0, 1.0)
        
        # Add volumetric mist (Perlin-like noise)
        mist = self._generate_mist(h, w)
        mist_strength = self.atmospheric_depth * 0.2  # Reduced from 0.3
        
        # Blend mist with proper normalization
        img_float = (
            img_float * (1 - mist * mist_strength) + 
            fog_color * mist * mist_strength
        )
        img_float = np.clip(img_float, 0.0, 1.0)
        
        # Convert back to uint8
        return (img_float * 255).astype(np.uint8)
    
    def _generate_mist(self, h: int, w: int) -> np.ndarray:
        """
        Generate volumetric mist using multi-scale noise
        
        Args:
            h: Image height
            w: Image width
            
        Returns:
            Mist layer
        """
        # Start with low resolution noise
        mist = np.random.rand(h // 8, w // 8).astype(np.float32)
        
        # Upscale with blur for smooth mist
        mist = cv2.resize(mist, (w, h), interpolation=cv2.INTER_LINEAR)
        mist = cv2.GaussianBlur(mist, (51, 51), 0)
        
        # Add finer detail
        detail = np.random.rand(h // 4, w // 4).astype(np.float32)
        detail = cv2.resize(detail, (w, h), interpolation=cv2.INTER_LINEAR)
        detail = cv2.GaussianBlur(detail, (25, 25), 0)
        
        mist = 0.7 * mist + 0.3 * detail
        
        # Normalize
        mist = (mist - mist.min()) / (mist.max() - mist.min() + 1e-6)
        
        return np.stack([mist] * 3, axis=-1)
    
    def add_water_droplets(self, image: np.ndarray) -> np.ndarray:
        """
        Add water droplets on vehicle surfaces and windshield
        
        Args:
            image: Input image
            
        Returns:
            Image with water droplets
        """
        h, w = image.shape[:2]
        droplet_layer = image.copy()
        
        # Generate random droplets
        num_droplets = random.randint(50, 150)
        
        for _ in range(num_droplets):
            # Random position
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            
            # Random droplet size
            radius = random.randint(2, 8)
            
            # Create droplet effect (lens distortion)
            if radius > 3:
                # Extract region
                y1, y2 = max(0, y - radius), min(h, y + radius)
                x1, x2 = max(0, x - radius), min(w, x + radius)
                
                if y2 - y1 > 0 and x2 - x1 > 0:
                    region = image[y1:y2, x1:x2].copy()
                    
                    # Create circular mask
                    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                    center = (radius, radius)
                    cv2.circle(mask, center, radius, 255, -1)
                    
                    # Slight magnification effect
                    region_h, region_w = region.shape[:2]
                    if region_h > 0 and region_w > 0:
                        magnified = cv2.resize(region, None, fx=1.1, fy=1.1, 
                                             interpolation=cv2.INTER_LINEAR)
                        
                        # Crop to original size
                        start_y = (magnified.shape[0] - region_h) // 2
                        start_x = (magnified.shape[1] - region_w) // 2
                        magnified = magnified[start_y:start_y + region_h, 
                                            start_x:start_x + region_w]
                        
                        # Blend with mask
                        mask_3ch = np.stack([mask] * 3, axis=-1) / 255.0
                        droplet_layer[y1:y2, x1:x2] = (
                            magnified * mask_3ch + 
                            droplet_layer[y1:y2, x1:x2] * (1 - mask_3ch)
                        ).astype(np.uint8)
            
            # Add highlight
            brightness = random.randint(200, 255)
            cv2.circle(droplet_layer, (x, y), max(1, radius // 3), 
                      (brightness, brightness, brightness), -1, cv2.LINE_AA)
        
        return droplet_layer
    
    def apply_color_grading(self, image: np.ndarray) -> np.ndarray:
        """
        Apply cool color grading for rainy day atmosphere (blue/cyan shift)
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Color graded image (uint8)
        """
        # Convert to float [0.0, 1.0]
        img_float = image.astype(np.float32) / 255.0
        
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
        
        # ENHANCED: More desaturation for rainy day look
        hsv[:, :, 1] *= (1.0 - self.color_temperature * 0.35)
        
        # Convert back to BGR
        img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_float = np.clip(img_float, 0.0, 1.0)
        
        # ENHANCED: Stronger cool color temperature (blue/cyan shift)
        # Shift white balance toward blue and cyan for 'cool weather' atmosphere
        color_shift = self.color_temperature * 0.12  # Increased from 0.08
        img_float[:, :, 0] *= (1.0 + color_shift * 1.2)  # Blue channel (stronger boost)
        img_float[:, :, 1] *= (1.0 + color_shift * 0.3)  # Green channel (slight boost for cyan)
        img_float[:, :, 2] *= (1.0 - color_shift * 0.7)  # Red channel (stronger reduction)
        
        # ENHANCED: More darkening for overcast rainy day
        img_float *= 0.88  # Increased from 0.92
        
        # Ensure values stay in valid range [0.0, 1.0]
        img_float = np.clip(img_float, 0.0, 1.0)
        
        return (img_float * 255).astype(np.uint8)
    
    def simulate_rain(self, image: np.ndarray) -> np.ndarray:
        """
        Apply complete rain simulation to image
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Rain-augmented image (uint8)
        """
        # Step 1: Apply color grading first
        result = self.apply_color_grading(image)
        
        # Step 2: Create wet surface effect
        result = self.create_wet_surface_effect(result)
        
        # Step 3: Generate and apply puddles
        puddle_mask, ripple_layer = self.generate_puddles(result)
        
        # Blend puddles with reflections
        puddle_mask_3ch = np.stack([puddle_mask] * 3, axis=-1)
        
        # Convert to float for blending
        result_float = result.astype(np.float32) / 255.0
        ripple_float = ripple_layer.astype(np.float32) / 255.0
        
        # Puddles show ripple distortion
        result_float = (
            result_float * (1 - puddle_mask_3ch * 0.4) + 
            ripple_float * puddle_mask_3ch * 0.4
        )
        result_float = np.clip(result_float, 0.0, 1.0)
        result = (result_float * 255).astype(np.uint8)
        
        # Step 4: Add atmospheric effects
        result = self.add_atmospheric_effects(result)
        
        # Step 5: Add water droplets
        result = self.add_water_droplets(result)
        
        # Step 6: Generate and blend rain streaks with SEMI-TRANSPARENT overlay
        rain_mask = self.generate_rain_streaks(result)  # Returns 0.0-1.0 mask
        
        # Convert result to float for blending
        result_float = result.astype(np.float32) / 255.0
        
        # Rain color (light grey-white)
        rain_color = np.array([0.85, 0.85, 0.85], dtype=np.float32)
        
        # Apply rain using alpha blending (semi-transparent overlay)
        # result = background * (1 - alpha) + rain_color * alpha
        rain_mask_3ch = np.stack([rain_mask] * 3, axis=-1)
        result_float = (
            result_float * (1.0 - rain_mask_3ch) + 
            rain_color * rain_mask_3ch
        )
        
        # Ensure values are in valid range
        result_float = np.clip(result_float, 0.0, 1.0)
        
        # Convert back to uint8
        return (result_float * 255).astype(np.uint8)


def process_images(input_dir: Path, output_dir: Path, 
                   rain_intensity: float = 0.8,
                   puddle_density: float = 0.6,
                   wetness_level: float = 0.7,
                   atmospheric_depth: float = 0.5,
                   color_temperature: float = 0.6):
    """
    Process all images in input directory
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed images
        rain_intensity: Rain streak intensity (0.0-1.0)
        puddle_density: Puddle density (0.0-1.0)
        wetness_level: Surface wetness level (0.0-1.0)
        atmospheric_depth: Atmospheric fog/mist strength (0.0-1.0)
        color_temperature: Cool color temperature (0.0-1.0)
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize simulator
    simulator = RainSimulator(
        rain_intensity=rain_intensity,
        puddle_density=puddle_density,
        wetness_level=wetness_level,
        atmospheric_depth=atmospheric_depth,
        color_temperature=color_temperature
    )
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files
    image_files = [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    print(f"Rain Intensity: {rain_intensity}")
    print(f"Puddle Density: {puddle_density}")
    print(f"Wetness Level: {wetness_level}")
    print(f"Atmospheric Depth: {atmospheric_depth}")
    print(f"Color Temperature: {color_temperature}")
    print("-" * 50)
    
    # Process each image
    for idx, image_path in enumerate(image_files, 1):
        try:
            # Read image
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"[{idx}/{len(image_files)}] Failed to read: {image_path.name}")
                continue
            
            # Apply rain simulation
            rain_image = simulator.simulate_rain(image)
            
            # Save result
            output_path = output_dir / f"rain_{image_path.name}"
            cv2.imwrite(str(output_path), rain_image)
            
            print(f"[{idx}/{len(image_files)}] Processed: {image_path.name} -> {output_path.name}")
            
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] Error processing {image_path.name}: {str(e)}")
    
    print("-" * 50)
    print(f"Processing complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Realistic Heavy Rainfall Simulator for Mining Site Data Augmentation'
    )
    
    parser.add_argument(
        '--input_dir',
        type=str,
        default='input_files',
        help='Input directory containing images (default: input_files)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Output directory for processed images (default: output)'
    )
    
    parser.add_argument(
        '--rain_intensity',
        type=float,
        default=0.8,
        help='Rain streak intensity (0.0-1.0, default: 0.8)'
    )
    
    parser.add_argument(
        '--puddle_density',
        type=float,
        default=0.6,
        help='Water puddle density (0.0-1.0, default: 0.6)'
    )
    
    parser.add_argument(
        '--wetness_level',
        type=float,
        default=0.7,
        help='Surface wetness/specularity level (0.0-1.0, default: 0.7)'
    )
    
    parser.add_argument(
        '--atmospheric_depth',
        type=float,
        default=0.5,
        help='Atmospheric fog/mist strength (0.0-1.0, default: 0.5)'
    )
    
    parser.add_argument(
        '--color_temperature',
        type=float,
        default=0.6,
        help='Cool color temperature (0.0-1.0, default: 0.6)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    # Process images
    process_images(
        input_dir=input_dir,
        output_dir=output_dir,
        rain_intensity=args.rain_intensity,
        puddle_density=args.puddle_density,
        wetness_level=args.wetness_level,
        atmospheric_depth=args.atmospheric_depth,
        color_temperature=args.color_temperature
    )


if __name__ == '__main__':
    main()
