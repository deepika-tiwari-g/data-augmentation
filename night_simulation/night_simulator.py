#!/usr/bin/env python3
"""
Night Simulation Script for Data Augmentation - Enhanced Version

This script converts day images to realistic midnight industrial scenes with:
- Pitch-black darkness with high contrast
- Bright LED vehicle headlights with volumetric light cones
- Warm sodium-vapor street lights with realistic glow
- Ground reflections on damp surfaces
- Lens flare effects
- Volumetric lighting and atmospheric effects

Author: Data Augmentation Pipeline
Date: 2026-01-06
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Optional
import argparse
import json


class NightSimulator:
    """Simulates realistic midnight industrial conditions on day images."""
    
    def __init__(self, config: dict):
        """
        Initialize the night simulator with configuration.
        
        Args:
            config: Dictionary containing simulation parameters
        """
        self.config = config
        # Pitch-black darkness settings
        self.darkening_factor = config.get('darkening_factor', 0.15)
        self.gamma = config.get('gamma', 3.5)
        
        # Light settings - REDUCED for sharp point sources
        self.headlight_brightness = config.get('headlight_brightness', 0.6)  # Reduced
        self.streetlight_brightness = config.get('streetlight_brightness', 0.5)  # Reduced
        self.headlight_enabled = config.get('headlight_enabled', True)
        self.streetlight_enabled = config.get('streetlight_enabled', True)
        
        # Advanced effects - CONTROLLED
        self.volumetric_enabled = config.get('volumetric_enabled', True)
        self.ground_reflection_enabled = config.get('ground_reflection_enabled', True)
        self.lens_flare_enabled = config.get('lens_flare_enabled', True)
        self.reflection_strength = config.get('reflection_strength', 0.25)  # Reduced
        
    def apply_pitch_black_darkening(self, image: np.ndarray) -> np.ndarray:
        """
        Apply darkening to simulate midnight conditions while preserving detail.
        Updated to preserve road textures and gravel detail.
        
        Args:
            image: Input BGR image
            
        Returns:
            Darkened image with preserved texture detail
        """
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply non-linear darkening (moderate gamma to preserve detail)
        darkened = np.power(img_float, self.gamma)
        
        # Apply brightness reduction (less aggressive to keep textures)
        darkened = darkened * self.darkening_factor
        
        # Moderate contrast boost to preserve texture
        darkened = np.clip((darkened - 0.02) * 1.15, 0, 1)
        
        # Reduce color saturation (night desaturation)
        hsv = cv2.cvtColor((darkened * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1] * 0.3).astype(np.uint8)  # Reduce saturation
        darkened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0
        
        # Slight cool tint (very subtle blue for night)
        darkened[:, :, 0] = np.clip(darkened[:, :, 0] * 1.05, 0, 1)  # Blue
        
        # Convert back to uint8
        result = (darkened * 255).astype(np.uint8)
        return result
    
    def detect_vehicle_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential vehicle regions for headlight placement.
        Enhanced detection for better accuracy.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply edge detection with multiple thresholds
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Dilate edges to connect nearby contours
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (potential vehicles)
        height, width = image.shape[:2]
        min_area = (width * height) * 0.008  # At least 0.8% of image
        max_area = (width * height) * 0.45   # At most 45% of image
        
        vehicle_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio (vehicles are typically wider than tall)
                aspect_ratio = w / h if h > 0 else 0
                if 0.4 < aspect_ratio < 5.0:
                    vehicle_regions.append((x, y, w, h))
        
        return vehicle_regions
    
    def find_headlight_positions(self, original_image: np.ndarray, 
                                vehicle_box: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """
        Find actual headlight positions by detecting brightest pixels in vehicle region.
        This allows realistic placement on angled vehicles.
        
        Args:
            original_image: Original day image (before darkening)
            vehicle_box: (x, y, w, h) bounding box of vehicle
        
        Returns:
            List of (x, y) coordinates for headlights (up to 2)
        """
        x, y, w, h = vehicle_box
        
        # Focus on bottom 50% of vehicle (where headlights typically are)
        search_y_start = y + int(h * 0.5)
        search_y_end = y + h
        
        # Ensure we don't go out of bounds
        search_y_start = max(0, search_y_start)
        search_y_end = min(original_image.shape[0], search_y_end)
        
        if search_y_start >= search_y_end:
            # Fallback if region is invalid
            center_y = y + h - int(h * 0.25)
            return [(x + int(w * 0.3), center_y), (x + int(w * 0.7), center_y)]
        
        search_region = original_image[search_y_start:search_y_end, x:x+w]
        
        # Convert to grayscale and find brightest pixels
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find very bright pixels (headlight glass/reflections)
        # Use adaptive threshold to handle varying lighting conditions
        mean_brightness = np.mean(gray)
        threshold_value = max(200, min(250, int(mean_brightness * 1.5)))
        _, bright_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        headlight_positions = []
        
        if len(contours) >= 2:
            # Sort by area and get top 2 largest bright regions
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            
            for contour in sorted_contours:
                area = cv2.contourArea(contour)
                # Filter out very small regions (noise)
                if area > 5:  # Minimum area threshold
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        # Calculate centroid in original image coordinates
                        cx = int(M['m10'] / M['m00']) + x
                        cy = int(M['m01'] / M['m00']) + search_y_start
                        headlight_positions.append((cx, cy))
        
        # Fallback to geometric placement if detection fails
        if len(headlight_positions) < 2:
            center_y = y + h - int(h * 0.25)
            headlight_positions = [
                (x + int(w * 0.3), center_y),
                (x + int(w * 0.7), center_y)
            ]
        
        # Ensure we return exactly 2 positions
        return headlight_positions[:2]
    
    def calculate_beam_direction(self, light_pos: Tuple[int, int], 
                                image_width: int) -> Tuple[float, float]:
        """
        Calculate beam direction based on light position in frame.
        Beams angle toward center for more realistic lighting.
        
        Args:
            light_pos: (x, y) position of light
            image_width: Width of image
        
        Returns:
            (dx, dy) normalized direction vector
        """
        x, y = light_pos
        center_x = image_width / 2
        
        # Calculate horizontal component based on position
        if x < center_x * 0.4:  # Left side
            dx = 0.15  # Angle toward right
        elif x > center_x * 1.6:  # Right side
            dx = -0.15  # Angle toward left
        else:  # Center
            dx = 0.05  # Mostly straight
        
        # Vertical component (mostly downward)
        dy = 0.99
        
        # Normalize
        magnitude = np.sqrt(dx**2 + dy**2)
        return (dx / magnitude, dy / magnitude)
    
    def detect_vertical_poles(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect vertical poles/structures using Hough Line Transform.
        
        Args:
            image: Input BGR image
        
        Returns:
            List of (x, y) coordinates at top of detected poles
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=100,
                               minLineLength=int(image.shape[0] * 0.2),
                               maxLineGap=10)
        
        pole_positions = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle of line
                if x2 - x1 != 0:
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                else:
                    angle = 90  # Perfectly vertical
                
                # Check if line is vertical (angle close to 90 degrees)
                if 80 <= angle <= 100:  # Nearly vertical
                    # Use top point of line
                    top_x = x1 if y1 < y2 else x2
                    top_y = min(y1, y2)
                    
                    # Avoid duplicates (merge nearby poles)
                    is_duplicate = False
                    for existing_x, existing_y in pole_positions:
                        if abs(top_x - existing_x) < 50 and abs(top_y - existing_y) < 50:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        pole_positions.append((top_x, top_y))
        
        return pole_positions
    
    def add_volumetric_light_cone(self, image: np.ndarray, start_pos: Tuple[int, int], 
                                   direction: Tuple[float, float], length: int,
                                   width: int, intensity: float, color: Tuple[int, int, int]) -> None:
        """
        Add NARROW volumetric light cone effect (focused headlight beam).
        
        Args:
            image: Image to modify (in-place)
            start_pos: (x, y) starting position of light
            direction: (dx, dy) normalized direction vector
            length: Length of light cone
            width: Width at the end of cone (NARROW)
            intensity: Light intensity (CONTROLLED)
            color: BGR color of light
        """
        height, width_img = image.shape[:2]
        overlay = image.copy().astype(np.float32)
        
        # Create cone mask
        x_start, y_start = start_pos
        dx, dy = direction
        
        # Create gradient along the cone
        Y, X = np.ogrid[:height, :width_img]
        
        # Calculate distance from cone axis
        px = X - x_start
        py = Y - y_start
        
        # Project onto direction vector
        proj_length = px * dx + py * dy
        proj_length = np.clip(proj_length, 0, length)
        
        # Perpendicular distance from axis
        perp_dist = np.abs(px * (-dy) + py * dx)
        
        # NARROW cone - starts small, stays relatively narrow
        start_width = 3  # Very narrow start
        cone_width = start_width + (proj_length / length) * width
        
        # Create cone mask with SHARP edges
        cone_mask = np.where(
            (proj_length > 0) & (proj_length < length) & (perp_dist < cone_width),
            1.0, 0.0
        )
        
        # AGGRESSIVE falloff for focused beam
        length_falloff = 1.0 - (proj_length / length) * 0.7  # Stronger falloff
        width_falloff = np.power(1.0 - (perp_dist / (cone_width + 1)), 2)  # Quadratic falloff
        
        cone_mask = cone_mask * length_falloff * width_falloff * intensity
        cone_mask = np.clip(cone_mask, 0, 1)
        
        # Apply colored light with REDUCED intensity
        for i, c in enumerate(color):
            overlay[:, :, i] = overlay[:, :, i] + cone_mask * c * 0.4  # Reduced to 40%
        
        # Blend with original
        np.copyto(image, np.clip(overlay, 0, 255).astype(np.uint8))
    
    def add_bright_led_headlights(self, image: np.ndarray, 
                                  original_image: np.ndarray,
                                  vehicle_regions: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Add SMALL, SHARP LED headlights with intelligent placement and directional beams.
        Uses brightness detection to find actual headlight positions.
        
        Args:
            image: Darkened image to add lights to
            original_image: Original day image for brightness detection
            vehicle_regions: List of vehicle bounding boxes
            
        Returns:
            Image with sharp LED headlights added
        """
        result = image.copy()
        height, width = image.shape[:2]
        
        for vehicle_box in vehicle_regions:
            # Find actual headlight positions using brightness detection
            headlight_positions = self.find_headlight_positions(original_image, vehicle_box)
            
            # LED headlights are bright white (cool white)
            led_color = (255, 255, 255)  # Pure white for LED
            
            for light_x, light_y in headlight_positions:
                if 0 <= light_x < width and 0 <= light_y < height:
                    # Add SMALL, SHARP point source
                    x, y, w, h = vehicle_box
                    self._add_sharp_point_light(result, (light_x, light_y),
                                               radius=max(8, int(min(w, h) * 0.05)),
                                               intensity=self.headlight_brightness,
                                               color=led_color)
                    
                    # Add NARROW volumetric light cone with directional angle
                    if self.volumetric_enabled:
                        cone_length = int(height * 0.5)
                        cone_width = int(width * 0.08)
                        
                        # Calculate direction based on position in frame
                        direction = self.calculate_beam_direction((light_x, light_y), width)
                        
                        self.add_volumetric_light_cone(result, (light_x, light_y),
                                                       direction, cone_length, cone_width,
                                                       self.headlight_brightness * 0.3,
                                                       led_color)
                    
                    # Add SUBTLE lens flare if enabled
                    if self.lens_flare_enabled:
                        self._add_subtle_lens_flare(result, (light_x, light_y),
                                                   intensity=self.headlight_brightness * 0.2)
        
        return result
    
    def add_site_infrastructure_lights(self, image: np.ndarray,
                                       original_image: np.ndarray,
                                       custom_coords: Optional[List[Tuple[int, int]]] = None,
                                       auto_detect: bool = True) -> np.ndarray:
        """
        Add sodium-vapor lights at infrastructure locations (poles, towers).
        Can use custom coordinates or auto-detect vertical poles.
        
        Args:
            image: Darkened image to add lights to
            original_image: Original day image for pole detection
            custom_coords: Optional list of (x, y) coordinates for lights
            auto_detect: Use pole detection if no custom coords provided
        
        Returns:
            Image with infrastructure lights added
        """
        result = image.copy()
        height, width = image.shape[:2]
        
        # Get light positions
        if custom_coords:
            light_positions = custom_coords
        elif auto_detect:
            light_positions = self.detect_vertical_poles(original_image)
            # If no poles detected, fallback to grid
            if not light_positions:
                return self.add_sodium_vapor_streetlights(result)
        else:
            # Fallback to grid
            return self.add_sodium_vapor_streetlights(result)
        
        # Sodium vapor color: warm orange-yellow
        sodium_color = (80, 160, 240)  # BGR: Orange-yellow
        
        for light_x, light_y in light_positions:
            # Add SMALL, SHARP point source
            self._add_sharp_point_light(result, (light_x, light_y),
                                       radius=max(10, int(width * 0.015)),
                                       intensity=self.streetlight_brightness,
                                       color=sodium_color)
            
            # Add SUBTLE glow halo
            self._add_sharp_point_light(result, (light_x, light_y),
                                       radius=max(20, int(width * 0.04)),
                                       intensity=self.streetlight_brightness * 0.3,
                                       color=sodium_color)
            
            # Add NARROW volumetric downward cone
            if self.volumetric_enabled:
                cone_length = int(height * 0.6)
                cone_width = int(width * 0.06)
                direction = (0, 1)  # Straight down
                
                self.add_volumetric_light_cone(result, (light_x, light_y),
                                               direction, cone_length, cone_width,
                                               self.streetlight_brightness * 0.25,
                                               sodium_color)
        
        return result
    
    def add_sodium_vapor_streetlights(self, image: np.ndarray, num_lights: Optional[int] = None) -> np.ndarray:
        """
        Add SMALL, DEFINED sodium-vapor street lights with SUBTLE glow.
        
        Args:
            image: Input BGR image
            num_lights: Number of street lights (auto-detected if None)
            
        Returns:
            Image with sharp street lights added
        """
        result = image.copy()
        height, width = image.shape[:2]
        
        # Auto-detect number of lights based on image width
        if num_lights is None:
            num_lights = max(1, width // 400)  # Fewer lights
        
        # Place lights at top portion of image
        light_y = int(height * 0.12)  # Upper portion
        
        # Distribute lights evenly
        spacing = width // (num_lights + 1)
        
        # Sodium vapor color: warm orange-yellow
        sodium_color = (80, 160, 240)  # BGR: Orange-yellow (slightly muted)
        
        for i in range(1, num_lights + 1):
            light_x = spacing * i
            
            # Add SMALL, SHARP point source
            self._add_sharp_point_light(result, (light_x, light_y),
                                       radius=max(10, int(width * 0.015)),  # MUCH smaller
                                       intensity=self.streetlight_brightness,
                                       color=sodium_color)
            
            # Add SUBTLE glow halo (optional, very small)
            self._add_sharp_point_light(result, (light_x, light_y),
                                       radius=max(20, int(width * 0.04)),  # Small halo
                                       intensity=self.streetlight_brightness * 0.3,  # Subtle
                                       color=sodium_color)
            
            # Add NARROW volumetric downward cone
            if self.volumetric_enabled:
                cone_length = int(height * 0.6)  # Shorter
                cone_width = int(width * 0.06)  # MUCH narrower
                direction = (0, 1)  # Straight down
                
                self.add_volumetric_light_cone(result, (light_x, light_y),
                                               direction, cone_length, cone_width,
                                               self.streetlight_brightness * 0.25,  # Reduced
                                               sodium_color)
        
        return result
    
    def add_ground_reflections(self, image: np.ndarray) -> np.ndarray:
        """
        Add ground reflections to simulate damp/wet surface.
        
        Args:
            image: Input BGR image
            
        Returns:
            Image with ground reflections
        """
        result = image.copy()
        height, width = image.shape[:2]
        
        # Define ground region (bottom 40% of image)
        ground_start = int(height * 0.6)
        
        # Extract the upper portion (lights)
        upper_portion = result[:ground_start, :]
        
        # Flip vertically for reflection
        reflection = cv2.flip(upper_portion, 0)
        
        # Create gradient mask (stronger at top of ground, fading down)
        gradient = np.linspace(self.reflection_strength, 0, ground_start)
        gradient = gradient[:, np.newaxis, np.newaxis]
        
        # Apply reflection to ground area
        ground_region = result[ground_start:ground_start + ground_start, :]
        
        # Ensure sizes match
        if ground_region.shape[0] == reflection.shape[0]:
            # Blend reflection with ground
            reflection_blurred = cv2.GaussianBlur(reflection, (15, 15), 0)
            blended = (ground_region.astype(np.float32) * (1 - gradient) + 
                      reflection_blurred.astype(np.float32) * gradient)
            result[ground_start:ground_start + ground_start, :] = blended.astype(np.uint8)
        
        return result
    
    def _add_sharp_point_light(self, image: np.ndarray, position: Tuple[int, int],
                               radius: int, intensity: float, color: Tuple[int, int, int]) -> None:
        """
        Add a SMALL, SHARP point light source with CONTROLLED glow.
        
        Args:
            image: Image to modify (in-place)
            position: (x, y) center of light
            radius: Radius of light glow (SMALL)
            intensity: Light intensity (CONTROLLED)
            color: BGR color of light
        """
        height, width = image.shape[:2]
        x, y = position
        
        # Create overlay for light
        overlay = image.copy().astype(np.float32)
        
        # Create radial gradient for glow effect
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
        
        # SHARP, FOCUSED core with AGGRESSIVE falloff
        # Use higher exponent for sharper edges
        core_mask = np.exp(-(dist_from_center**3) / (2 * (radius * 0.4)**3))  # Cubic falloff
        core_mask = np.clip(core_mask * intensity, 0, 1)
        
        # Apply colored glow with CONTROLLED intensity
        for i, c in enumerate(color):
            overlay[:, :, i] = overlay[:, :, i] + core_mask * c * 0.7  # Reduced multiplier
        
        # Blend with original
        np.copyto(image, np.clip(overlay, 0, 255).astype(np.uint8))
    
    def _add_subtle_lens_flare(self, image: np.ndarray, position: Tuple[int, int], intensity: float) -> None:
        """
        Add SUBTLE, SMALL lens flare effect around bright lights.
        
        Args:
            image: Image to modify (in-place)
            position: (x, y) center of light source
            intensity: Flare intensity (REDUCED)
        """
        height, width = image.shape[:2]
        x, y = position
        
        overlay = image.copy().astype(np.float32)
        
        # Create SMALL hexagonal flare pattern
        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        
        # FEWER, SMALLER flare rings
        for i, radius in enumerate([15, 30]):  # Only 2 rings, smaller
            ring_mask = np.exp(-((dist - radius)**2) / (2 * 5**2))  # Sharper falloff
            ring_mask = np.clip(ring_mask * intensity * 0.15 / (i + 1), 0, 1)  # Much weaker
            
            # Add subtle white flare
            for c in range(3):
                overlay[:, :, c] = overlay[:, :, c] + ring_mask * 80  # Reduced brightness
        
        # Add SUBTLE star burst effect
        angle = np.arctan2(Y - y, X - x)
        starburst = np.abs(np.sin(angle * 4)) * np.exp(-dist / 50)  # Smaller, 4-point
        starburst = np.clip(starburst * intensity * 0.15, 0, 1)  # Much weaker
        
        for c in range(3):
            overlay[:, :, c] = overlay[:, :, c] + starburst * 60  # Reduced brightness
        
        np.copyto(image, np.clip(overlay, 0, 255).astype(np.uint8))
    
    def simulate_night(self, image: np.ndarray, 
                      detect_vehicles: bool = True,
                      infrastructure_coords: Optional[List[Tuple[int, int]]] = None,
                      auto_detect_poles: bool = True) -> np.ndarray:
        """
        Apply complete midnight industrial simulation to an image.
        
        Args:
            image: Input BGR image
            detect_vehicles: Whether to auto-detect vehicles for headlights
            infrastructure_coords: Optional list of (x, y) coordinates for infrastructure lights
            auto_detect_poles: Whether to auto-detect vertical poles for lights
            
        Returns:
            Night-simulated image with intelligent lighting
        """
        # Store original for feature detection
        original_image = image.copy()
        
        # Step 1: Apply darkening (preserves texture detail)
        result = self.apply_pitch_black_darkening(image)
        
        # Step 2: Add infrastructure lights (poles/towers)
        if self.streetlight_enabled:
            result = self.add_site_infrastructure_lights(result, original_image,
                                                         custom_coords=infrastructure_coords,
                                                         auto_detect=auto_detect_poles)
        
        # Step 3: Add vehicle headlights with intelligent placement
        if self.headlight_enabled and detect_vehicles:
            vehicle_regions = self.detect_vehicle_regions(original_image)
            if vehicle_regions:
                result = self.add_bright_led_headlights(result, original_image, vehicle_regions)
        
        # Step 4: Add ground reflections if enabled
        if self.ground_reflection_enabled:
            result = self.add_ground_reflections(result)
        
        return result


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from JSON file or use defaults.
    
    Args:
        config_path: Path to config JSON file
        
    Returns:
        Configuration dictionary
    """
    default_config = {
        'darkening_factor': 0.35,        # Increased from 0.15 for better visibility
        'gamma': 2.2,                   # Reduced from 3.5 to keep road textures
        'headlight_brightness': 0.9,     # Increased to make beams "pop"
        'streetlight_brightness': 0.8,   # Increased for better environment light
        'headlight_enabled': True,
        'streetlight_enabled': True,
        'volumetric_enabled': True,
        'ground_reflection_enabled': True,
        'lens_flare_enabled': True,
        'reflection_strength': 0.45,     # Increased to help define the road surface
        'detect_vehicles': True
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            default_config.update(user_config)
    
    return default_config


def process_images(input_dir: str, output_dir: str, config: dict, 
                  infrastructure_coords: Optional[List[Tuple[int, int]]] = None,
                  verbose: bool = True):
    """
    Process all images in input directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output images
        config: Configuration dictionary
        infrastructure_coords: Optional list of (x, y) coordinates for infrastructure lights
        verbose: Print progress messages
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize simulator
    simulator = NightSimulator(config)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get list of images
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print("Applying midnight industrial simulation with intelligent lighting:")
    print("  ✓ Pitch-black darkness")
    print("  ✓ Feature-based headlight detection")
    print("  ✓ Directional beam angles")
    print("  ✓ Infrastructure pole detection")
    print("  ✓ Sodium-vapor street lights")
    print("  ✓ Ground reflections & lens flare\n")
    
    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        if verbose:
            print(f"Processing [{idx}/{len(image_files)}]: {image_file.name}")
        
        # Read image
        image = cv2.imread(str(image_file))
        if image is None:
            print(f"  Warning: Could not read {image_file.name}, skipping...")
            continue
        
        # Apply night simulation with intelligent lighting
        night_image = simulator.simulate_night(image, 
                                               detect_vehicles=config.get('detect_vehicles', True),
                                               infrastructure_coords=infrastructure_coords,
                                               auto_detect_poles=True)
        
        # Save result
        output_path = Path(output_dir) / f"night_{image_file.name}"
        cv2.imwrite(str(output_path), night_image)
        
        if verbose:
            print(f"  ✓ Saved to: {output_path.name}")
    
    print(f"\n✓ Processing complete! {len(image_files)} images saved to {output_dir}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Convert day images to realistic midnight industrial scenes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python night_simulator.py
  
  # Custom input/output directories
  python night_simulator.py -i ./my_images -o ./night_output
  
  # Use custom configuration
  python night_simulator.py -c config.json
  
  # Disable vehicle detection
  python night_simulator.py --no-detect-vehicles
  
  # Disable specific effects
  python night_simulator.py --no-volumetric --no-reflections
        """
    )
    
    parser.add_argument('-i', '--input', 
                       default='input_files',
                       help='Input directory containing day images (default: input_files)')
    
    parser.add_argument('-o', '--output',
                       default='output_files',
                       help='Output directory for night images (default: output_files)')
    
    parser.add_argument('-c', '--config',
                       help='Path to JSON configuration file')
    
    parser.add_argument('--no-detect-vehicles',
                       action='store_true',
                       help='Disable automatic vehicle detection for headlights')
    
    parser.add_argument('--no-volumetric',
                       action='store_true',
                       help='Disable volumetric light cones')
    
    parser.add_argument('--no-reflections',
                       action='store_true',
                       help='Disable ground reflections')
    
    parser.add_argument('--no-lens-flare',
                       action='store_true',
                       help='Disable lens flare effects')
    
    parser.add_argument('-q', '--quiet',
                       action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.no_detect_vehicles:
        config['detect_vehicles'] = False
    if args.no_volumetric:
        config['volumetric_enabled'] = False
    if args.no_reflections:
        config['ground_reflection_enabled'] = False
    if args.no_lens_flare:
        config['lens_flare_enabled'] = False
    
    # Process images
    process_images(args.input, args.output, config, verbose=not args.quiet)


if __name__ == '__main__':
    main()
