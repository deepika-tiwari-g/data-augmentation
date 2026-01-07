#!/usr/bin/env python3
"""
Fog Simulation for Data Augmentation

This script simulates realistic fog effects on images of mining/industrial sites.
The fog intensity varies based on distance (depth), creating natural atmospheric perspective.

Author: Data Augmentation Pipeline
Date: 2026-01-07
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional, List
import argparse
from datetime import datetime


class FogSimulator:
    """
    Simulates realistic fog effects with depth-based intensity variation.
    
    The fog effect uses atmospheric scattering principles where objects
    further from the camera appear more obscured by fog.
    """
    
    def __init__(self, fog_intensity: str = 'medium'):
        """
        Initialize the fog simulator.
        
        Args:
            fog_intensity: One of 'light', 'medium', 'heavy', 'variable'
        """
        self.fog_intensity = fog_intensity
        self.fog_params = self._get_fog_parameters()
        
    def _get_fog_parameters(self) -> dict:
        """
        Get fog parameters based on intensity level.
        
        Returns:
            Dictionary containing fog parameters
        """
        params = {
            'light': {
                'density': 0.3,
                'color': (220, 230, 235),  # Light grayish-white
                'depth_factor': 0.4,
                'turbulence': 0.1
            },
            'medium': {
                'density': 0.5,
                'color': (200, 210, 220),  # Medium gray-white
                'depth_factor': 0.6,
                'turbulence': 0.15
            },
            'heavy': {
                'density': 0.7,
                'color': (180, 190, 200),  # Heavier gray
                'depth_factor': 0.8,
                'turbulence': 0.2
            },
            'variable': {
                'density': np.random.uniform(0.3, 0.7),
                'color': (
                    np.random.randint(180, 230),
                    np.random.randint(190, 235),
                    np.random.randint(200, 240)
                ),
                'depth_factor': np.random.uniform(0.4, 0.8),
                'turbulence': np.random.uniform(0.1, 0.25)
            }
        }
        
        return params.get(self.fog_intensity, params['medium'])
    
    def _create_depth_map(self, image: np.ndarray, method: str = 'gradient') -> np.ndarray:
        """
        Create a synthetic depth map for the image.
        
        Args:
            image: Input image
            method: Depth estimation method ('gradient', 'edge', 'hybrid')
            
        Returns:
            Normalized depth map (0-1, where 1 is furthest)
        """
        height, width = image.shape[:2]
        
        if method == 'gradient':
            # Simple gradient-based depth (top = far, bottom = near)
            depth_map = np.linspace(1, 0, height)
            depth_map = np.tile(depth_map[:, np.newaxis], (1, width))
            
        elif method == 'edge':
            # Edge-based depth estimation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_blur = cv2.GaussianBlur(edges.astype(float), (21, 21), 0)
            depth_map = 1 - (edges_blur / 255.0)
            
        else:  # hybrid
            # Combine gradient and edge information
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Gradient component
            gradient_depth = np.linspace(1, 0, height)
            gradient_depth = np.tile(gradient_depth[:, np.newaxis], (1, width))
            
            # Edge component (inverted - less edges = further)
            edges = cv2.Canny(gray, 30, 100)
            edges_blur = cv2.GaussianBlur(edges.astype(float), (31, 31), 0)
            edge_depth = 1 - (edges_blur / 255.0) * 0.3
            
            # Brightness component (darker = further in many outdoor scenes)
            brightness = gray.astype(float) / 255.0
            brightness_depth = 1 - brightness * 0.2
            
            # Combine
            depth_map = gradient_depth * 0.6 + edge_depth * 0.25 + brightness_depth * 0.15
            depth_map = np.clip(depth_map, 0, 1)
        
        return depth_map
    
    def _add_turbulence(self, depth_map: np.ndarray, turbulence: float) -> np.ndarray:
        """
        Add turbulence/variation to the depth map for natural fog patchiness.
        
        Args:
            depth_map: Base depth map
            turbulence: Amount of turbulence (0-1)
            
        Returns:
            Depth map with turbulence added
        """
        height, width = depth_map.shape
        
        # Create Perlin-like noise using multiple octaves
        noise = np.zeros_like(depth_map)
        
        for octave in range(3):
            scale = 2 ** octave
            freq = 1.0 / (scale * 50)
            
            # Create coordinate grids
            x = np.linspace(0, width * freq, width)
            y = np.linspace(0, height * freq, height)
            xx, yy = np.meshgrid(x, y)
            
            # Generate smooth noise
            octave_noise = np.sin(xx * 2 * np.pi) * np.cos(yy * 2 * np.pi)
            octave_noise += np.sin(xx * 3 * np.pi + 1.5) * np.cos(yy * 2.5 * np.pi + 0.7)
            
            # Normalize and add to total noise
            octave_noise = (octave_noise - octave_noise.min()) / (octave_noise.max() - octave_noise.min())
            noise += octave_noise / (2 ** octave)
        
        # Normalize final noise
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Apply turbulence
        turbulent_depth = depth_map * (1 - turbulence) + noise * turbulence
        
        return np.clip(turbulent_depth, 0, 1)
    
    def _apply_fog_layer(self, image: np.ndarray, depth_map: np.ndarray, 
                         fog_color: Tuple[int, int, int], density: float) -> np.ndarray:
        """
        Apply fog effect based on depth map.
        
        Args:
            image: Original image
            depth_map: Depth map (0=near, 1=far)
            fog_color: RGB color of fog
            density: Overall fog density
            
        Returns:
            Image with fog applied
        """
        # Convert fog color to float
        fog_color_array = np.array(fog_color, dtype=np.float32)
        
        # Calculate fog amount based on depth and density
        # Using exponential fog model: fog_amount = 1 - exp(-density * depth)
        fog_amount = 1 - np.exp(-density * depth_map * 3)
        
        # Expand fog_amount to 3 channels
        fog_amount_3ch = np.stack([fog_amount] * 3, axis=-1)
        
        # Blend original image with fog color
        fogged_image = image.astype(np.float32) * (1 - fog_amount_3ch) + \
                       fog_color_array * fog_amount_3ch
        
        return np.clip(fogged_image, 0, 255).astype(np.uint8)
    
    def _adjust_contrast_and_saturation(self, image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Reduce contrast and saturation for distant objects (atmospheric perspective).
        
        Args:
            image: Input image
            depth_map: Depth map
            
        Returns:
            Adjusted image
        """
        # Convert to HSV for saturation adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Reduce saturation based on depth
        saturation_reduction = depth_map * 0.6  # Up to 60% reduction
        hsv[:, :, 1] = hsv[:, :, 1] * (1 - saturation_reduction)
        
        # Convert back to BGR
        adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Reduce contrast for distant areas
        contrast_factor = 1 - depth_map * 0.3  # Up to 30% contrast reduction
        contrast_factor = np.stack([contrast_factor] * 3, axis=-1)
        
        mean_color = np.mean(adjusted, axis=(0, 1))
        adjusted = adjusted.astype(np.float32)
        adjusted = mean_color + (adjusted - mean_color) * contrast_factor
        
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def simulate_fog(self, image: np.ndarray, depth_method: str = 'hybrid') -> np.ndarray:
        """
        Apply realistic fog simulation to an image.
        
        Args:
            image: Input image (BGR format)
            depth_method: Method for depth estimation
            
        Returns:
            Image with fog effect applied
        """
        # Refresh parameters for variable mode
        if self.fog_intensity == 'variable':
            self.fog_params = self._get_fog_parameters()
        
        # Create depth map
        depth_map = self._create_depth_map(image, method=depth_method)
        
        # Add turbulence for natural variation
        depth_map = self._add_turbulence(depth_map, self.fog_params['turbulence'])
        
        # Apply depth-based fog density variation
        depth_map = depth_map ** (1.0 / self.fog_params['depth_factor'])
        
        # Adjust contrast and saturation first
        adjusted_image = self._adjust_contrast_and_saturation(image, depth_map)
        
        # Apply fog layer
        fogged_image = self._apply_fog_layer(
            adjusted_image,
            depth_map,
            self.fog_params['color'],
            self.fog_params['density']
        )
        
        return fogged_image
    
    def process_image_file(self, input_path: str, output_path: str, 
                          depth_method: str = 'hybrid') -> bool:
        """
        Process a single image file.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            depth_method: Depth estimation method
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not read image {input_path}")
                return False
            
            # Apply fog simulation
            fogged_image = self.simulate_fog(image, depth_method)
            
            # Save result
            cv2.imwrite(output_path, fogged_image)
            return True
            
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")
            return False
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     depth_method: str = 'hybrid',
                     prefix: str = 'fog_') -> dict:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            depth_method: Depth estimation method
            prefix: Prefix for output filenames
            
        Returns:
            Dictionary with processing statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Get list of image files
        image_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return {'total': 0, 'successful': 0, 'failed': 0}
        
        print(f"\nProcessing {len(image_files)} images with {self.fog_intensity} fog intensity...")
        print(f"Depth method: {depth_method}")
        print(f"Output directory: {output_dir}\n")
        
        successful = 0
        failed = 0
        
        for idx, image_file in enumerate(image_files, 1):
            output_file = output_path / f"{prefix}{image_file.name}"
            
            print(f"[{idx}/{len(image_files)}] Processing {image_file.name}...", end=' ')
            
            if self.process_image_file(str(image_file), str(output_file), depth_method):
                print("✓")
                successful += 1
            else:
                print("✗")
                failed += 1
        
        stats = {
            'total': len(image_files),
            'successful': successful,
            'failed': failed
        }
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total images: {stats['total']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """Main function to run the fog simulator."""
    parser = argparse.ArgumentParser(
        description='Simulate realistic fog effects on images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with medium fog (default)
  python fog_simulator.py
  
  # Process with heavy fog
  python fog_simulator.py --intensity heavy
  
  # Process with variable fog (random intensity per image)
  python fog_simulator.py --intensity variable
  
  # Use custom input/output directories
  python fog_simulator.py -i /path/to/input -o /path/to/output
  
  # Use different depth estimation method
  python fog_simulator.py --depth-method gradient
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        default='input_files',
        help='Input directory containing images (default: input_files)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='output_files',
        help='Output directory for processed images (default: output_files)'
    )
    
    parser.add_argument(
        '--intensity',
        type=str,
        choices=['light', 'medium', 'heavy', 'variable'],
        default='medium',
        help='Fog intensity level (default: medium)'
    )
    
    parser.add_argument(
        '--depth-method',
        type=str,
        choices=['gradient', 'edge', 'hybrid'],
        default='hybrid',
        help='Depth estimation method (default: hybrid)'
    )
    
    parser.add_argument(
        '--prefix',
        type=str,
        default='fog_',
        help='Prefix for output filenames (default: fog_)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("FOG SIMULATION FOR DATA AUGMENTATION")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fog intensity: {args.intensity}")
    print(f"Depth method: {args.depth_method}")
    print("="*60 + "\n")
    
    # Create simulator
    simulator = FogSimulator(fog_intensity=args.intensity)
    
    # Process images
    stats = simulator.process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        depth_method=args.depth_method,
        prefix=args.prefix
    )
    
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()
