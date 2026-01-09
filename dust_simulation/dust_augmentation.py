"""
Data Augmentation Script: Dust Accumulation on Camera Lens
Simulates dust/dirt accumulation on CCTV camera lenses for mine/plant site images
Author: Generated for industrial vehicle monitoring dataset creation
"""

import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from typing import Tuple, List
import random


class DustAugmentation:
    """
    Applies realistic dust accumulation effects to images simulating
    dirty camera lenses in industrial environments (mines, plants, etc.)
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize the dust augmentation processor
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_dust_mask(self, shape: Tuple[int, int],
                          intensity: float = 0.2,
                          coverage: float = 0.5) -> np.ndarray:
        """
        Generate a dust/dirt mask overlay
        
        Args:
            shape: Image shape (height, width)
            intensity: Dust opacity (0.0 to 1.0)
            coverage: Percentage of image covered by dust (0.0 to 1.0)
            
        Returns:
            Dust mask as numpy array
        """
        height, width = shape
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Number of dust particles based on coverage
        num_particles = int(coverage * 1000)
        
        # Generate dust spots of varying sizes
        for _ in range(num_particles):
            # Random position
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            # Random size (small to medium spots)
            radius = random.randint(1, 8)  # Reduced for finer dust particles
            
            # Random opacity for this particle
            particle_intensity = random.uniform(0.3, 1.0) * intensity
            
            # Draw circular dust spot with gaussian blur
            cv2.circle(mask, (x, y), radius, particle_intensity, -1)
        
        # Apply gaussian blur to make it more realistic
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def generate_smudge_pattern(self, shape: Tuple[int, int],
                               intensity: float = 0.4) -> np.ndarray:
        """
        Generate smudge/streak patterns (like wiped dust)
        
        Args:
            shape: Image shape (height, width)
            intensity: Smudge opacity
            
        Returns:
            Smudge mask as numpy array
        """
        height, width = shape
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Create 2-4 random smudge streaks
        num_smudges = random.randint(2, 4)
        
        for _ in range(num_smudges):
            # Random starting point
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            
            # Random direction and length
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(50, 200)
            
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            
            # Draw thick line with varying thickness
            thickness = random.randint(10, 30)
            smudge_intensity = random.uniform(0.2, 0.6) * intensity
            
            cv2.line(mask, (x1, y1), (x2, y2), smudge_intensity, thickness)
        
        # Blur to make it look more natural
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        
        return mask
    
    def generate_edge_dust(self, shape: Tuple[int, int],
                          intensity: float = 0.6) -> np.ndarray:
        """
        Generate dust accumulation around edges (common in real scenarios)
        
        Args:
            shape: Image shape (height, width)
            intensity: Edge dust opacity
            
        Returns:
            Edge dust mask as numpy array
        """
        height, width = shape
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Create gradient from edges
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Distance from edges
        dist_from_edges = np.minimum(
            np.minimum(x_coords, width - x_coords),
            np.minimum(y_coords, height - y_coords)
        )
        
        # Normalize and invert (higher values at edges)
        max_dist = min(height, width) // 4
        edge_mask = np.clip(1 - (dist_from_edges / max_dist), 0, 1)
        
        # Add some randomness
        noise = np.random.rand(height, width) * 0.5
        edge_mask = edge_mask * noise * intensity
        
        # Blur for smooth transition
        mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (51, 51), 0)
        
        return mask
    
    def apply_dust_effect(self, image: np.ndarray,
                         dust_level: str = 'medium',
                         custom_params: dict = None) -> np.ndarray:
        """
        Apply dust effect to an image
        
        Args:
            image: Input image (BGR format)
            dust_level: Preset dust level ('light', 'medium', 'heavy', 'extreme')
            custom_params: Custom parameters dict with keys:
                          - dust_intensity: float (0.0-1.0)
                          - dust_coverage: float (0.0-1.0)
                          - smudge_intensity: float (0.0-1.0)
                          - edge_dust_intensity: float (0.0-1.0)
                          - color_tint: tuple (B, G, R) for dust color
        
        Returns:
            Augmented image with dust effect
        """
        height, width = image.shape[:2]
        
        # Preset parameters based on dust level
        presets = {
            'light': {
                'dust_intensity': 0.3,
                'dust_coverage': 0.2,
                'smudge_intensity': 0.2,
                'edge_dust_intensity': 0.3,
                'color_tint': (180, 170, 160)  # Light brownish
            },
            'medium': {
                'dust_intensity': 0.5,
                'dust_coverage': 0.4,
                'smudge_intensity': 0.4,
                'edge_dust_intensity': 0.5,
                'color_tint': (150, 140, 120)  # Medium brown
            },
            'heavy': {
                'dust_intensity': 0.7,
                'dust_coverage': 0.6,
                'smudge_intensity': 0.6,
                'edge_dust_intensity': 0.7,
                'color_tint': (120, 110, 90)  # Dark brown
            },
            'extreme': {
                'dust_intensity': 0.9,
                'dust_coverage': 0.8,
                'smudge_intensity': 0.8,
                'edge_dust_intensity': 0.9,
                'color_tint': (100, 90, 70)  # Very dark brown
            }
        }
        
        # Use custom params or preset
        if custom_params:
            params = custom_params
        else:
            params = presets.get(dust_level, presets['medium'])
        
        # Generate different dust components
        dust_mask = self.generate_dust_mask(
            (height, width),
            intensity=params['dust_intensity'],
            coverage=params['dust_coverage']
        )
        
        smudge_mask = self.generate_smudge_pattern(
            (height, width),
            intensity=params['smudge_intensity']
        )
        
        edge_mask = self.generate_edge_dust(
            (height, width),
            intensity=params['edge_dust_intensity']
        )
        
        # Combine all masks
        combined_mask = np.clip(dust_mask + smudge_mask + edge_mask, 0, 1)
        
        # Convert to 3-channel mask
        combined_mask_3ch = np.stack([combined_mask] * 3, axis=-1)
        
        # Create dust color overlay
        dust_color = np.full_like(image, params['color_tint'], dtype=np.uint8)
        
        # Apply dust effect
        result = image.astype(np.float32)
        dust_overlay = dust_color.astype(np.float32)
        
        # Blend original image with dust overlay
        result = result * (1 - combined_mask_3ch) + dust_overlay * combined_mask_3ch
        
        # Add slight blur to simulate reduced sharpness
        blur_amount = int(params['dust_intensity'] * 3)
        if blur_amount > 0:
            result = cv2.GaussianBlur(result, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
        
        # Reduce contrast slightly
        result = result * (1 - params['dust_intensity'] * 0.2)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def process_folder(self, input_folder: str, output_folder: str,
                      dust_levels: List[str] = None,
                      file_extensions: List[str] = None):
        """
        Process all images in a folder
        
        Args:
            input_folder: Path to input folder containing images
            output_folder: Path to output folder for augmented images
            dust_levels: List of dust levels to apply (creates multiple versions)
                        Default: ['light', 'medium', 'heavy']
            file_extensions: List of file extensions to process
                           Default: ['.jpg', '.jpeg', '.png', '.bmp']
        """
        if dust_levels is None:
            dust_levels = ['light', 'medium', 'heavy']
        
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Create output folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        input_path = Path(input_folder)
        image_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in file_extensions
        ]
        
        print(f"Found {len(image_files)} images to process")
        print(f"Applying dust levels: {dust_levels}")
        print(f"Output folder: {output_folder}\n")
        
        total_processed = 0
        
        for img_file in image_files:
            try:
                # Read image
                image = cv2.imread(str(img_file))
                
                if image is None:
                    print(f"Warning: Could not read {img_file.name}, skipping...")
                    continue
                
                # Apply each dust level
                for dust_level in dust_levels:
                    # Apply dust effect
                    augmented = self.apply_dust_effect(image, dust_level=dust_level)
                    
                    # Create output filename
                    output_name = f"{img_file.stem}_dust_{dust_level}{img_file.suffix}"
                    output_path = Path(output_folder) / output_name
                    
                    # Save augmented image
                    cv2.imwrite(str(output_path), augmented)
                    total_processed += 1
                    
                    print(f"✓ Processed: {img_file.name} -> {output_name} ({dust_level})")
            
            except Exception as e:
                print(f"Error processing {img_file.name}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total images processed: {total_processed}")
        print(f"Output location: {output_folder}")
        print(f"{'='*60}")


def main():
    """Main function to run the script from command line"""
    parser = argparse.ArgumentParser(
        description='Data Augmentation: Simulate dust on camera lens for industrial site images'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='input_files',
        help='Input folder containing images (default: input_files)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output_files',
        help='Output folder for augmented images (default: output_files)'
    )
    
    parser.add_argument(
        '--levels', '-l',
        type=str,
        nargs='+',
        default=['light', 'medium', 'heavy'],
        choices=['light', 'medium', 'heavy', 'extreme'],
        help='Dust levels to apply (space-separated)'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--extensions', '-e',
        type=str,
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp'],
        help='File extensions to process (space-separated, include the dot)'
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist!")
        return
    
    # Create augmentation processor
    augmentor = DustAugmentation(seed=args.seed)
    
    # Process folder
    augmentor.process_folder(
        input_folder=args.input,
        output_folder=args.output,
        dust_levels=args.levels,
        file_extensions=args.extensions
    )


if __name__ == "__main__":
    main()
