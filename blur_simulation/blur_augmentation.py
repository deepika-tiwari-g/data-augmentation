"""
Blur Augmentation Script for Industrial/Mining Site Images
This script applies various blur effects to images and objects within them.
Useful for simulating camera focus issues, motion blur, and data augmentation.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import random
from typing import Tuple, List
import argparse


class BlurAugmentation:
    """Class to handle various blur augmentation techniques"""
    
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Apply Gaussian blur to the image"""
        # Ensure kernel size is odd
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def motion_blur(self, image: np.ndarray, kernel_size: int = 15, angle: float = 0) -> np.ndarray:
        """Apply motion blur to simulate camera or vehicle movement"""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        
        # Rotate the kernel to apply directional blur
        center = (kernel_size // 2, kernel_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
        
        return cv2.filter2D(image, -1, kernel)
    
    def average_blur(self, image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Apply average blur"""
        return cv2.blur(image, (kernel_size, kernel_size))
    
    def median_blur(self, image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Apply median blur"""
        # Ensure kernel size is odd
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return cv2.medianBlur(image, kernel_size)
    
    def bilateral_blur(self, image: np.ndarray, d: int = 15, sigma_color: int = 80, sigma_space: int = 80) -> np.ndarray:
        """Apply bilateral blur (preserves edges better)"""
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def defocus_blur(self, image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Apply defocus blur using circular kernel"""
        kernel = np.zeros((kernel_size, kernel_size), np.uint8)
        cv2.circle(kernel, (kernel_size // 2, kernel_size // 2), kernel_size // 2, 1, -1)
        kernel = kernel / np.sum(kernel)
        return cv2.filter2D(image, -1, kernel)
    
    def random_region_blur(self, image: np.ndarray, num_regions: int = 3, 
                          blur_type: str = 'gaussian') -> np.ndarray:
        """Apply blur to random regions of the image (simulating objects)"""
        result = image.copy()
        h, w = image.shape[:2]
        
        for _ in range(num_regions):
            # Random region size (10% to 40% of image dimensions)
            region_w = random.randint(int(w * 0.1), int(w * 0.4))
            region_h = random.randint(int(h * 0.1), int(h * 0.4))
            
            # Random position
            x = random.randint(0, max(0, w - region_w))
            y = random.randint(0, max(0, h - region_h))
            
            # Extract region
            region = result[y:y+region_h, x:x+region_w].copy()
            
            # Apply blur to region
            kernel_size = random.randint(7, 21)
            if blur_type == 'gaussian':
                blurred_region = self.gaussian_blur(region, kernel_size)
            elif blur_type == 'motion':
                angle = random.randint(0, 360)
                blurred_region = self.motion_blur(region, kernel_size, angle)
            elif blur_type == 'average':
                blurred_region = self.average_blur(region, kernel_size)
            else:
                blurred_region = self.median_blur(region, kernel_size)
            
            # Blend the blurred region back
            result[y:y+region_h, x:x+region_w] = blurred_region
        
        return result
    
    def progressive_blur(self, image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
        """Apply progressive blur (simulating depth of field)"""
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)
        
        if direction == 'horizontal':
            for i in range(w):
                # Calculate blur intensity based on position
                intensity = int((i / w) * 20) + 1
                intensity = intensity if intensity % 2 == 1 else intensity + 1
                column = result[:, i:i+1].copy()
                blurred = cv2.GaussianBlur(column, (1, intensity), 0)
                result[:, i:i+1] = blurred
        else:  # vertical
            for i in range(h):
                intensity = int((i / h) * 20) + 1
                intensity = intensity if intensity % 2 == 1 else intensity + 1
                row = result[i:i+1, :].copy()
                blurred = cv2.GaussianBlur(row, (intensity, 1), 0)
                result[i:i+1, :] = blurred
        
        return result.astype(np.uint8)
    
    def process_image(self, image_path: Path, blur_types: List[str] = None) -> None:
        """Process a single image with multiple blur variations"""
        if blur_types is None:
            blur_types = ['gaussian', 'motion', 'average', 'median', 'defocus', 
                         'random_region', 'progressive']
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error reading image: {image_path}")
            return
        
        base_name = image_path.stem
        extension = image_path.suffix
        
        print(f"Processing: {image_path.name}")
        
        # Apply different blur types
        for blur_type in blur_types:
            if blur_type == 'gaussian':
                for kernel_size in [7, 15, 25]:
                    blurred = self.gaussian_blur(image, kernel_size)
                    output_name = f"{base_name}_gaussian_k{kernel_size}{extension}"
                    cv2.imwrite(str(self.output_folder / output_name), blurred)
            
            elif blur_type == 'motion':
                for kernel_size in [15, 25]:
                    for angle in [0, 45, 90]:
                        blurred = self.motion_blur(image, kernel_size, angle)
                        output_name = f"{base_name}_motion_k{kernel_size}_a{angle}{extension}"
                        cv2.imwrite(str(self.output_folder / output_name), blurred)
            
            elif blur_type == 'average':
                for kernel_size in [7, 15]:
                    blurred = self.average_blur(image, kernel_size)
                    output_name = f"{base_name}_average_k{kernel_size}{extension}"
                    cv2.imwrite(str(self.output_folder / output_name), blurred)
            
            elif blur_type == 'median':
                for kernel_size in [7, 15]:
                    blurred = self.median_blur(image, kernel_size)
                    output_name = f"{base_name}_median_k{kernel_size}{extension}"
                    cv2.imwrite(str(self.output_folder / output_name), blurred)
            
            elif blur_type == 'defocus':
                for kernel_size in [11, 21]:
                    blurred = self.defocus_blur(image, kernel_size)
                    output_name = f"{base_name}_defocus_k{kernel_size}{extension}"
                    cv2.imwrite(str(self.output_folder / output_name), blurred)
            
            elif blur_type == 'random_region':
                # Simulate blurred objects/vehicles
                for i in range(3):
                    blur_method = random.choice(['gaussian', 'motion', 'average'])
                    blurred = self.random_region_blur(image, num_regions=random.randint(2, 5), 
                                                     blur_type=blur_method)
                    output_name = f"{base_name}_region_blur_{i+1}{extension}"
                    cv2.imwrite(str(self.output_folder / output_name), blurred)
            
            elif blur_type == 'progressive':
                for direction in ['horizontal', 'vertical']:
                    blurred = self.progressive_blur(image, direction)
                    output_name = f"{base_name}_progressive_{direction}{extension}"
                    cv2.imwrite(str(self.output_folder / output_name), blurred)
        
        print(f"  ✓ Generated blur variations for {image_path.name}")
    
    def process_all_images(self, blur_types: List[str] = None) -> None:
        """Process all images in the input folder"""
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Get all image files
        image_files = [f for f in self.input_folder.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {self.input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        print(f"Output will be saved to: {self.output_folder}\n")
        
        for image_file in image_files:
            self.process_image(image_file, blur_types)
        
        print(f"\n✓ Processing complete! All augmented images saved to {self.output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='Blur Augmentation for Industrial/Mining Site Images'
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
        '--blur-types', '-b',
        nargs='+',
        choices=['gaussian', 'motion', 'average', 'median', 'defocus', 
                'random_region', 'progressive', 'all'],
        default=['all'],
        help='Types of blur to apply (default: all)'
    )
    
    args = parser.parse_args()
    
    # Handle 'all' option
    blur_types = args.blur_types
    if 'all' in blur_types:
        blur_types = ['gaussian', 'motion', 'average', 'median', 'defocus', 
                     'random_region', 'progressive']
    
    # Create augmentation instance
    augmenter = BlurAugmentation(args.input, args.output)
    
    # Process all images
    augmenter.process_all_images(blur_types)


if __name__ == "__main__":
    main()
