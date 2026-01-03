#!/usr/bin/env python3
"""
Twilight Simulation Script for Data Augmentation

This script converts daytime images to twilight images by:
1. Detecting and modifying the sky region with twilight colors
2. Darkening the overall image to simulate twilight lighting conditions

Author: Data Augmentation Project
Date: 2026-01-03
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


class TwilightSimulator:
    """Simulates evening/dusk lighting conditions using brightness, contrast, and color adjustments."""
    
    def __init__(self, brightness_reduction=0.3, contrast_reduction=0.05, 
                 saturation_reduction=15, value_reduction=15):
        """
        Initialize the TwilightSimulator with evening lighting parameters.
        
        Args:
            brightness_reduction (float): How much to reduce brightness (0.0 to 1.0)
            contrast_reduction (float): How much to reduce contrast (0.0 to 1.0)
            saturation_reduction (int): How much to reduce saturation (0 to 255)
            value_reduction (int): How much to reduce value/brightness in HSV (0 to 255)
        """
        self.brightness_reduction = brightness_reduction
        self.contrast_reduction = contrast_reduction
        self.saturation_reduction = saturation_reduction
        self.value_reduction = value_reduction
    
    def adjust_brightness_contrast(self, image, brightness_factor, contrast_factor):
        """
        Adjust brightness and contrast of the image.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            brightness_factor (float): Brightness adjustment (-1.0 to 1.0, negative = darker)
            contrast_factor (float): Contrast adjustment (-1.0 to 1.0, negative = lower contrast)
            
        Returns:
            numpy.ndarray: Adjusted image
        """
        # Convert to float for calculations
        img = image.astype(np.float32)
        
        # Apply brightness adjustment
        # brightness_factor: -1.0 (black) to 1.0 (white)
        img = img + (brightness_factor * 255)
        
        # Apply contrast adjustment
        # contrast_factor: -1.0 (gray) to 1.0 (high contrast)
        # Formula: img = (img - 127.5) * (1 + contrast_factor) + 127.5
        img = (img - 127.5) * (1 + contrast_factor) + 127.5
        
        # Clip values to valid range
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def adjust_hsv(self, image, sat_shift, val_shift):
        """
        Adjust saturation and value in HSV color space.
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            sat_shift (int): Saturation shift (-255 to 255)
            val_shift (int): Value shift (-255 to 255)
            
        Returns:
            numpy.ndarray: Adjusted image
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        
        # Adjust saturation (S channel)
        hsv[:, :, 1] = hsv[:, :, 1] + sat_shift
        
        # Adjust value (V channel)
        hsv[:, :, 2] = hsv[:, :, 2] + val_shift
        
        # Clip values to valid range
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        
        # Convert back to BGR
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return result
    
    def simulate_twilight(self, image):
        """
        Apply evening/dusk lighting simulation to an image.
        
        This method applies:
        1. Brightness reduction (darkening)
        2. Contrast reduction (softer lighting)
        3. Saturation reduction (muted colors)
        4. Value reduction (dimmer appearance)
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            numpy.ndarray: Evening-simulated image
        """
        # Step 1: Adjust brightness and contrast
        # Darken the image and reduce contrast for softer evening lighting
        result = self.adjust_brightness_contrast(
            image,
            brightness_factor=-self.brightness_reduction,  # Negative = darker
            contrast_factor=-self.contrast_reduction        # Negative = lower contrast
        )
        
        # Step 2: Adjust HSV (saturation and value)
        # Reduce saturation for muted colors
        # Reduce value for dimmer appearance
        result = self.adjust_hsv(
            result,
            sat_shift=-self.saturation_reduction,  # Negative = less saturated
            val_shift=-self.value_reduction        # Negative = dimmer
        )
        
        return result


def process_images(input_dir, output_dir, brightness_reduction=0.3, contrast_reduction=0.05,
                   saturation_reduction=15, value_reduction=15):
    """
    Process all images in the input directory and save evening versions.
    
    Args:
        input_dir (str): Path to input directory containing images
        output_dir (str): Path to output directory for processed images
        brightness_reduction (float): How much to reduce brightness (0.0 to 1.0)
        contrast_reduction (float): How much to reduce contrast (0.0 to 1.0)
        saturation_reduction (int): How much to reduce saturation (0 to 255)
        value_reduction (int): How much to reduce value/brightness in HSV (0 to 255)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize simulator
    simulator = TwilightSimulator(brightness_reduction, contrast_reduction,
                                   saturation_reduction, value_reduction)
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Brightness Reduction: {brightness_reduction}")
    print(f"Contrast Reduction: {contrast_reduction}")
    print(f"Saturation Reduction: {saturation_reduction}")
    print(f"Value Reduction: {value_reduction}")
    print("-" * 50)
    
    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        try:
            # Read image
            image = cv2.imread(str(image_file))
            
            if image is None:
                print(f"[{idx}/{len(image_files)}] ❌ Failed to read: {image_file.name}")
                continue
            
            # Apply twilight simulation
            twilight_image = simulator.simulate_twilight(image)
            
            # Generate output filename
            output_filename = f"twilight_{image_file.name}"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save processed image
            cv2.imwrite(output_path, twilight_image)
            
            print(f"[{idx}/{len(image_files)}] ✓ Processed: {image_file.name} -> {output_filename}")
            
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] ❌ Error processing {image_file.name}: {str(e)}")
    
    print("-" * 50)
    print(f"✓ Processing complete! Output saved to: {output_dir}")


def main():
    """Main function to parse arguments and run the evening/dusk simulation."""
    parser = argparse.ArgumentParser(
        description='Convert daytime images to evening/dusk images for data augmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python twilight_simulator.py
  
  # Custom parameters
  python twilight_simulator.py --brightness 0.4 --contrast 0.1 --saturation 20 --value 20
  
  # Custom input/output directories
  python twilight_simulator.py --input ./my_images --output ./evening_output
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='input_files',
        help='Input directory containing images (default: input_files)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output_files',
        help='Output directory for evening images (default: output_files)'
    )
    
    parser.add_argument(
        '--brightness',
        type=float,
        default=0.3,
        help='Brightness reduction (0.0 to 1.0, default: 0.3)'
    )
    
    parser.add_argument(
        '--contrast',
        type=float,
        default=0.05,
        help='Contrast reduction (0.0 to 1.0, default: 0.05)'
    )
    
    parser.add_argument(
        '--saturation',
        type=int,
        default=15,
        help='Saturation reduction (0 to 255, default: 15)'
    )
    
    parser.add_argument(
        '--value',
        type=int,
        default=15,
        help='Value/brightness reduction in HSV (0 to 255, default: 15)'
    )
    
    args = parser.parse_args()
    
    # Validate parameters
    if not 0.0 <= args.brightness <= 1.0:
        print("Error: Brightness must be between 0.0 and 1.0")
        return
    
    if not 0.0 <= args.contrast <= 1.0:
        print("Error: Contrast must be between 0.0 and 1.0")
        return
    
    if not 0 <= args.saturation <= 255:
        print("Error: Saturation must be between 0 and 255")
        return
    
    if not 0 <= args.value <= 255:
        print("Error: Value must be between 0 and 255")
        return
    
    # Check if input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return
    
    # Process images
    process_images(args.input, args.output, args.brightness, args.contrast,
                   args.saturation, args.value)


if __name__ == "__main__":
    main()
