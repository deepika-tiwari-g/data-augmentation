#!/usr/bin/env python3
"""
Quick example script demonstrating dust augmentation usage
"""

import cv2
import numpy as np
from dust_augmentation import DustAugmentation
import os

def create_sample_image():
    """Create a sample industrial scene image for testing"""
    # Create a simple test image (simulating a CCTV frame)
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    
    # Add some features (simulating road, vehicle, etc.)
    cv2.rectangle(img, (100, 400), (1180, 600), (80, 80, 80), -1)  # Road
    cv2.rectangle(img, (400, 300), (700, 500), (60, 80, 180), -1)  # Vehicle (reddish)
    cv2.putText(img, "MINE SITE - CCTV CAM 01", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img

def main():
    print("="*60)
    print("Dust Augmentation - Quick Example")
    print("="*60)
    
    # Create sample image
    print("\n1. Creating sample industrial site image...")
    sample_img = create_sample_image()
    
    # Create directories
    os.makedirs("example_input", exist_ok=True)
    os.makedirs("example_output", exist_ok=True)
    
    # Save sample image
    cv2.imwrite("example_input/sample_cctv_frame.jpg", sample_img)
    print("   ✓ Saved sample image to: example_input/sample_cctv_frame.jpg")
    
    # Create augmentor
    print("\n2. Initializing dust augmentation processor...")
    augmentor = DustAugmentation(seed=42)
    print("   ✓ Augmentor ready")
    
    # Apply different dust levels
    print("\n3. Applying dust effects...")
    dust_levels = ['light', 'medium', 'heavy', 'extreme']
    
    for level in dust_levels:
        dusty_img = augmentor.apply_dust_effect(sample_img, dust_level=level)
        output_path = f"example_output/sample_cctv_frame_dust_{level}.jpg"
        cv2.imwrite(output_path, dusty_img)
        print(f"   ✓ Created {level} dust version: {output_path}")
    
    print("\n" + "="*60)
    print("Example complete!")
    print("Check the 'example_output' folder to see the results")
    print("="*60)
    
    print("\n📁 Files created:")
    print("   - example_input/sample_cctv_frame.jpg (original)")
    print("   - example_output/sample_cctv_frame_dust_light.jpg")
    print("   - example_output/sample_cctv_frame_dust_medium.jpg")
    print("   - example_output/sample_cctv_frame_dust_heavy.jpg")
    print("   - example_output/sample_cctv_frame_dust_extreme.jpg")

if __name__ == "__main__":
    main()
