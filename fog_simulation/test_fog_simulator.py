#!/usr/bin/env python3
"""
Test script for fog simulator.
Creates synthetic test images and applies fog effects to verify functionality.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import sys

# Add parent directory to path to import fog_simulator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fog_simulator import FogSimulator


def create_test_image(width=800, height=600):
    """
    Create a synthetic test image with gradient to simulate depth.
    
    Returns:
        Test image with various elements at different "depths"
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create sky gradient (top = far)
    for y in range(height // 3):
        intensity = int(135 + (y / (height // 3)) * 50)
        image[y, :] = [intensity, intensity + 20, intensity + 40]
    
    # Create ground gradient (bottom = near)
    for y in range(height // 3, height):
        intensity = int(80 - ((y - height // 3) / (2 * height // 3)) * 30)
        image[y, :] = [intensity - 20, intensity, intensity - 10]
    
    # Add some "objects" at different depths
    # Far object (small, top)
    cv2.rectangle(image, (50, 100), (150, 180), (100, 100, 120), -1)
    cv2.putText(image, 'FAR', (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Medium distance object
    cv2.rectangle(image, (300, 200), (500, 350), (80, 120, 100), -1)
    cv2.putText(image, 'MEDIUM', (330, 285), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Near object (large, bottom)
    cv2.rectangle(image, (200, 400), (600, 580), (60, 140, 80), -1)
    cv2.putText(image, 'NEAR', (350, 510), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    return image


def test_fog_intensities():
    """Test all fog intensity levels."""
    print("\n" + "="*60)
    print("TESTING FOG SIMULATOR")
    print("="*60 + "\n")
    
    # Create test directory
    test_dir = Path('test_output')
    test_dir.mkdir(exist_ok=True)
    
    # Create test image
    print("Creating synthetic test image...")
    test_image = create_test_image()
    cv2.imwrite(str(test_dir / 'original.png'), test_image)
    print("✓ Saved original test image\n")
    
    # Test each intensity level
    intensities = ['light', 'medium', 'heavy']
    
    for intensity in intensities:
        print(f"Testing {intensity} fog intensity...")
        
        simulator = FogSimulator(fog_intensity=intensity)
        fogged = simulator.simulate_fog(test_image, depth_method='hybrid')
        
        output_path = test_dir / f'fog_{intensity}.png'
        cv2.imwrite(str(output_path), fogged)
        print(f"✓ Saved {intensity} fog result\n")
    
    # Test variable intensity (3 samples)
    print("Testing variable fog intensity (3 samples)...")
    for i in range(3):
        simulator = FogSimulator(fog_intensity='variable')
        fogged = simulator.simulate_fog(test_image, depth_method='hybrid')
        
        output_path = test_dir / f'fog_variable_{i+1}.png'
        cv2.imwrite(str(output_path), fogged)
    print("✓ Saved 3 variable fog samples\n")
    
    # Create comparison image
    print("Creating comparison image...")
    create_comparison_image(test_dir)
    
    print("="*60)
    print("TEST COMPLETE!")
    print(f"Results saved to: {test_dir.absolute()}")
    print("="*60 + "\n")
    
    print("Visual Verification Checklist:")
    print("  [ ] Far objects are more obscured than near objects")
    print("  [ ] Fog has natural patchiness (not uniform)")
    print("  [ ] Light fog: subtle haze")
    print("  [ ] Medium fog: moderate obscuration")
    print("  [ ] Heavy fog: significant obscuration")
    print("  [ ] Variable fog: different intensities\n")


def create_comparison_image(test_dir):
    """Create a side-by-side comparison of all fog levels."""
    original = cv2.imread(str(test_dir / 'original.png'))
    light = cv2.imread(str(test_dir / 'fog_light.png'))
    medium = cv2.imread(str(test_dir / 'fog_medium.png'))
    heavy = cv2.imread(str(test_dir / 'fog_heavy.png'))
    
    # Add labels
    def add_label(img, text):
        img_copy = img.copy()
        cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2, cv2.LINE_AA)
        return img_copy
    
    original_labeled = add_label(original, 'ORIGINAL')
    light_labeled = add_label(light, 'LIGHT FOG')
    medium_labeled = add_label(medium, 'MEDIUM FOG')
    heavy_labeled = add_label(heavy, 'HEAVY FOG')
    
    # Stack images
    top_row = np.hstack([original_labeled, light_labeled])
    bottom_row = np.hstack([medium_labeled, heavy_labeled])
    comparison = np.vstack([top_row, bottom_row])
    
    cv2.imwrite(str(test_dir / 'comparison.png'), comparison)
    print("✓ Saved comparison image\n")


def test_depth_methods():
    """Test different depth estimation methods."""
    print("\n" + "="*60)
    print("TESTING DEPTH ESTIMATION METHODS")
    print("="*60 + "\n")
    
    test_dir = Path('test_output')
    test_image = create_test_image()
    
    simulator = FogSimulator(fog_intensity='medium')
    
    methods = ['gradient', 'edge', 'hybrid']
    
    for method in methods:
        print(f"Testing {method} depth method...")
        fogged = simulator.simulate_fog(test_image, depth_method=method)
        
        output_path = test_dir / f'depth_{method}.png'
        cv2.imwrite(str(output_path), fogged)
        print(f"✓ Saved {method} result\n")
    
    print("="*60)
    print("DEPTH METHOD TEST COMPLETE!")
    print("="*60 + "\n")


if __name__ == '__main__':
    test_fog_intensities()
    test_depth_methods()
    
    print("\nTo view results:")
    print("  cd test_output")
    print("  # Open comparison.png to see all fog levels side-by-side\n")
