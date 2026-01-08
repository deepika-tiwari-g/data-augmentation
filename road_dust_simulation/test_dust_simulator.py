"""
Test script for dust simulation
Demonstrates the dust wake effect with different parameters
"""

import cv2
import numpy as np
from dust_simulator import add_dust_wake
import os


def create_test_image():
    """Create a synthetic test image with a vehicle-like rectangle"""
    # Create a road-like background
    img = np.ones((600, 800, 3), dtype=np.uint8)
    
    # Road color (brownish-grey)
    img[:, :] = [100, 110, 120]
    
    # Add some texture to the road
    noise = np.random.randint(-20, 20, (600, 800, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Draw a simple vehicle (dark rectangle)
    vehicle_color = [40, 50, 60]
    cv2.rectangle(img, (300, 350), (500, 550), vehicle_color, -1)
    
    # Add wheels
    cv2.circle(img, (340, 540), 20, [20, 20, 20], -1)
    cv2.circle(img, (460, 540), 20, [20, 20, 20], -1)
    
    return img


def test_dust_simulation():
    """Test dust simulation with various parameters"""
    
    print("Testing Dust Simulation")
    print("=" * 60)
    
    # Create test image
    test_img = create_test_image()
    
    # Vehicle bounding box (matches the rectangle we drew)
    vehicle_bbox = (300, 350, 500, 550)
    
    # Test different speeds
    test_cases = [
        ("slow_speed", 0.5),
        ("normal_speed", 1.0),
        ("fast_speed", 1.5),
        ("very_fast", 2.0),
    ]
    
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original
    cv2.imwrite(f"{output_dir}/test_original.png", test_img)
    print(f"Saved: test_original.png")
    
    # Generate dust effects
    for name, speed in test_cases:
        print(f"\nGenerating: {name} (speed={speed})")
        result = add_dust_wake(test_img.copy(), vehicle_bbox, vehicle_speed=speed)
        
        output_path = f"{output_dir}/test_{name}.png"
        cv2.imwrite(output_path, result)
        print(f"Saved: test_{name}.png")
    
    print("\n" + "=" * 60)
    print(f"Test complete! Check the '{output_dir}' folder for results")
    print("\nYou should see:")
    print("  - test_original.png: Original image without dust")
    print("  - test_slow_speed.png: Light dust wake")
    print("  - test_normal_speed.png: Moderate dust wake")
    print("  - test_fast_speed.png: Heavy dust wake")
    print("  - test_very_fast.png: Very heavy dust wake")


if __name__ == "__main__":
    test_dust_simulation()
