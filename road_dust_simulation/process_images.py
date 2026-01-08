"""
Main script for batch processing images with dust simulation
Processes all images in input_files folder and saves augmented versions
"""

import cv2
import os
import sys
from pathlib import Path
from dust_simulator import add_dust_wake
import numpy as np


def detect_vehicle_bbox(image: np.ndarray) -> tuple:
    """
    Placeholder for vehicle detection
    In production, this would use a trained object detection model
    For now, assumes vehicle is in the center-bottom portion of the image
    
    Args:
        image: Input image
    
    Returns:
        (x1, y1, x2, y2) bounding box
    """
    h, w = image.shape[:2]
    
    # Default: assume vehicle occupies center-bottom 40% of image
    x1 = int(w * 0.3)
    x2 = int(w * 0.7)
    y1 = int(h * 0.5)
    y2 = int(h * 0.9)
    
    return (x1, y1, x2, y2)


def process_image(
    input_path: str,
    output_path: str,
    vehicle_bbox: tuple = None,
    vehicle_speed: float = 1.0,
    num_variants: int = 1
):
    """
    Process a single image and generate dust-augmented versions
    
    Args:
        input_path: Path to input image
        output_path: Path to save output image
        vehicle_bbox: Optional (x1, y1, x2, y2) bounding box. If None, auto-detect
        vehicle_speed: Speed multiplier for dust intensity
        num_variants: Number of augmented variants to generate
    """
    # Read image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image {input_path}")
        return
    
    # Detect or use provided bounding box
    if vehicle_bbox is None:
        vehicle_bbox = detect_vehicle_bbox(image)
    
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"  Vehicle bbox: {vehicle_bbox}")
    print(f"  Speed: {vehicle_speed}x")
    
    # Generate variants with different speeds if requested
    if num_variants > 1:
        speeds = np.linspace(0.5, 2.0, num_variants)
    else:
        speeds = [vehicle_speed]
    
    for idx, speed in enumerate(speeds):
        # Apply dust simulation
        result = add_dust_wake(image, vehicle_bbox, vehicle_speed=speed)
        
        # Prepare output filename
        base_name = Path(input_path).stem
        ext = Path(input_path).suffix
        
        if num_variants > 1:
            output_file = f"{base_name}_dust_v{idx+1}_speed{speed:.1f}{ext}"
        else:
            output_file = f"{base_name}_dust{ext}"
        
        final_output_path = os.path.join(os.path.dirname(output_path), output_file)
        
        # Save result
        cv2.imwrite(final_output_path, result)
        print(f"  Saved: {output_file}")


def process_batch(
    input_dir: str = "input_files",
    output_dir: str = "output_images",
    vehicle_speed: float = 1.0,
    num_variants: int = 1
):
    """
    Process all images in input directory
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        vehicle_speed: Default speed multiplier
        num_variants: Number of variants per image
    """
    # Get script directory
    script_dir = Path(__file__).parent
    input_path = script_dir / input_dir
    output_path = script_dir / output_dir
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all images
    image_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {input_path}")
        print("Please add images to the input_files folder")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {output_path}")
    print("-" * 60)
    
    # Process each image
    for img_file in image_files:
        try:
            output_file = output_path / img_file.name
            process_image(
                str(img_file),
                str(output_file),
                vehicle_speed=vehicle_speed,
                num_variants=num_variants
            )
        except Exception as e:
            print(f"Error processing {img_file.name}: {str(e)}")
            continue
    
    print("-" * 60)
    print(f"Processing complete! Check {output_path} for results")


def main():
    """Main entry point with command-line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate realistic dust wake effects on vehicle images"
    )
    parser.add_argument(
        '--input-dir',
        default='input_files',
        help='Input directory containing images (default: input_files)'
    )
    parser.add_argument(
        '--output-dir',
        default='output_images',
        help='Output directory for processed images (default: output_images)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Vehicle speed multiplier (0.5=slow, 1.0=normal, 2.0=fast)'
    )
    parser.add_argument(
        '--variants',
        type=int,
        default=1,
        help='Number of variants to generate per image with different speeds'
    )
    
    args = parser.parse_args()
    
    # Run batch processing
    process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vehicle_speed=args.speed,
        num_variants=args.variants
    )


if __name__ == "__main__":
    main()
