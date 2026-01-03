# Blur Augmentation for Industrial/Mining Site Images

This script generates blurred variations of images for data augmentation purposes, particularly useful for mining, plant, and industrial site CCTV camera images featuring vehicles like trucks, dumpers, tippers, graders, dozers, and other heavy machinery.

## Features

The script provides multiple blur augmentation techniques:

1. **Gaussian Blur** - Simulates out-of-focus camera conditions
2. **Motion Blur** - Simulates vehicle or camera movement at different angles
3. **Average Blur** - General smoothing effect
4. **Median Blur** - Reduces noise while preserving edges
5. **Defocus Blur** - Simulates camera defocus with circular kernel
6. **Random Region Blur** - Blurs random regions to simulate blurred objects/vehicles
7. **Progressive Blur** - Simulates depth of field effects

## Installation

### Requirements

```bash
pip install opencv-python numpy
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

1. Place your input images in the `input_files` folder
2. Run the script:

```bash
python blur_augmentation.py
```

The augmented images will be saved in the `output_files` folder.

### Advanced Usage

**Specify custom input/output folders:**
```bash
python blur_augmentation.py --input /path/to/images --output /path/to/output
```

**Apply specific blur types only:**
```bash
python blur_augmentation.py --blur-types gaussian motion
```

**Available blur types:**
- `gaussian` - Gaussian blur with multiple kernel sizes
- `motion` - Motion blur with different angles
- `average` - Average blur
- `median` - Median blur
- `defocus` - Defocus blur
- `random_region` - Random region blur (simulates blurred objects)
- `progressive` - Progressive blur (depth of field)
- `all` - Apply all blur types (default)

### Examples

```bash
# Apply only Gaussian and motion blur
python blur_augmentation.py --blur-types gaussian motion

# Use custom folders
python blur_augmentation.py -i my_images -o augmented_output

# Apply only region blur (for blurring vehicles/objects)
python blur_augmentation.py --blur-types random_region
```

## Output

For each input image, the script generates multiple variations:

- **Gaussian blur**: 3 variations (kernel sizes: 7, 15, 25)
- **Motion blur**: 6 variations (2 kernel sizes × 3 angles: 0°, 45°, 90°)
- **Average blur**: 2 variations (kernel sizes: 7, 15)
- **Median blur**: 2 variations (kernel sizes: 7, 15)
- **Defocus blur**: 2 variations (kernel sizes: 11, 21)
- **Random region blur**: 3 variations (simulating blurred objects/vehicles)
- **Progressive blur**: 2 variations (horizontal and vertical)

**Total**: Up to 20 augmented images per input image (when using all blur types)

## File Naming Convention

Output files are named with descriptive suffixes:
- `original_gaussian_k15.jpg` - Gaussian blur with kernel size 15
- `original_motion_k25_a45.jpg` - Motion blur with kernel 25 at 45° angle
- `original_region_blur_1.jpg` - Random region blur variation 1
- `original_progressive_horizontal.jpg` - Progressive blur in horizontal direction

## Use Cases

- **Data Augmentation**: Increase dataset diversity for training computer vision models
- **Simulating Camera Issues**: Test model robustness against focus problems
- **Motion Simulation**: Simulate moving vehicles or camera shake
- **Privacy**: Blur sensitive regions or license plates
- **Depth of Field**: Simulate different camera focus depths

## Folder Structure

```
blur_simulation/
├── blur_augmentation.py    # Main script
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── input_files/           # Place your input images here
└── output_files/          # Augmented images will be saved here
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Tips

1. **For vehicle blur**: Use `random_region` blur type to simulate blurred vehicles
2. **For motion**: Use `motion` blur with different angles to simulate vehicle movement
3. **For camera issues**: Use `gaussian` or `defocus` blur
4. **For depth effects**: Use `progressive` blur

## Notes

- The script preserves the original image format
- All blur operations are non-destructive (original images remain unchanged)
- Processing time depends on image size and number of blur types selected
- Larger kernel sizes produce stronger blur effects
