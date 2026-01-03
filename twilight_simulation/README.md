# Twilight Simulation - Data Augmentation

This tool converts daytime images to twilight images for data augmentation purposes. It's specifically designed for industrial/mining site images containing vehicles like trucks, dumpers, tippers, graders, dozers, and other heavy machinery.

## Features

- **Sky Detection**: Automatically detects sky regions in images using color-based segmentation
- **Twilight Color Gradient**: Applies realistic twilight colors (deep blue → purple → orange → pink)
- **Image Darkening**: Simulates twilight lighting conditions by darkening the overall scene
- **Twilight Ambiance**: Adds subtle color tinting for atmospheric effect
- **Batch Processing**: Process multiple images at once
- **Customizable Parameters**: Adjust twilight intensity and darkness levels

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Place your input images in the `input_files` folder and run:

```bash
python twilight_simulator.py
```

This will process all images and save the twilight versions in the `output_files` folder with the prefix `twilight_`.

### Advanced Usage

#### Custom Intensity and Darkness

```bash
# Higher twilight intensity (more dramatic sky colors)
python twilight_simulator.py --intensity 0.8 --darkness 0.6

# Lighter twilight effect
python twilight_simulator.py --intensity 0.4 --darkness 0.3
```

#### Custom Input/Output Directories

```bash
python twilight_simulator.py --input ./my_images --output ./twilight_results
```

### Parameters

- `--input`: Input directory containing images (default: `input_files`)
- `--output`: Output directory for processed images (default: `output_files`)
- `--intensity`: Twilight intensity for sky coloring (0.0 to 1.0, default: 0.6)
  - Lower values: Subtle twilight effect
  - Higher values: More dramatic twilight colors
- `--darkness`: Overall image darkness factor (0.0 to 1.0, default: 0.5)
  - Lower values: Brighter twilight (early evening)
  - Higher values: Darker twilight (late evening)

## How It Works

The twilight simulation process consists of four main steps:

1. **Sky Detection**: Uses HSV color space and position-based heuristics to identify sky regions
2. **Twilight Sky Coloring**: Applies a gradient of twilight colors (blue → purple → orange → pink) to the sky
3. **Image Darkening**: Reduces overall brightness to simulate twilight lighting
4. **Ambiance Addition**: Adds subtle blue-purple tint for atmospheric effect

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Examples

### Example 1: Default Settings
```bash
python twilight_simulator.py
```
- Twilight intensity: 0.6
- Darkness factor: 0.5
- Good for general-purpose twilight simulation

### Example 2: Dramatic Twilight
```bash
python twilight_simulator.py --intensity 0.8 --darkness 0.7
```
- Creates a more dramatic, late-evening twilight effect

### Example 3: Subtle Twilight
```bash
python twilight_simulator.py --intensity 0.4 --darkness 0.3
```
- Creates a subtle, early-evening twilight effect

## Output

Processed images are saved with the prefix `twilight_` followed by the original filename. For example:
- Input: `truck_001.jpg`
- Output: `twilight_truck_001.jpg`

## Project Structure

```
twilight_simulation/
├── twilight_simulator.py    # Main script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── input_files/             # Place input images here
│   └── .gitkeep
└── output_files/            # Processed images saved here
    └── .gitkeep
```

## Tips for Best Results

1. **Image Quality**: Works best with clear daytime images that have visible sky
2. **Sky Visibility**: Images with more sky area will show more dramatic twilight effects
3. **Parameter Tuning**: Experiment with different intensity and darkness values for your specific use case
4. **Industrial Sites**: Optimized for outdoor industrial/mining environments with vehicles

## Use Cases

- Data augmentation for computer vision models
- Training object detection systems for different lighting conditions
- Simulating rare twilight scenarios in industrial environments
- Creating diverse datasets for vehicle detection in mining/plant sites

## Troubleshooting

- **No sky detected**: If the script doesn't detect sky properly, try images with more visible sky area
- **Too dark/bright**: Adjust the `--darkness` parameter
- **Sky colors too intense**: Reduce the `--intensity` parameter
- **No images processed**: Ensure images are in supported formats and in the input directory

## License

This tool is part of the data augmentation project for industrial/mining site image processing.
