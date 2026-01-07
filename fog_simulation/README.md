# Fog Simulation for Data Augmentation

A realistic fog simulation tool for augmenting images of mining, industrial, and plant sites. The fog effect varies naturally with distance, creating authentic atmospheric perspective.

## Features

- **Depth-Based Fog Intensity**: Fog becomes denser with distance, mimicking real-world atmospheric scattering
- **Multiple Intensity Levels**: Light, medium, heavy, and variable fog options
- **Natural Turbulence**: Patchy fog distribution for realistic appearance
- **Atmospheric Perspective**: Automatic contrast and saturation reduction for distant objects
- **Batch Processing**: Process entire directories of images efficiently
- **Flexible Depth Estimation**: Multiple methods for depth map generation

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

Place your images in the `input_files` folder and run:

```bash
python fog_simulator.py
```

This will process all images with medium fog intensity and save results to `output_files`.

### Advanced Usage

#### Different Fog Intensities

```bash
# Light fog
python fog_simulator.py --intensity light

# Heavy fog
python fog_simulator.py --intensity heavy

# Variable fog (random intensity for each image)
python fog_simulator.py --intensity variable
```

#### Custom Directories

```bash
python fog_simulator.py -i /path/to/input -o /path/to/output
```

#### Depth Estimation Methods

```bash
# Gradient-based (simple, fast)
python fog_simulator.py --depth-method gradient

# Edge-based (detail-aware)
python fog_simulator.py --depth-method edge

# Hybrid (best quality, default)
python fog_simulator.py --depth-method hybrid
```

#### Custom Output Prefix

```bash
python fog_simulator.py --prefix foggy_
```

### Complete Example

```bash
python fog_simulator.py \
    --input-dir ./input_files \
    --output-dir ./output_files \
    --intensity heavy \
    --depth-method hybrid \
    --prefix heavy_fog_
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input-dir` | Input directory containing images | `input_files` |
| `-o, --output-dir` | Output directory for processed images | `output_files` |
| `--intensity` | Fog intensity: `light`, `medium`, `heavy`, `variable` | `medium` |
| `--depth-method` | Depth estimation: `gradient`, `edge`, `hybrid` | `hybrid` |
| `--prefix` | Prefix for output filenames | `fog_` |

## How It Works

### 1. Depth Map Generation

The script creates a synthetic depth map using one of three methods:

- **Gradient**: Simple top-to-bottom gradient (sky=far, ground=near)
- **Edge**: Edge detection to identify object boundaries
- **Hybrid**: Combines gradient, edges, and brightness information

### 2. Turbulence Addition

Natural fog isn't uniform. The script adds Perlin-like noise to create patchy, realistic fog distribution.

### 3. Atmospheric Perspective

Objects further away lose contrast and color saturation, just like in real foggy conditions.

### 4. Fog Layer Application

Uses an exponential fog model: `fog_amount = 1 - exp(-density * depth)`

This creates realistic fog falloff where nearby objects are clear and distant objects are heavily obscured.

## Fog Intensity Levels

| Intensity | Density | Best For |
|-----------|---------|----------|
| **Light** | 30% | Early morning mist, slight haze |
| **Medium** | 50% | Typical foggy conditions |
| **Heavy** | 70% | Dense fog, low visibility |
| **Variable** | Random | Diverse dataset with mixed conditions |

## Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- BMP (`.bmp`)
- TIFF (`.tiff`, `.tif`)

## Project Structure

```
fog_simulation/
├── fog_simulator.py      # Main script
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── input_files/         # Place input images here
└── output_files/        # Processed images saved here
```

## Examples

### Light Fog
Subtle atmospheric haze, maintains most image clarity.

### Medium Fog
Balanced fog effect, good for general augmentation.

### Heavy Fog
Dense fog, significantly reduces visibility of distant objects.

### Variable Fog
Each image gets random fog intensity, creating diverse training data.

## Tips for Best Results

1. **Use Hybrid Depth Method**: Provides the most realistic depth estimation
2. **Variable Intensity**: For training datasets, use variable intensity to create diverse examples
3. **Batch Processing**: Process all images at once for efficiency
4. **Original Preservation**: Keep original images; the script creates new files with prefixes

## Technical Details

### Fog Color

The fog color is automatically adjusted based on intensity:
- Light fog: Lighter grayish-white (220, 230, 235)
- Medium fog: Medium gray-white (200, 210, 220)
- Heavy fog: Heavier gray (180, 190, 200)
- Variable: Random within realistic range

### Performance

Processing time depends on:
- Image resolution
- Depth estimation method (gradient is fastest, hybrid is most accurate)
- System specifications

Typical processing: 0.5-2 seconds per image on modern hardware.

## Troubleshooting

### No images found
- Ensure images are in the `input_files` directory
- Check that images have supported extensions

### Poor fog quality
- Try the `hybrid` depth method for best results
- Adjust intensity level to match your needs

### Memory issues with large images
- Process images in smaller batches
- Resize images before processing if needed

## Integration with ML Pipelines

This tool is designed to integrate seamlessly with computer vision training pipelines:

```python
from fog_simulator import FogSimulator

# Create simulator
simulator = FogSimulator(fog_intensity='variable')

# Process single image
import cv2
image = cv2.imread('input.jpg')
fogged = simulator.simulate_fog(image)
cv2.imwrite('output.jpg', fogged)
```

## License

This tool is part of a data augmentation pipeline for mining/industrial site image processing.

## Author

Data Augmentation Pipeline - 2026

---

For questions or issues, please refer to the script documentation or modify parameters to suit your specific use case.
