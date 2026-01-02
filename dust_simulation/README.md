# Dust Accumulation Data Augmentation

This script simulates realistic dust and dirt accumulation on camera lenses for industrial CCTV footage augmentation. Perfect for creating training datasets for mine, plant, and industrial site vehicle monitoring systems.

## Features

- **Multiple Dust Levels**: Light, Medium, Heavy, and Extreme dust accumulation
- **Realistic Effects**:
  - Random dust particles of varying sizes
  - Smudge/streak patterns (like wiped dust)
  - Edge accumulation (common in real scenarios)
  - Color tinting (brownish dust color)
  - Slight blur and contrast reduction
- **Batch Processing**: Process entire folders of images
- **Reproducible**: Optional random seed for consistent results
- **Flexible**: Support for multiple image formats (JPG, PNG, BMP)

## Requirements

```bash
pip install opencv-python numpy
```

## Usage

### Basic Usage

Process all images in a folder with default settings (light, medium, heavy dust):

```bash
python dust_augmentation.py --input ./input_images --output ./output_images
```

### Custom Dust Levels

Apply specific dust levels:

```bash
python dust_augmentation.py --input ./input_images --output ./output_images --levels light heavy extreme
```

### With Random Seed (for reproducibility)

```bash
python dust_augmentation.py --input ./input_images --output ./output_images --seed 42
```

### Custom File Extensions

```bash
python dust_augmentation.py --input ./input_images --output ./output_images --extensions .jpg .png
```

### Complete Example

```bash
python dust_augmentation.py \
    --input /home/deepika/Documents/data-augmentation/input \
    --output /home/deepika/Documents/data-augmentation/output \
    --levels light medium heavy \
    --seed 42 \
    --extensions .jpg .jpeg .png
```

## Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--input` | `-i` | Input folder containing images | Required |
| `--output` | `-o` | Output folder for augmented images | Required |
| `--levels` | `-l` | Dust levels to apply | `light medium heavy` |
| `--seed` | `-s` | Random seed for reproducibility | None |
| `--extensions` | `-e` | File extensions to process | `.jpg .jpeg .png .bmp` |

## Dust Levels

- **Light**: Minimal dust, slight visibility reduction (30% intensity, 20% coverage)
- **Medium**: Moderate dust accumulation (50% intensity, 40% coverage)
- **Heavy**: Significant dust, noticeable image degradation (70% intensity, 60% coverage)
- **Extreme**: Severe dust accumulation, heavily obscured view (90% intensity, 80% coverage)

## Output

For each input image, the script creates separate augmented versions for each dust level:

```
input_images/
  └── truck_001.jpg

output_images/
  ├── truck_001_dust_light.jpg
  ├── truck_001_dust_medium.jpg
  └── truck_001_dust_heavy.jpg
```

## Use Cases

Perfect for augmenting datasets for:
- Vehicle detection in mining sites
- Industrial plant monitoring
- CCTV footage analysis
- Robust computer vision models that handle real-world camera conditions
- Training models for vehicles: trucks, dumpers, tippers, LMVs, HMVs, graders, dozers, cars

## Programmatic Usage

You can also use the script as a Python module:

```python
from dust_augmentation import DustAugmentation

# Create augmentor
augmentor = DustAugmentation(seed=42)

# Process a single image
import cv2
image = cv2.imread('input.jpg')
dusty_image = augmentor.apply_dust_effect(image, dust_level='medium')
cv2.imwrite('output.jpg', dusty_image)

# Process entire folder
augmentor.process_folder(
    input_folder='./input',
    output_folder='./output',
    dust_levels=['light', 'medium', 'heavy']
)
```

### Custom Parameters

For fine-grained control:

```python
custom_params = {
    'dust_intensity': 0.6,
    'dust_coverage': 0.5,
    'smudge_intensity': 0.4,
    'edge_dust_intensity': 0.7,
    'color_tint': (140, 130, 110)  # BGR format
}

dusty_image = augmentor.apply_dust_effect(image, custom_params=custom_params)
```

## Technical Details

The script applies multiple layers of dust effects:

1. **Dust Particles**: Random circular spots with gaussian blur
2. **Smudge Patterns**: Streak-like patterns simulating wiped dust
3. **Edge Accumulation**: Gradient-based dust buildup around image edges
4. **Color Tinting**: Brownish overlay matching real dust color
5. **Image Degradation**: Slight blur and contrast reduction

All effects are combined and blended with the original image for realistic results.

## License

Free to use for data augmentation and research purposes.
