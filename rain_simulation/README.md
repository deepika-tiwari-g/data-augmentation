# Rain Simulation for Data Augmentation

Realistic rain simulation for mining site images using OpenCV and NumPy. Applies multi-layer rain effects with atmospheric adjustments.

## Features

- **Multi-layer rain**: 3 depth layers (background/midground/foreground) for realism
- **Atmospheric effects**: Contrast reduction, desaturation, cool color shift
- **Surface wetness**: Gradient darkening on ground (no fake reflections)
- **Depth-based fog**: Thicker at top (distant objects), clearer at bottom

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place images in `input_files/`
2. Run the script:
   ```bash
   python realistic_rain_gen.py
   ```
3. Results saved in `output_files/` with `rain_` prefix

### Programmatic Usage

```python
from realistic_rain_gen import apply_rain
import cv2

img = cv2.imread("your_image.jpg")
result = apply_rain(img)
cv2.imwrite("output.jpg", result)
```

## Pipeline

```
Input → Atmosphere (contrast, color) → Wetness → Rain Streaks → Fog → Output
```

## Parameters

Edit values in `realistic_rain_gen.py` to adjust:

| Effect | Function | Key Parameter |
|--------|----------|---------------|
| Rain density | `_create_multi_layer_rain()` | `density` values |
| Fog intensity | `_apply_fog()` | `max_intensity` |
| Wetness | `_apply_wetness()` | darkening factor |
| Color temp | `_apply_atmosphere()` | blue/red channel offsets |

## Directory Structure

```
rain_simulation/
├── input_files/          # Source images
├── output_files/         # Processed images
├── realistic_rain_gen.py # Main script
├── requirements.txt      # Dependencies
└── README.md
```
