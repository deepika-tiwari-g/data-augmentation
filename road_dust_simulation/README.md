# Road Dust Simulation - Data Augmentation

Realistic dust cloud simulation for mining/industrial site vehicle images. Generates volumetric dust wakes with particle physics for edge-case data augmentation.

## Features

✨ **Realistic Physics**
- Particle-based dust simulation with Brownian motion
- Tire-based emitter system (rear wheels)
- Wind drift and atmospheric dispersion
- Speed-dependent dust intensity

🎨 **Visual Realism**
- Volumetric clouds using Perlin noise
- Road color sampling for natural blending
- Alpha blending with density gradients
- Atmospheric haze effects

⚙️ **Flexible Configuration**
- Adjustable vehicle speed (0.5x - 2.0x)
- Multiple variant generation
- Batch processing support
- Command-line interface

## Installation

```bash
cd road_dust_simulation
pip install -r requirements.txt
```

## Quick Start

### 1. Add Your Images

Place your input images in the `input_files` folder:

```bash
cp /path/to/your/images/*.jpg input_files/
```

### 2. Run Processing

**Basic usage (default settings):**
```bash
python process_images.py
```

**Custom speed:**
```bash
python process_images.py --speed 1.5
```

**Generate multiple variants:**
```bash
python process_images.py --variants 3
```

**Full options:**
```bash
python process_images.py \
  --input-dir input_files \
  --output-dir output_images \
  --speed 1.0 \
  --variants 1
```

### 3. Check Results

Processed images will be saved in `output_images/` folder.

## Testing

Run the test script to see dust simulation with different speeds:

```bash
python test_dust_simulator.py
```

This generates synthetic test images showing:
- Slow speed (light dust)
- Normal speed (moderate dust)
- Fast speed (heavy dust)
- Very fast speed (very heavy dust)

## Usage as Python Module

```python
from dust_simulator import add_dust_wake
import cv2

# Load image
image = cv2.imread('vehicle.jpg')

# Define vehicle bounding box (x1, y1, x2, y2)
vehicle_bbox = (100, 200, 400, 500)

# Add dust wake
result = add_dust_wake(
    image,
    vehicle_bbox,
    vehicle_speed=1.5,  # 1.5x speed = more dust
    num_frames=20       # Simulation frames
)

# Save result
cv2.imwrite('dusty_vehicle.jpg', result)
```

## Parameters

### `add_dust_wake()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | np.ndarray | - | Input image (BGR format) |
| `vehicle_bbox` | tuple | - | (x1, y1, x2, y2) bounding box |
| `vehicle_speed` | float | 1.0 | Speed multiplier (0.5=slow, 2.0=fast) |
| `num_frames` | int | 20 | Simulation frames for dust generation |

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | input_files | Input directory |
| `--output-dir` | output_images | Output directory |
| `--speed` | 1.0 | Vehicle speed multiplier |
| `--variants` | 1 | Number of variants per image |

## How It Works

### 1. Particle Physics Engine

Each dust particle has:
- Position (x, y)
- Velocity (vx, vy) with upward drift
- Age and lifespan
- Size and opacity

Particles update each frame with:
- **Brownian motion**: Random walk for natural turbulence
- **Wind drift**: Directional movement
- **Gravity/buoyancy**: Upward movement simulating dust rising
- **Fade out**: Opacity decreases with age

### 2. Volumetric Clouds

Instead of simple circles, dust uses:
- **Perlin noise**: 3-octave noise for organic cloud shapes
- **Gaussian blobs**: Smooth falloff from particle centers
- **Additive blending**: Multiple particles create dense regions

### 3. Tire Emitters

Dust generates from rear wheel positions:
- Calculated from vehicle bounding box
- Bottom corners (20% inset from edges)
- Continuous emission over simulation frames

### 4. Visual Blending

- **Road color sampling**: Extracts average color below vehicle
- **Alpha compositing**: Smooth transparency blending
- **Density gradient**: Thicker at ground, thinner at top
- **Atmospheric haze**: Subtle fog effect in dust regions

## Project Structure

```
road_dust_simulation/
├── dust_simulator.py       # Core simulation engine
├── process_images.py       # Batch processing script
├── test_dust_simulator.py  # Testing utilities
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── input_files/           # Place input images here
└── output_images/         # Processed images saved here
```

## Vehicle Types Supported

Designed for mining/industrial vehicles:
- Trucks
- Dumpers
- Tippers
- Graders
- Dozers
- LMV/HMV
- Cars

## Integration with Object Detection

For automatic vehicle detection, integrate with your existing detection model:

```python
# Example with YOLO or similar detector
detections = your_detector.detect(image)

for detection in detections:
    if detection.class_name in ['truck', 'dumper', 'vehicle']:
        bbox = detection.bbox  # (x1, y1, x2, y2)
        dusty_image = add_dust_wake(image, bbox, vehicle_speed=1.2)
```

## Tips for Best Results

1. **Speed Selection**:
   - 0.5-0.7: Light dust for slow-moving vehicles
   - 1.0-1.5: Normal dust for typical speeds
   - 1.5-2.0: Heavy dust for fast-moving vehicles

2. **Multiple Variants**:
   - Use `--variants 3` to generate low/medium/high dust versions
   - Increases dataset diversity

3. **Bounding Boxes**:
   - Accurate bounding boxes produce better results
   - Ensure boxes include the full vehicle
   - Rear wheels should be at bottom of box

4. **Image Quality**:
   - Works best with clear road surfaces
   - Handles various lighting conditions
   - Maintains original image quality

## Troubleshooting

**White/washed out images:**
- Check that input images are valid
- Ensure proper color space (BGR for OpenCV)

**No dust visible:**
- Increase `vehicle_speed` parameter
- Check vehicle bounding box is correct
- Verify `num_frames` is sufficient (try 30-40)

**Unrealistic colors:**
- Dust color is sampled from road
- Ensure road is visible below vehicle
- Adjust bounding box if needed

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Pillow
- noise (Perlin noise library)

## License

This script is provided for data augmentation purposes in computer vision projects.

## Related Projects

Check out other augmentation scripts in the parent directory:
- `fog_simulation/` - Fog and mist effects
- `rain_simulation/` - Rainfall simulation
- `night_simulation/` - Low-light conditions

## Support

For issues or questions, check:
1. Test script output: `python test_dust_simulator.py`
2. Verify dependencies: `pip install -r requirements.txt`
3. Check input image format and paths
