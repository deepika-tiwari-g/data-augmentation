# Night Simulation for Data Augmentation

This tool converts day images to realistic night images for data augmentation in computer vision projects, specifically designed for mining, plant, and industrial site surveillance footage.

## Features

✨ **Pitch-Black Darkness**: Extreme gamma correction (3.5) and deep darkening for realistic midnight conditions  
🚗 **Bright LED Headlights**: Pure white LED lights with high intensity and realistic beam patterns  
🌟 **Volumetric Light Cones**: Long, realistic light beams casting forward from headlights and downward from street lights  
💡 **Sodium-Vapor Street Lights**: Warm orange-yellow industrial lighting with authentic glow  
💧 **Ground Reflections**: Wet surface reflections for damp industrial environments  
✨ **Lens Flare Effects**: Realistic camera lens flare around bright light sources  
🎨 **High Contrast**: Deep shadows with bright light sources for dramatic industrial scenes  
⚙️ **Fully Configurable**: Control every aspect via JSON config file or command-line arguments

## Installation

### Requirements
- Python 3.7+
- OpenCV (cv2)
- NumPy

### Install Dependencies

```bash
pip install opencv-python numpy
```

## Usage

### Basic Usage

1. Place your day images in the `input_files` folder
2. Run the script:

```bash
python night_simulator.py
```

3. Find realistic midnight images in the `output_files` folder

### Advanced Usage

#### Custom Input/Output Directories

```bash
python night_simulator.py -i /path/to/input -o /path/to/output
```

#### Use Custom Configuration

```bash
python night_simulator.py -c custom_config.json
```

#### Disable Specific Effects

```bash
# Disable vehicle detection
python night_simulator.py --no-detect-vehicles

# Disable volumetric light cones
python night_simulator.py --no-volumetric

# Disable ground reflections
python night_simulator.py --no-reflections

# Disable lens flare
python night_simulator.py --no-lens-flare

# Combine multiple options
python night_simulator.py --no-volumetric --no-reflections
```

#### Quiet Mode (No Progress Messages)

```bash
python night_simulator.py -q
```

## Configuration

Edit `config.json` to customize the simulation parameters:

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `darkening_factor` | Overall brightness reduction | 0.1 - 0.25 | 0.15 |
| `gamma` | Gamma correction for deep blacks | 2.5 - 4.0 | 3.5 |
| `headlight_brightness` | LED headlight intensity | 1.0 - 2.0 | 1.5 |
| `streetlight_brightness` | Sodium-vapor light intensity | 0.8 - 1.5 | 1.2 |
| `headlight_enabled` | Enable vehicle headlights | true/false | true |
| `streetlight_enabled` | Enable street lights | true/false | true |
| `volumetric_enabled` | Enable volumetric light cones | true/false | true |
| `ground_reflection_enabled` | Enable ground reflections | true/false | true |
| `lens_flare_enabled` | Enable lens flare effects | true/false | true |
| `reflection_strength` | Ground reflection intensity | 0.2 - 0.6 | 0.4 |
| `detect_vehicles` | Auto-detect vehicles | true/false | true |

### Example Custom Configurations

#### Extremely Dark Night (Minimal Lighting)

```json
{
  "darkening_factor": 0.12,
  "gamma": 4.0,
  "headlight_brightness": 1.8,
  "streetlight_enabled": false,
  "volumetric_enabled": true
}
```

#### Well-Lit Industrial Site

```json
{
  "darkening_factor": 0.18,
  "gamma": 3.2,
  "streetlight_brightness": 1.4,
  "headlight_brightness": 1.3,
  "reflection_strength": 0.5
}
```

#### Dry Surface (No Reflections)

```json
{
  "darkening_factor": 0.15,
  "ground_reflection_enabled": false,
  "lens_flare_enabled": false
}
```

## How It Works

### 1. Pitch-Black Darkening
- Applies aggressive gamma correction (γ=3.5) for deep blacks
- Reduces brightness to 15% of original
- Increases contrast in remaining visible areas
- Desaturates colors for night-time appearance
- Adds subtle cool tint

### 2. Vehicle Detection
- Enhanced edge detection with bilateral filtering
- Multi-threshold Canny edge detection
- Contour analysis with size and aspect ratio filtering
- Supports all vehicle types: trucks, dumpers, tippers, LMVs, HMVs, graders, dozers, cars

### 3. LED Headlight Simulation
- Pure white LED color (255, 255, 255)
- Bright core with multi-layer glow
- Volumetric light cones extending forward and downward
- Lens flare effects with hexagonal patterns and starburst
- Realistic beam length (60% of image height)

### 4. Sodium-Vapor Street Lights
- Warm orange-yellow color (BGR: 100, 180, 255)
- Multiple glow layers (bright core + halo)
- Volumetric downward cones for area illumination
- Positioned in upper portion of frame

### 5. Ground Reflections
- Simulates wet/damp industrial surfaces
- Vertical flip of upper portion (lights)
- Gradient-based blending (stronger near lights, fading down)
- Gaussian blur for realistic reflection

### 6. Lens Flare
- Multiple concentric rings around bright lights
- Starburst pattern radiating outward
- Intensity decreases with distance from source

## Tips for Best Results

1. **Pitch-Black Nights**: Use `darkening_factor` 0.12-0.15 and `gamma` 3.5-4.0
2. **Lighter Nights**: Use `darkening_factor` 0.18-0.22 and `gamma` 3.0-3.2
3. **Brighter Headlights**: Increase `headlight_brightness` to 1.7-2.0
4. **Subtle Lighting**: Decrease brightness values to 1.0-1.2
5. **Dry Surfaces**: Set `ground_reflection_enabled` to false
6. **Wet Surfaces**: Increase `reflection_strength` to 0.5-0.6
7. **No Street Lights**: Set `streetlight_enabled` to false for remote sites
8. **Reduce Lens Flare**: Set `lens_flare_enabled` to false for cleaner look

## Troubleshooting

**Images too dark?**
- Increase `darkening_factor` to 0.18-0.22
- Decrease `gamma` to 3.0-3.2

**Headlights not appearing?**
- Ensure vehicles are clearly visible in original image
- Check that `headlight_enabled` is true
- Verify `detect_vehicles` is enabled

**Lights too bright?**
- Decrease `headlight_brightness` to 1.2-1.3
- Decrease `streetlight_brightness` to 0.9-1.0

**Lights too dim?**
- Increase brightness parameters to 1.6-1.8
- Check that volumetric effects are enabled

**Reflections too strong?**
- Decrease `reflection_strength` to 0.2-0.3
- Disable with `ground_reflection_enabled: false`

**Lens flare too intense?**
- Disable with `lens_flare_enabled: false`
- Reduce light brightness values

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Project Structure

```
night_simulation/
├── night_simulator.py    # Main script (600+ lines)
├── config.json          # Configuration file
├── README.md           # This file
├── .gitignore          # Git ignore rules
├── input_files/        # Place input images here
└── output_files/       # Midnight images saved here
```

## Use Cases

- **Training object detection models** for midnight/night-time scenarios
- **Augmenting datasets** for mining/industrial site surveillance
- **Simulating extreme low-light conditions** for vehicle tracking
- **Creating diverse training data** when midnight footage is unavailable
- **Testing computer vision models** under challenging lighting conditions
- **Security camera simulation** for industrial environments

## Technical Details

### Image Processing Pipeline
1. Load BGR image → Convert to float32
2. Apply pitch-black darkening (gamma 3.5, factor 0.15)
3. Increase contrast and reduce saturation
4. Detect vehicles using enhanced edge detection
5. Add sodium-vapor street lights with volumetric cones
6. Add bright LED headlights with volumetric beams
7. Apply lens flare effects around bright lights
8. Add ground reflections for wet surface simulation
9. Save final result

### Performance
- Processing time: ~1-3 seconds per image (depends on resolution)
- Memory usage: Proportional to image size
- Recommended max resolution: 4K (3840×2160)

## License

This script is provided as-is for data augmentation purposes.

## Author

Data Augmentation Pipeline - 2026

## Version History

- **v2.0** (2026-01-06): Enhanced with volumetric lighting, lens flare, ground reflections
- **v1.0** (2026-01-03): Initial release with basic night simulation
