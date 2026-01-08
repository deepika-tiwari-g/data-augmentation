# Realistic Heavy Rainfall Simulator

A comprehensive data augmentation tool for simulating realistic heavy rainfall conditions at mining sites. This simulator creates cinematic weather effects including volumetric rain, wet surfaces, puddles, atmospheric fog, and water droplets.

## Features

### 🌧️ Volumetric Rain Streaks
- Variable opacity and motion blur
- Realistic rain streak angles and lengths
- High-density rainfall simulation
- Cinematic motion blur effects

### 💧 Wet Surface Effects
- High specularity on wet asphalt and soil
- Real-time reflections of vehicles
- Darkened wet surfaces
- Enhanced contrast for realistic wet look

### 🌊 Water Puddles
- Scattered puddle generation
- Subtle ripple patterns
- Realistic water reflections
- Smooth puddle edges with natural shapes

### 🌫️ Atmospheric Effects
- Depth-based hazy mist
- Volumetric fog in background
- Cool color temperature
- Desaturated tones for overcast conditions

### 💦 Water Droplets
- Droplets on vehicle surfaces
- Windshield water effects
- Lens distortion simulation
- Specular highlights

### 🎨 Color Grading
- Cool color temperature
- Desaturated tones
- Damp surface textures
- Overcast day atmosphere

## Installation

```bash
# Navigate to the rain_simulation directory
cd rain_simulation

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

Place your input images in the `input_files` folder and run:

```bash
python rain_simulator.py
```

This will process all images with default settings and save results to the `output` folder.

### Custom Parameters

```bash
python rain_simulator.py \
    --input_dir /path/to/input \
    --output_dir /path/to/output \
    --rain_intensity 0.8 \
    --puddle_density 0.6 \
    --wetness_level 0.7 \
    --atmospheric_depth 0.5 \
    --color_temperature 0.6
```

### Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `--input_dir` | path | `input_files` | Input directory containing images |
| `--output_dir` | path | `output` | Output directory for processed images |
| `--rain_intensity` | 0.0-1.0 | 0.8 | Intensity of rain streaks |
| `--puddle_density` | 0.0-1.0 | 0.6 | Density of water puddles |
| `--wetness_level` | 0.0-1.0 | 0.7 | Level of surface wetness/specularity |
| `--atmospheric_depth` | 0.0-1.0 | 0.5 | Strength of atmospheric fog/mist |
| `--color_temperature` | 0.0-1.0 | 0.6 | Cool color temperature (0=neutral, 1=very cool) |

## Examples

### Light Rain
```bash
python rain_simulator.py \
    --rain_intensity 0.4 \
    --puddle_density 0.3 \
    --wetness_level 0.4 \
    --atmospheric_depth 0.2
```

### Heavy Rainfall (Default)
```bash
python rain_simulator.py \
    --rain_intensity 0.8 \
    --puddle_density 0.6 \
    --wetness_level 0.7 \
    --atmospheric_depth 0.5
```

### Extreme Storm
```bash
python rain_simulator.py \
    --rain_intensity 1.0 \
    --puddle_density 0.9 \
    --wetness_level 0.9 \
    --atmospheric_depth 0.7 \
    --color_temperature 0.8
```

## Project Structure

```
rain_simulation/
├── rain_simulator.py      # Main simulation script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── input_files/          # Place input images here
└── output/               # Processed images saved here
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Technical Details

### Rain Simulation Pipeline

1. **Color Grading**: Apply cool color temperature and desaturation
2. **Wet Surface Effect**: Darken surfaces, add specularity and reflections
3. **Puddle Generation**: Create scattered puddles with ripple patterns
4. **Atmospheric Effects**: Add depth-based fog and volumetric mist
5. **Water Droplets**: Simulate droplets on vehicle surfaces
6. **Rain Streaks**: Generate volumetric rain with motion blur

### Algorithm Highlights

- **Volumetric Rain**: Uses random streak generation with motion blur kernels
- **Wet Surfaces**: Combines darkening, contrast enhancement, and specular highlights
- **Puddles**: Elliptical puddle shapes with Perlin-like ripple displacement
- **Atmospheric Fog**: Multi-scale noise for realistic volumetric mist
- **Water Droplets**: Lens distortion simulation with specular highlights
- **Color Grading**: HSV-based desaturation with RGB color temperature shift

## Performance

Processing time depends on image resolution and parameter settings:
- 1920×1080 image: ~2-4 seconds per image
- 3840×2160 image: ~8-12 seconds per image

## Use Cases

- **Mining Site Surveillance**: Augment CCTV footage for all-weather training
- **Vehicle Detection**: Improve model robustness in rainy conditions
- **Safety Systems**: Train models to detect vehicles in adverse weather
- **Dataset Balancing**: Generate rare weather condition samples

## Tips for Best Results

1. **Mining Sites**: Use higher `wetness_level` (0.7-0.9) for realistic wet soil
2. **Vehicle Focus**: Moderate `atmospheric_depth` (0.4-0.6) to maintain visibility
3. **Heavy Rain**: Combine high `rain_intensity` with high `puddle_density`
4. **Overcast Look**: Use `color_temperature` 0.6-0.8 for cool, desaturated tones
5. **Cinematic Effect**: Balance all parameters for photorealistic results

## Troubleshooting

**Issue**: Images too dark
- **Solution**: Reduce `wetness_level` or `atmospheric_depth`

**Issue**: Rain streaks not visible
- **Solution**: Increase `rain_intensity` to 0.9-1.0

**Issue**: Too many puddles
- **Solution**: Reduce `puddle_density` to 0.3-0.5

**Issue**: Colors too desaturated
- **Solution**: Reduce `color_temperature` to 0.3-0.5

## License

This tool is provided for data augmentation purposes in computer vision projects.

## Author

Created for mining site data augmentation projects focusing on vehicle detection in adverse weather conditions.
