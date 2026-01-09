# Night Simulation for Data Augmentation

This tool converts day images to realistic twilight/evening scenes for data augmentation in computer vision projects, specifically designed for mining, plant, and industrial site surveillance footage.

## Features

🌆 **Twilight Simulation**: Moderate darkening with gamma correction for realistic evening/dusk conditions  
🚗 **Intelligent Headlights**: White LED headlights with automatic placement based on vehicle position and direction  
🎯 **Custom YOLO Support**: Integrates with your custom-trained YOLO models for accurate vehicle detection  
💡 **Right-Side Street Lights**: Golden-orange industrial street lights positioned on the right side with wide spread  
🔦 **Directional Light Beams**: Smart beam projection based on vehicle orientation (left, right, or facing camera)  
🌟 **Volumetric Light Cones**: Realistic light beams with falloff and cone expansion  
⚙️ **Fully Configurable**: Control darkness, brightness, and spread via JSON config or command-line arguments

## Installation

### Requirements
- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy
- Ultralytics YOLO (optional, for vehicle detection)

### Install Dependencies

```bash
pip install opencv-python numpy
```

### Optional: Install YOLO for Vehicle Detection

```bash
pip install ultralytics
```

> **Note**: If YOLO is not installed, the script will use a fallback edge-based vehicle detection method.

## Usage

### Basic Usage

1. Place your day images in the `input_files` folder
2. Run the script:

```bash
python night_sim_final.py
```

3. Find twilight/evening images in the `output_files` folder with `night_` prefix

### Advanced Usage

#### Custom Input/Output Directories

```bash
python night_sim_final.py -i /path/to/input -o /path/to/output
```

#### Use Custom YOLO Model

```bash
python night_sim_final.py -m /path/to/your_model.pt
```

Default model path: `site_2_yolov11n_v1+v2.pt`

#### Use Custom Configuration

```bash
python night_sim_final.py -c custom_config.json
```

#### Combined Options

```bash
python night_sim_final.py -i ./images -o ./output -m custom_yolo.pt -c config.json
```

## Configuration

Create a `config.json` file to customize simulation parameters:

```json
{
  "night_darkness": 0.40,
  "headlight_brightness": 0.35,
  "streetlight_brightness": 1.15,
  "streetlight_spread": 100
}
```

### Configuration Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `night_darkness` | Overall scene brightness (higher = brighter) | 0.2 - 0.6 | 0.40 |
| `headlight_brightness` | Vehicle headlight intensity | 0.2 - 1.0 | 0.35 |
| `streetlight_brightness` | Street light intensity | 0.8 - 1.5 | 1.15 |
| `streetlight_spread` | Street light cone spread radius | 50 - 150 | 100 |

### Example Configurations

#### Darker Evening (Late Dusk)

```json
{
  "night_darkness": 0.30,
  "headlight_brightness": 0.50,
  "streetlight_brightness": 1.30,
  "streetlight_spread": 120
}
```

#### Lighter Twilight (Early Evening)

```json
{
  "night_darkness": 0.50,
  "headlight_brightness": 0.25,
  "streetlight_brightness": 1.00,
  "streetlight_spread": 80
}
```

## How It Works

### 1. Vehicle Detection

**With Custom YOLO Model**:
- Loads your custom-trained YOLO model (e.g., `site_2_yolov11n_v1+v2.pt`)
- Detects vehicles with confidence threshold > 0.25
- Supports custom vehicle classes: vehicle, truck, dumper, tipper, etc.
- Falls back to YOLOv8n if custom model not found

**Without YOLO (Fallback)**:
- Uses edge detection with Canny algorithm
- Analyzes contours for vehicle-like shapes
- Filters by area (1-40% of image) and aspect ratio (0.5-4.0)

### 2. Twilight Darkening

- Applies moderate gamma correction (**γ=1.8**) for realistic evening light
- Reduces brightness to **40%** of original (configurable)
- Adds subtle **blue tint** for twilight atmosphere
- **Gradient sky darkening** (darker at top, lighter at bottom)
- Preserves visibility while creating evening ambiance

### 3. Intelligent Headlight Placement

Automatically determines headlight position and beam direction based on vehicle location:

- **Left side vehicles** (< 35% width): Headlights on **right edge**, beaming **right**
- **Right side vehicles** (> 65% width): Headlights on **left edge**, beaming **left**
- **Center vehicles** (35-65% width): **Both headlights visible**, beaming **forward**

Each headlight includes:
- Bright white LED core (small, intense)
- Soft glow halo (medium spread)
- Volumetric beam cone (45% of image height)
- Realistic falloff and expansion

### 4. Right-Side Street Lights

- **2 street lights** positioned at **75%** and **90%** of image width (right side only)
- **Golden-orange color** (BGR: 40, 150, 240) for industrial sodium-vapor appearance
- Large bright core with extensive glow
- Wide downward cone (65% of image height, 18% width spread)
- High intensity for prominent illumination effect

## Custom YOLO Model Integration

The script is designed to work with your custom-trained YOLO models:

### Vehicle Class Mapping

Update the `VEHICLE_CLASSES` dictionary in the script to match your model's classes:

```python
VEHICLE_CLASSES = {
    0: 'vehicle', 
    1: 'truck', 
    2: 'dumper', 
    3: 'tipper'
}
```

### Model Requirements

- **Format**: PyTorch (.pt) YOLO model
- **Classes**: Any vehicle types (truck, dumper, tipper, excavator, etc.)
- **Confidence**: Detections above 0.25 confidence are used

### Fallback Behavior

- If custom model not found → tries YOLOv8n
- If YOLO unavailable → uses edge detection
- Script always produces output regardless of detection method

## Supported Image Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)

## Project Structure

```
night_simulation/
├── night_sim_final.py           # Main script (346 lines)
├── site_2_yolov11n_v1+v2.pt    # Your custom YOLO model (optional)
├── config.json                  # Configuration file (optional)
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── input_files/                # Place input images here
└── output_files/               # Evening images saved here (auto-created)
```

## Use Cases

- **Training object detection models** for twilight/evening scenarios
- **Augmenting datasets** for mining/industrial site surveillance
- **Simulating low-light conditions** when evening footage is unavailable
- **Testing computer vision models** under reduced lighting
- **Creating diverse training data** for all-day operation scenarios
- **Edge case simulation** for autonomous vehicle systems

## Technical Details

### Image Processing Pipeline

1. **Load image** → Convert BGR to float32
2. **Detect vehicles** using custom YOLO or fallback edge detection
3. **Apply twilight darkening** (gamma 1.8, 40% brightness, blue tint, sky gradient)
4. **Add street lights** on right side (2 lights at 75% and 90% width)
5. **Determine headlight placement** based on vehicle position in frame
6. **Add headlights** with directional beams (left, right, or forward)
7. **Save result** with `night_` prefix

### Performance

- **Processing time**: ~1-3 seconds per image (depends on resolution and detection method)
- **YOLO detection**: Faster and more accurate
- **Edge detection**: Slower but works without dependencies
- **Memory usage**: Proportional to image size
- **Recommended resolution**: Up to 4K (3840×2160)

## Troubleshooting

### Images too dark?
- Increase `night_darkness` to 0.45-0.55
- Increase `streetlight_brightness` to 1.3-1.5

### Images too bright (not evening-like)?
- Decrease `night_darkness` to 0.30-0.35
- Consider darker twilight settings

### Headlights not appearing?
- Check if vehicles are detected (console output shows detections)
- Ensure YOLO model is loaded or edge detection is working
- Verify vehicles are clearly visible in original image

### Headlights in wrong position?
- Check vehicle position in frame (left/center/right logic)
- Adjust YOLO confidence threshold if needed
- Custom model may need retraining if bounding boxes are inaccurate

### Street lights too bright?
- Decrease `streetlight_brightness` to 0.9-1.0
- Reduce `streetlight_spread` to 70-80

### Street lights too dim?
- Increase `streetlight_brightness` to 1.3-1.5
- Increase `streetlight_spread` to 120-150

### YOLO model not loading?
- Check model path is correct (default: `site_2_yolov11n_v1+v2.pt`)
- Verify `ultralytics` is installed: `pip install ultralytics`
- Script will use fallback detection if model fails

## Tips for Best Results

1. **Use clear day images** with visible vehicles for best headlight placement
2. **Custom YOLO model** provides much better results than edge detection
3. **Train your YOLO model** on vehicles from your specific site for optimal accuracy
4. **Adjust `night_darkness`** based on desired time: 0.30-0.35 for late dusk, 0.45-0.50 for early evening
5. **Balance headlight brightness**: Too bright hides vehicles, too dim looks unrealistic
6. **Street light spread**: Wider spread (>100) creates more dramatic lighting

## Known Limitations

- Street lights always appear on right side only (by design for realism)
- Headlight direction determined by position, not actual vehicle orientation
- No ground reflections or lens flare effects (simplified version)
- Edge detection fallback is less accurate than YOLO

## License

This script is provided as-is for data augmentation purposes.

## Author

Data Augmentation Pipeline - 2026

## Version History

- **v3.0** (2026-01-09): Final simplified version with custom YOLO, twilight simulation, intelligent headlights
- **v2.0** (2026-01-06): Enhanced with volumetric lighting, lens flare, ground reflections
- **v1.0** (2026-01-03): Initial release with basic night simulation
