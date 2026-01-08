# Semantic-Aware Rain Simulation - New Methods

## These methods should be added to rain_simulator.py before the simulate_rain() method (around line 646)

```python
def apply_vehicle_effects(self, image: np.ndarray, vehicle_mask: np.ndarray) -> np.ndarray:
    """
    Apply vehicle-specific rain effects (water droplets, streaking)
    
    Args:
        image: Input image (uint8)
        vehicle_mask: Vehicle region mask [0.0-1.0]
        
    Returns:
        Image with vehicle effects (uint8)
    """
    if vehicle_mask.max() < 0.01:  # No vehicles detected
        return image
    
    h, w = image.shape[:2]
    img_float = image.astype(np.float32) / 255.0
    
    # Moderate darkening for wet vehicle surfaces (less than road)
    darkening = 0.20 * vehicle_mask
    darkening_3ch = np.stack([darkening] * 3, axis=-1)
    img_float = img_float * (1.0 - darkening_3ch)
    
    # Add water droplets concentrated on vehicle areas
    droplet_layer = self._add_vehicle_droplets(image, vehicle_mask)
    droplet_float = droplet_layer.astype(np.float32) / 255.0
    
    # Blend droplets with vehicle areas
    vehicle_mask_3ch = np.stack([vehicle_mask] * 3, axis=-1)
    img_float = img_float * (1 - vehicle_mask_3ch * 0.3) + droplet_float * vehicle_mask_3ch * 0.3
    
    img_float = np.clip(img_float, 0.0, 1.0)
    return (img_float * 255).astype(np.uint8)

def _add_vehicle_droplets(self, image: np.ndarray, vehicle_mask: np.ndarray) -> np.ndarray:
    """
    Add concentrated water droplets on vehicle surfaces
    
    Args:
        image: Input image
        vehicle_mask: Vehicle region mask
        
    Returns:
        Image with droplets
    """
    h, w = image.shape[:2]
    droplet_layer = image.copy()
    
    # More droplets on vehicles
    num_droplets = int(vehicle_mask.sum() / 100)  # Scale with vehicle area
    
    for _ in range(num_droplets):
        # Random position weighted by vehicle mask
        y, x = np.where(vehicle_mask > 0.3)
        if len(y) == 0:
            continue
        idx = random.randint(0, len(y) - 1)
        pos_x, pos_y = x[idx], y[idx]
        
        radius = random.randint(3, 10)
        
        # Add droplet with highlight
        brightness = random.randint(180, 255)
        cv2.circle(droplet_layer, (pos_x, pos_y), max(1, radius // 3),
                  (brightness, brightness, brightness), -1, cv2.LINE_AA)
    
    return droplet_layer

def apply_vegetation_effects(self, image: np.ndarray, vegetation_mask: np.ndarray) -> np.ndarray:
    """
    Apply vegetation-specific effects (wet leaves, no puddles)
    
    Args:
        image: Input image (uint8)
        vegetation_mask: Vegetation region mask [0.0-1.0]
        
    Returns:
        Image with vegetation effects (uint8)
    """
    if vegetation_mask.max() < 0.01:  # No vegetation detected
        return image
    
    img_float = image.astype(np.float32) / 255.0
    
    # Moderate darkening for wet leaves
    darkening = 0.25 * vegetation_mask
    darkening_3ch = np.stack([darkening] * 3, axis=-1)
    img_float = img_float * (1.0 - darkening_3ch)
    
    # Subtle specular highlights on wet leaves
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    _, bright_mask = cv2.threshold((gray * 255).astype(np.uint8), 140, 255, cv2.THRESH_BINARY)
    bright_mask = cv2.GaussianBlur(bright_mask, (7, 7), 0).astype(np.float32) / 255.0
    
    # Apply specular only in vegetation areas
    specular = np.stack([bright_mask] * 3, axis=-1) * vegetation_mask.reshape(vegetation_mask.shape + (1,)) * 0.1
    img_float = np.clip(img_float + specular, 0.0, 1.0)
    
    return (img_float * 255).astype(np.uint8)
```

## Modified simulate_rain() method - replace the existing one

```python
def simulate_rain(self, image: np.ndarray) -> np.ndarray:
    """
    Apply complete rain simulation to image (semantic-aware or uniform)
    
    Args:
        image: Input image (uint8)
        
    Returns:
        Rain-augmented image (uint8)
    """
    if not self.semantic_aware:
        # Use original uniform approach
        return self._simulate_rain_uniform(image)
    
    # SEMANTIC-AWARE APPROACH
    # Step 1: Detect regions
    regions = self.region_detector.detect_regions(image)
    
    # Step 2: Apply color grading globally
    result = self.apply_color_grading(image)
    
    # Step 3: Apply region-specific wet surface effects
    result = self._create_wet_surface_semantic(result, regions)
    
    # Step 4: Apply region-specific puddles (road only)
    result = self._generate_puddles_semantic(result, regions['road'])
    
    # Step 5: Apply vehicle-specific effects
    result = self.apply_vehicle_effects(result, regions['vehicles'])
    
    # Step 6: Apply vegetation effects
    result = self.apply_vegetation_effects(result, regions['vegetation'])
    
    # Step 7: Apply region-aware atmospheric effects
    result = self._add_atmospheric_effects_semantic(result, regions)
    
    # Step 8: Apply rain streaks (global but region-aware opacity)
    result = self._apply_rain_streaks_semantic(result, regions)
    
    return result

def _simulate_rain_uniform(self, image: np.ndarray) -> np.ndarray:
    """
    Original uniform rain simulation (fallback when semantic_aware=False)
    """
    # Step 1: Apply color grading first
    result = self.apply_color_grading(image)
    
    # Step 2: Create wet surface effect
    result = self.create_wet_surface_effect(result)
    
    # Step 3: Generate and apply puddles
    puddle_mask, ripple_layer = self.generate_puddles(result)
    
    # Blend puddles with reflections
    puddle_mask_3ch = np.stack([puddle_mask] * 3, axis=-1)
    
    # Convert to float for blending
    result_float = result.astype(np.float32) / 255.0
    ripple_float = ripple_layer.astype(np.float32) / 255.0
    
    # Puddles show ripple distortion
    result_float = (
        result_float * (1 - puddle_mask_3ch * 0.4) + 
        ripple_float * puddle_mask_3ch * 0.4
    )
    result_float = np.clip(result_float, 0.0, 1.0)
    result = (result_float * 255).astype(np.uint8)
    
    # Step 4: Add atmospheric effects
    result = self.add_atmospheric_effects(result)
    
    # Step 5: Add water droplets
    result = self.add_water_droplets(result)
    
    # Step 6: Generate and blend rain streaks
    rain_mask = self.generate_rain_streaks(result)
    
    # Convert result to float for blending
    result_float = result.astype(np.float32) / 255.0
    
    # Rain color (light grey-white)
    rain_color = np.array([0.85, 0.85, 0.85], dtype=np.float32)
    
    # Apply rain using alpha blending
    rain_mask_3ch = np.stack([rain_mask] * 3, axis=-1)
    result_float = (
        result_float * (1.0 - rain_mask_3ch) + 
        rain_color * rain_mask_3ch
    )
    
    result_float = np.clip(result_float, 0.0, 1.0)
    return (result_float * 255).astype(np.uint8)

def _create_wet_surface_semantic(self, image: np.ndarray, regions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Apply region-specific wet surface effects
    """
    img_float = image.astype(np.float32) / 255.0
    
    # Road: Strong darkening (45%)
    road_darkening = regions['road'] * 0.45
    
    # Vehicles: Moderate darkening (20%)
    vehicle_darkening = regions['vehicles'] * 0.20
    
    # Vegetation: Moderate darkening (25%)
    veg_darkening = regions['vegetation'] * 0.25
    
    # Sky: No darkening
    # Combine all darkenings
    total_darkening = road_darkening + vehicle_darkening + veg_darkening
    total_darkening_3ch = np.stack([total_darkening] * 3, axis=-1)
    
    img_float = img_float * (1.0 - total_darkening_3ch)
    
    # Enhanced contrast for road areas
    road_mask_3ch = np.stack([regions['road']] * 3, axis=-1)
    contrast = 1.5
    road_contrast = (img_float - 0.5) * contrast + 0.5
    img_float = img_float * (1 - road_mask_3ch) + road_contrast * road_mask_3ch
    
    img_float = np.clip(img_float, 0.0, 1.0)
    
    # Specular highlights (stronger on road)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    _, bright_mask = cv2.threshold((gray * 255).astype(np.uint8), 120, 255, cv2.THRESH_BINARY)
    bright_mask = cv2.GaussianBlur(bright_mask, (11, 11), 0).astype(np.float32) / 255.0
    
    # Road gets stronger specular
    specular_strength = regions['road'] * 0.35 + regions['vehicles'] * 0.15
    specular = np.stack([bright_mask] * 3, axis=-1) * specular_strength.reshape(specular_strength.shape + (1,))
    
    img_float = np.clip(img_float + specular, 0.0, 1.0)
    
    # Reflections on road only
    h, w = image.shape[:2]
    reflection_height = h // 2
    reflection = np.flipud(img_float[:reflection_height, :]) * 0.7
    
    fade = np.linspace(0.25, 0, reflection_height)
    fade = np.tile(fade.reshape(-1, 1, 1), (1, w, 3))
    
    # Apply reflection only to road areas in lower half
    blend_start = h - reflection_height
    road_lower = regions['road'][blend_start:, :]
    road_lower_3ch = np.stack([road_lower] * 3, axis=-1)
    
    img_float[blend_start:, :] = np.clip(
        img_float[blend_start:, :] * (1 - fade * road_lower_3ch) + reflection * fade * road_lower_3ch,
        0.0, 1.0
    )
    
    return (img_float * 255).astype(np.uint8)

def _generate_puddles_semantic(self, image: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """
    Generate puddles only on road areas
    """
    if road_mask.max() < 0.01:
        return image
    
    h, w = image.shape[:2]
    puddle_mask = np.zeros((h, w), dtype=np.float32)
    
    # More puddles, but only on roads
    num_puddles = int(self.puddle_density * 20)
    
    for _ in range(num_puddles):
        # Sample position from road areas
        y_coords, x_coords = np.where(road_mask > 0.3)
        if len(y_coords) == 0:
            continue
        
        idx = random.randint(0, len(y_coords) - 1)
        center_x, center_y = x_coords[idx], y_coords[idx]
        
        # Random puddle size
        radius_x = random.randint(40, 120)
        radius_y = random.randint(25, 70)
        
        # Create elliptical puddle
        y_grid, x_grid = np.ogrid[:h, :w]
        ellipse = (((x_grid - center_x) / radius_x) ** 2 + 
                  ((y_grid - center_y) / radius_y) ** 2 <= 1)
        
        puddle_mask[ellipse] = np.maximum(puddle_mask[ellipse], random.uniform(0.6, 1.0))
    
    # Constrain puddles to road areas
    puddle_mask = puddle_mask * road_mask
    puddle_mask = cv2.GaussianBlur(puddle_mask, (31, 31), 0)
    
    # Create ripples
    ripple_layer = self._create_ripples(image, puddle_mask)
    
    # Blend
    puddle_mask_3ch = np.stack([puddle_mask] * 3, axis=-1)
    result_float = image.astype(np.float32) / 255.0
    ripple_float = ripple_layer.astype(np.float32) / 255.0
    
    result_float = (
        result_float * (1 - puddle_mask_3ch * 0.5) + 
        ripple_float * puddle_mask_3ch * 0.5
    )
    
    return (np.clip(result_float, 0.0, 1.0) * 255).astype(np.uint8)

def _add_atmospheric_effects_semantic(self, image: np.ndarray, regions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Apply atmospheric effects stronger in sky areas
    """
    h, w = image.shape[:2]
    img_float = image.astype(np.float32) / 255.0
    
    # Depth gradient (stronger at top)
    depth_gradient = np.linspace(self.atmospheric_depth, 0, h)
    depth_gradient = np.tile(depth_gradient.reshape(-1, 1, 1), (1, w, 3))
    
    # Enhance fog in sky areas
    sky_boost = regions['sky'].reshape(regions['sky'].shape + (1,)) * 0.3
    depth_gradient = np.clip(depth_gradient + sky_boost, 0.0, 1.0)
    
    fog_color = np.array([0.71, 0.73, 0.71], dtype=np.float32)
    
    img_float = img_float * (1 - depth_gradient) + fog_color * depth_gradient
    img_float = np.clip(img_float, 0.0, 1.0)
    
    # Volumetric mist
    mist = self._generate_mist(h, w)
    mist_strength = self.atmospheric_depth * 0.2
    
    img_float = (
        img_float * (1 - mist * mist_strength) + 
        fog_color * mist * mist_strength
    )
    
    return (np.clip(img_float, 0.0, 1.0) * 255).astype(np.uint8)

def _apply_rain_streaks_semantic(self, image: np.ndarray, regions: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Apply rain streaks with region-aware opacity
    """
    rain_mask = self.generate_rain_streaks(image)
    
    # Reduce rain opacity in vehicle areas (blocked by vehicles)
    rain_mask = rain_mask * (1.0 - regions['vehicles'] * 0.4)
    
    # Convert to float
    result_float = image.astype(np.float32) / 255.0
    rain_color = np.array([0.85, 0.85, 0.85], dtype=np.float32)
    
    # Apply rain
    rain_mask_3ch = np.stack([rain_mask] * 3, axis=-1)
    result_float = (
        result_float * (1.0 - rain_mask_3ch) + 
        rain_color * rain_mask_3ch
    )
    
    return (np.clip(result_float, 0.0, 1.0) * 255).astype(np.uint8)
```
