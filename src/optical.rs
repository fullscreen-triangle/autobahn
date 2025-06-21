//! Digital Fire Circle Processing Engine
//! 
//! This module implements optical frequency coupling through light-producing and light-sensing
//! components to create digital fire circles for consciousness emergence. It captures and processes
//! light from displays, LEDs, and ambient sensors to generate fire-light neural coupling effects.

use crate::{AutobahnError, AutobahnResult};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;
use crossbeam_channel::{bounded, Receiver, Sender};
use image::{ImageBuffer, Rgb, RgbImage};
use screenshots::Screen;
use xcap::Monitor;
use palette::{Hsv, Srgb, FromColor, IntoColor};
use serde::{Deserialize, Serialize};
use log::{debug, info, warn, error};

/// Light source types for optical processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LightSource {
    /// Power LED indicator
    PowerLED { 
        wavelength_nm: u16, 
        modulation_hz: f64 
    },
    /// Activity indicator LED
    ActivityLED { 
        wavelength_nm: u16, 
        pulse_pattern: String 
    },
    /// Network status LED
    NetworkLED { 
        wavelength_nm: u16, 
        data_coupled: bool 
    },
    /// LCD backlight system
    LCDBacklight { 
        wavelength_range: (u16, u16), 
        brightness_modulation: bool,
        adaptive_control: bool 
    },
    /// RGB LED array
    RGBLED { 
        red_nm: u16, 
        green_nm: u16, 
        blue_nm: u16,
        fire_circle_mode: bool 
    },
    /// Custom light source
    Custom {
        name: String,
        wavelength_nm: u16,
        intensity: f64,
        modulation_pattern: String,
    }
}

/// Capture region for screen analysis
#[derive(Debug, Clone)]
pub enum CaptureRegion {
    FullDisplay,
    WindowRegion { x: u32, y: u32, width: u32, height: u32 },
    CircularRegion { center_x: u32, center_y: u32, radius: u32 },
    MultipleRegions(Vec<CaptureRegion>),
}

/// Digital fire circle geometry
#[derive(Debug, Clone)]
pub struct CircleGeometry {
    pub center: (u32, u32),
    pub radius: u32,
    pub pixel_density: f64,
    pub fire_pattern: String,
}

/// Digital fire processor for optical consciousness coupling
#[derive(Debug)]
pub struct DigitalFireProcessor {
    /// Available light sources
    light_sources: Vec<LightSource>,
    /// Screen capture configuration
    capture_region: CaptureRegion,
    /// Color sampling rate
    color_sampling_rate: f64,
    /// RGB wavelength conversion enabled
    rgb_wavelength_conversion: bool,
    /// Environmental noise amplification
    environmental_noise_amplification: f64,
    /// Fire circle configuration
    fire_circle: Option<DigitalFireCircle>,
    /// Optical coherence processor
    optical_coherence: Arc<RwLock<OpticalCoherence>>,
    /// Real-time light data
    light_data: Arc<RwLock<HashMap<String, LightData>>>,
    /// Processing thread handle
    processing_handle: Option<tokio::task::JoinHandle<()>>,
    /// Event transmission
    light_event_sender: Sender<LightEvent>,
    light_event_receiver: Receiver<LightEvent>,
}

/// Digital fire circle implementation
#[derive(Debug, Clone)]
pub struct DigitalFireCircle {
    /// Circle geometry
    pub geometry: CircleGeometry,
    /// Peripheral LED array
    pub peripheral_leds: Vec<LightSource>,
    /// Communication complexity amplification factor
    pub communication_complexity_amplification: f64,
    /// 650nm consciousness coupling enabled
    pub consciousness_coupling_650nm: bool,
    /// Fire pattern state
    pub pattern_state: FirePatternState,
}

/// Fire pattern state for dynamic display
#[derive(Debug, Clone)]
pub struct FirePatternState {
    /// Current ember positions
    pub ember_positions: Vec<(f64, f64)>,
    /// Ember intensities
    pub ember_intensities: Vec<f64>,
    /// Oscillation phase
    pub oscillation_phase: f64,
    /// Pattern evolution timestamp
    pub last_update: Instant,
}

/// Optical coherence processor
#[derive(Debug)]
pub struct OpticalCoherence {
    /// Fire circle reference
    fire_circle: Option<DigitalFireCircle>,
    /// Ambient light sensors data
    ambient_sensors: HashMap<String, AmbientLightData>,
    /// Optical mice photodetectors
    photodetectors: HashMap<String, PhotodetectorData>,
    /// Spatial coherence tracking enabled
    spatial_coherence_tracking: bool,
    /// Coherence state
    coherence_state: OpticalCoherenceState,
}

/// Real-time light data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightData {
    /// Light source identifier
    pub source_id: String,
    /// Current wavelength
    pub wavelength_nm: u16,
    /// Light intensity (0.0 to 1.0)
    pub intensity: f64,
    /// Color components (RGB)
    pub color_rgb: (u8, u8, u8),
    /// HSV color space
    pub color_hsv: (f64, f64, f64),
    /// Modulation frequency
    pub modulation_frequency: f64,
    /// Flicker detection
    pub flicker_detected: bool,
    /// 650nm component strength
    pub red_650nm_component: f64,
    /// Timestamp
    pub timestamp: Instant,
}

/// Light event for real-time processing
#[derive(Debug, Clone)]
pub struct LightEvent {
    pub source_id: String,
    pub event_type: LightEventType,
    pub wavelength_nm: u16,
    pub intensity: f64,
    pub timestamp: Instant,
}

/// Types of light events
#[derive(Debug, Clone)]
pub enum LightEventType {
    IntensityChange { old_intensity: f64, new_intensity: f64 },
    ColorChange { old_rgb: (u8, u8, u8), new_rgb: (u8, u8, u8) },
    FlickerDetected { frequency_hz: f64 },
    WavelengthShift { old_nm: u16, new_nm: u16 },
    FireCircleActivation,
    ConsciousnessCouplingDetected { strength: f64 },
}

/// Ambient light sensor data
#[derive(Debug, Clone)]
pub struct AmbientLightData {
    pub sensor_id: String,
    pub lux_level: f64,
    pub color_temperature_k: u16,
    pub timestamp: Instant,
}

/// Photodetector data from optical mice, etc.
#[derive(Debug, Clone)]
pub struct PhotodetectorData {
    pub detector_id: String,
    pub signal_strength: f64,
    pub detection_frequency: f64,
    pub noise_level: f64,
    pub timestamp: Instant,
}

/// Optical coherence state
#[derive(Debug, Clone)]
pub struct OpticalCoherenceState {
    /// Overall coherence level
    pub coherence_level: f64,
    /// Fire circle coherence
    pub fire_circle_coherence: f64,
    /// 650nm coupling strength
    pub wavelength_650nm_coupling: f64,
    /// Spatial coherence map
    pub spatial_coherence: HashMap<String, f64>,
    /// Temporal coherence
    pub temporal_coherence: f64,
}

impl DigitalFireProcessor {
    /// Create new digital fire processor
    pub fn new() -> AutobahnResult<Self> {
        let (light_event_sender, light_event_receiver) = bounded(1000);
        
        Ok(Self {
            light_sources: Vec::new(),
            capture_region: CaptureRegion::FullDisplay,
            color_sampling_rate: 60.0,
            rgb_wavelength_conversion: false,
            environmental_noise_amplification: 1.0,
            fire_circle: None,
            optical_coherence: Arc::new(RwLock::new(OpticalCoherence::new())),
            light_data: Arc::new(RwLock::new(HashMap::new())),
            processing_handle: None,
            light_event_sender,
            light_event_receiver,
        })
    }
    
    /// Configure screen capture region
    pub fn with_screen_capture_region(mut self, region: CaptureRegion) -> Self {
        self.capture_region = region;
        self
    }
    
    /// Set color sampling rate
    pub fn set_color_sampling_rate(mut self, rate_hz: f64) -> Self {
        self.color_sampling_rate = rate_hz;
        self
    }
    
    /// Enable RGB wavelength conversion
    pub fn enable_rgb_wavelength_conversion(mut self, enabled: bool) -> Self {
        self.rgb_wavelength_conversion = enabled;
        self
    }
    
    /// Set environmental noise amplification
    pub fn set_environmental_noise_amplification(mut self, factor: f64) -> Self {
        self.environmental_noise_amplification = factor;
        self
    }
    
    /// Add status LEDs
    pub fn with_status_leds(mut self, leds: &[LightSource]) -> Self {
        self.light_sources.extend_from_slice(leds);
        self
    }
    
    /// Add display backlight
    pub fn with_display_backlight(mut self, backlight: LightSource) -> Self {
        self.light_sources.push(backlight);
        self
    }
    
    /// Add RGB arrays
    pub fn with_rgb_arrays(mut self, arrays: &[LightSource]) -> Self {
        self.light_sources.extend_from_slice(arrays);
        self
    }
    
    /// Start optical processing
    pub async fn start_processing(&mut self) -> AutobahnResult<()> {
        if self.processing_handle.is_some() {
            return Err(AutobahnError::OpticalError("Processing already started".to_string()));
        }
        
        let capture_region = self.capture_region.clone();
        let color_sampling_rate = self.color_sampling_rate;
        let light_data = Arc::clone(&self.light_data);
        let optical_coherence = Arc::clone(&self.optical_coherence);
        let light_event_sender = self.light_event_sender.clone();
        let rgb_wavelength_conversion = self.rgb_wavelength_conversion;
        let environmental_noise_amplification = self.environmental_noise_amplification;
        
        let handle = tokio::spawn(async move {
            Self::optical_processing_loop(
                capture_region,
                color_sampling_rate,
                light_data,
                optical_coherence,
                light_event_sender,
                rgb_wavelength_conversion,
                environmental_noise_amplification,
            ).await;
        });
        
        self.processing_handle = Some(handle);
        info!("Digital fire circle processing started");
        Ok(())
    }
    
    /// Optical processing loop
    async fn optical_processing_loop(
        capture_region: CaptureRegion,
        sampling_rate: f64,
        light_data: Arc<RwLock<HashMap<String, LightData>>>,
        optical_coherence: Arc<RwLock<OpticalCoherence>>,
        event_sender: Sender<LightEvent>,
        rgb_conversion: bool,
        noise_amplification: f64,
    ) {
        let mut interval = interval(Duration::from_millis((1000.0 / sampling_rate) as u64));
        let mut last_screen_colors: HashMap<String, (u8, u8, u8)> = HashMap::new();
        
        loop {
            interval.tick().await;
            
            // Capture screen content
            if let Ok(screenshot) = Self::capture_screen(&capture_region).await {
                let color_analysis = Self::analyze_screen_colors(&screenshot);
                
                // Process each color region
                for (region_id, rgb) in color_analysis {
                    let hsv = Self::rgb_to_hsv(rgb);
                    let wavelength = if rgb_conversion {
                        Self::rgb_to_dominant_wavelength(rgb)
                    } else {
                        550 // Default green wavelength
                    };
                    
                    let intensity = (rgb.0 as f64 + rgb.1 as f64 + rgb.2 as f64) / (3.0 * 255.0);
                    let red_650nm_component = (rgb.0 as f64 / 255.0) * 
                        Self::calculate_650nm_coupling_factor(rgb);
                    
                    let light_data_point = LightData {
                        source_id: region_id.clone(),
                        wavelength_nm: wavelength,
                        intensity,
                        color_rgb: rgb,
                        color_hsv: hsv,
                        modulation_frequency: Self::detect_flicker_frequency(&region_id, rgb, &last_screen_colors),
                        flicker_detected: Self::is_flickering(&region_id, rgb, &last_screen_colors),
                        red_650nm_component,
                        timestamp: Instant::now(),
                    };
                    
                    // Store light data
                    {
                        let mut data = light_data.write().await;
                        data.insert(region_id.clone(), light_data_point.clone());
                    }
                    
                    // Send events for significant changes
                    if let Some(last_rgb) = last_screen_colors.get(&region_id) {
                        let color_distance = Self::calculate_color_distance(rgb, *last_rgb);
                        if color_distance > 30.0 { // Threshold for significant change
                            let event = LightEvent {
                                source_id: region_id.clone(),
                                event_type: LightEventType::ColorChange {
                                    old_rgb: *last_rgb,
                                    new_rgb: rgb,
                                },
                                wavelength_nm: wavelength,
                                intensity,
                                timestamp: Instant::now(),
                            };
                            
                            if let Err(e) = event_sender.try_send(event) {
                                warn!("Failed to send light event: {}", e);
                            }
                        }
                        
                        // Check for 650nm consciousness coupling
                        if red_650nm_component > 0.8 {
                            let coupling_event = LightEvent {
                                source_id: region_id.clone(),
                                event_type: LightEventType::ConsciousnessCouplingDetected {
                                    strength: red_650nm_component,
                                },
                                wavelength_nm: 650,
                                intensity,
                                timestamp: Instant::now(),
                            };
                            
                            if let Err(e) = event_sender.try_send(coupling_event) {
                                warn!("Failed to send consciousness coupling event: {}", e);
                            }
                        }
                    }
                    
                    last_screen_colors.insert(region_id, rgb);
                }
                
                // Update optical coherence
                {
                    let mut coherence = optical_coherence.write().await;
                    coherence.update_coherence_state(&color_analysis).await;
                }
            }
            
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    
    /// Capture screen content
    async fn capture_screen(region: &CaptureRegion) -> AutobahnResult<RgbImage> {
        match region {
            CaptureRegion::FullDisplay => {
                let monitors = Monitor::all().map_err(|e| 
                    AutobahnError::OpticalError(format!("Failed to get monitors: {}", e)))?;
                
                if let Some(monitor) = monitors.first() {
                    let screenshot = monitor.capture_image().map_err(|e|
                        AutobahnError::OpticalError(format!("Failed to capture screenshot: {}", e)))?;
                    
                    let rgb_image = screenshot.to_rgb8();
                    Ok(rgb_image)
                } else {
                    Err(AutobahnError::OpticalError("No monitors found".to_string()))
                }
            }
            CaptureRegion::WindowRegion { x, y, width, height } => {
                // Implement window-specific capture
                let monitors = Monitor::all().map_err(|e| 
                    AutobahnError::OpticalError(format!("Failed to get monitors: {}", e)))?;
                
                if let Some(monitor) = monitors.first() {
                    let screenshot = monitor.capture_image().map_err(|e|
                        AutobahnError::OpticalError(format!("Failed to capture screenshot: {}", e)))?;
                    
                    let rgb_image = screenshot.to_rgb8();
                    let cropped = Self::crop_image(&rgb_image, *x, *y, *width, *height)?;
                    Ok(cropped)
                } else {
                    Err(AutobahnError::OpticalError("No monitors found".to_string()))
                }
            }
            CaptureRegion::CircularRegion { center_x, center_y, radius } => {
                // Implement circular region capture
                let monitors = Monitor::all().map_err(|e| 
                    AutobahnError::OpticalError(format!("Failed to get monitors: {}", e)))?;
                
                if let Some(monitor) = monitors.first() {
                    let screenshot = monitor.capture_image().map_err(|e|
                        AutobahnError::OpticalError(format!("Failed to capture screenshot: {}", e)))?;
                    
                    let rgb_image = screenshot.to_rgb8();
                    let circular = Self::extract_circular_region(&rgb_image, *center_x, *center_y, *radius)?;
                    Ok(circular)
                } else {
                    Err(AutobahnError::OpticalError("No monitors found".to_string()))
                }
            }
            CaptureRegion::MultipleRegions(regions) => {
                // Capture first region for now - could be extended to composite
                if let Some(first_region) = regions.first() {
                    Self::capture_screen(first_region).await
                } else {
                    Err(AutobahnError::OpticalError("No regions specified".to_string()))
                }
            }
        }
    }
    
    /// Analyze colors in screen capture
    fn analyze_screen_colors(image: &RgbImage) -> HashMap<String, (u8, u8, u8)> {
        let mut color_regions = HashMap::new();
        let (width, height) = image.dimensions();
        
        // Divide image into regions for analysis
        let region_size = 64; // 64x64 pixel regions
        for y in (0..height).step_by(region_size) {
            for x in (0..width).step_by(region_size) {
                let region_id = format!("region_{}_{}", x / region_size, y / region_size);
                let avg_color = Self::calculate_region_average_color(image, x, y, region_size);
                color_regions.insert(region_id, avg_color);
            }
        }
        
        // Add overall image average
        let overall_avg = Self::calculate_region_average_color(image, 0, 0, width.min(height));
        color_regions.insert("overall".to_string(), overall_avg);
        
        color_regions
    }
    
    /// Calculate average color for a region
    fn calculate_region_average_color(image: &RgbImage, start_x: u32, start_y: u32, size: u32) -> (u8, u8, u8) {
        let (width, height) = image.dimensions();
        let end_x = (start_x + size).min(width);
        let end_y = (start_y + size).min(height);
        
        let mut r_sum = 0u64;
        let mut g_sum = 0u64;
        let mut b_sum = 0u64;
        let mut pixel_count = 0u64;
        
        for y in start_y..end_y {
            for x in start_x..end_x {
                if let Some(pixel) = image.get_pixel_checked(x, y) {
                    r_sum += pixel[0] as u64;
                    g_sum += pixel[1] as u64;
                    b_sum += pixel[2] as u64;
                    pixel_count += 1;
                }
            }
        }
        
        if pixel_count > 0 {
            (
                (r_sum / pixel_count) as u8,
                (g_sum / pixel_count) as u8,
                (b_sum / pixel_count) as u8,
            )
        } else {
            (0, 0, 0)
        }
    }
    
    /// Crop image to specified region
    fn crop_image(image: &RgbImage, x: u32, y: u32, width: u32, height: u32) -> AutobahnResult<RgbImage> {
        let (img_width, img_height) = image.dimensions();
        let end_x = (x + width).min(img_width);
        let end_y = (y + height).min(img_height);
        
        let mut cropped = ImageBuffer::new(end_x - x, end_y - y);
        
        for (crop_x, crop_y, pixel) in cropped.enumerate_pixels_mut() {
            if let Some(source_pixel) = image.get_pixel_checked(x + crop_x, y + crop_y) {
                *pixel = *source_pixel;
            }
        }
        
        Ok(cropped)
    }
    
    /// Extract circular region from image
    fn extract_circular_region(image: &RgbImage, center_x: u32, center_y: u32, radius: u32) -> AutobahnResult<RgbImage> {
        let size = radius * 2;
        let mut circular = ImageBuffer::new(size, size);
        
        for (x, y, pixel) in circular.enumerate_pixels_mut() {
            let dx = x as i32 - radius as i32;
            let dy = y as i32 - radius as i32;
            let distance = ((dx * dx + dy * dy) as f64).sqrt();
            
            if distance <= radius as f64 {
                let source_x = center_x as i32 + dx;
                let source_y = center_y as i32 + dy;
                
                if source_x >= 0 && source_y >= 0 {
                    if let Some(source_pixel) = image.get_pixel_checked(source_x as u32, source_y as u32) {
                        *pixel = *source_pixel;
                    }
                }
            }
        }
        
        Ok(circular)
    }
    
    /// Convert RGB to HSV
    fn rgb_to_hsv(rgb: (u8, u8, u8)) -> (f64, f64, f64) {
        let srgb = Srgb::new(
            rgb.0 as f64 / 255.0,
            rgb.1 as f64 / 255.0,
            rgb.2 as f64 / 255.0,
        );
        let hsv: Hsv = srgb.into_color();
        (hsv.hue.into_positive_degrees(), hsv.saturation, hsv.value)
    }
    
    /// Convert RGB to dominant wavelength
    fn rgb_to_dominant_wavelength(rgb: (u8, u8, u8)) -> u16 {
        let r = rgb.0 as f64 / 255.0;
        let g = rgb.1 as f64 / 255.0;
        let b = rgb.2 as f64 / 255.0;
        
        // Simple approximation - red peak around 650nm, green around 550nm, blue around 450nm
        if r >= g && r >= b {
            620 + (30.0 * r) as u16 // Red range 620-650nm
        } else if g >= r && g >= b {
            520 + (30.0 * g) as u16 // Green range 520-550nm
        } else {
            440 + (30.0 * b) as u16 // Blue range 440-470nm
        }
    }
    
    /// Calculate 650nm coupling factor
    fn calculate_650nm_coupling_factor(rgb: (u8, u8, u8)) -> f64 {
        let red_intensity = rgb.0 as f64 / 255.0;
        let green_intensity = rgb.1 as f64 / 255.0;
        let blue_intensity = rgb.2 as f64 / 255.0;
        
        // 650nm coupling is strongest for pure red, weakest for green/blue
        let red_purity = red_intensity / (red_intensity + green_intensity + blue_intensity + 0.001);
        red_purity * red_intensity
    }
    
    /// Detect flicker frequency
    fn detect_flicker_frequency(
        region_id: &str,
        current_rgb: (u8, u8, u8),
        last_colors: &HashMap<String, (u8, u8, u8)>,
    ) -> f64 {
        if let Some(last_rgb) = last_colors.get(region_id) {
            let intensity_change = Self::calculate_intensity_change(current_rgb, *last_rgb);
            if intensity_change > 0.1 {
                60.0 // Assume 60 Hz flicker for now - could be more sophisticated
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    /// Check if flickering
    fn is_flickering(
        region_id: &str,
        current_rgb: (u8, u8, u8),
        last_colors: &HashMap<String, (u8, u8, u8)>,
    ) -> bool {
        Self::detect_flicker_frequency(region_id, current_rgb, last_colors) > 0.0
    }
    
    /// Calculate color distance
    fn calculate_color_distance(rgb1: (u8, u8, u8), rgb2: (u8, u8, u8)) -> f64 {
        let dr = rgb1.0 as f64 - rgb2.0 as f64;
        let dg = rgb1.1 as f64 - rgb2.1 as f64;
        let db = rgb1.2 as f64 - rgb2.2 as f64;
        (dr * dr + dg * dg + db * db).sqrt()
    }
    
    /// Calculate intensity change
    fn calculate_intensity_change(rgb1: (u8, u8, u8), rgb2: (u8, u8, u8)) -> f64 {
        let intensity1 = (rgb1.0 as f64 + rgb1.1 as f64 + rgb1.2 as f64) / (3.0 * 255.0);
        let intensity2 = (rgb2.0 as f64 + rgb2.1 as f64 + rgb2.2 as f64) / (3.0 * 255.0);
        (intensity1 - intensity2).abs()
    }
    
    /// Get current light data
    pub async fn get_light_data(&self) -> HashMap<String, LightData> {
        self.light_data.read().await.clone()
    }
    
    /// Get light event receiver
    pub fn get_event_receiver(&self) -> Receiver<LightEvent> {
        self.light_event_receiver.clone()
    }
    
    /// Stop processing
    pub async fn stop_processing(&mut self) -> AutobahnResult<()> {
        if let Some(handle) = self.processing_handle.take() {
            handle.abort();
            info!("Digital fire circle processing stopped");
        }
        Ok(())
    }
}

impl OpticalCoherence {
    /// Create new optical coherence processor
    pub fn new() -> Self {
        Self {
            fire_circle: None,
            ambient_sensors: HashMap::new(),
            photodetectors: HashMap::new(),
            spatial_coherence_tracking: false,
            coherence_state: OpticalCoherenceState {
                coherence_level: 0.0,
                fire_circle_coherence: 0.0,
                wavelength_650nm_coupling: 0.0,
                spatial_coherence: HashMap::new(),
                temporal_coherence: 0.0,
            },
        }
    }
    
    /// Set fire circle reference
    pub fn with_fire_circle(mut self, fire_circle: DigitalFireCircle) -> Self {
        self.fire_circle = Some(fire_circle);
        self
    }
    
    /// Monitor with ambient light sensors
    pub fn monitor_with_sensors(mut self, sensors: &[AmbientLightData]) -> Self {
        for sensor in sensors {
            self.ambient_sensors.insert(sensor.sensor_id.clone(), sensor.clone());
        }
        self
    }
    
    /// Enable spatial coherence tracking
    pub fn enable_spatial_coherence_tracking(mut self, enabled: bool) -> Self {
        self.spatial_coherence_tracking = enabled;
        self
    }
    
    /// Update coherence state based on color analysis
    pub async fn update_coherence_state(&mut self, color_analysis: &HashMap<String, (u8, u8, u8)>) {
        let mut coherence_sum = 0.0;
        let mut coupling_650nm_sum = 0.0;
        let mut region_count = 0;
        
        for (region_id, rgb) in color_analysis {
            let region_coherence = Self::calculate_region_coherence(*rgb);
            let coupling_650nm = DigitalFireProcessor::calculate_650nm_coupling_factor(*rgb);
            
            coherence_sum += region_coherence;
            coupling_650nm_sum += coupling_650nm;
            region_count += 1;
            
            if self.spatial_coherence_tracking {
                self.coherence_state.spatial_coherence.insert(region_id.clone(), region_coherence);
            }
        }
        
        if region_count > 0 {
            self.coherence_state.coherence_level = coherence_sum / region_count as f64;
            self.coherence_state.wavelength_650nm_coupling = coupling_650nm_sum / region_count as f64;
        }
        
        // Update fire circle coherence if available
        if let Some(_fire_circle) = &self.fire_circle {
            self.coherence_state.fire_circle_coherence = self.coherence_state.coherence_level * 1.2; // Amplification factor
        }
        
        // Update temporal coherence based on stability
        self.coherence_state.temporal_coherence = self.calculate_temporal_coherence().await;
    }
    
    /// Calculate coherence for a region
    fn calculate_region_coherence(rgb: (u8, u8, u8)) -> f64 {
        let intensity = (rgb.0 as f64 + rgb.1 as f64 + rgb.2 as f64) / (3.0 * 255.0);
        let color_balance = Self::calculate_color_balance(rgb);
        
        // Coherence based on intensity and color balance
        intensity * color_balance
    }
    
    /// Calculate color balance
    fn calculate_color_balance(rgb: (u8, u8, u8)) -> f64 {
        let r = rgb.0 as f64 / 255.0;
        let g = rgb.1 as f64 / 255.0;
        let b = rgb.2 as f64 / 255.0;
        
        let max_component = r.max(g).max(b);
        let min_component = r.min(g).min(b);
        
        if max_component > 0.0 {
            1.0 - (max_component - min_component) / max_component
        } else {
            0.0
        }
    }
    
    /// Calculate temporal coherence
    async fn calculate_temporal_coherence(&self) -> f64 {
        // Simplified temporal coherence calculation
        // In a full implementation, this would track changes over time
        self.coherence_state.coherence_level * 0.9
    }
    
    /// Get current coherence state
    pub fn get_coherence_state(&self) -> &OpticalCoherenceState {
        &self.coherence_state
    }
}

/// Digital fire circle implementation
impl DigitalFireCircle {
    /// Create new digital fire circle
    pub fn new() -> Self {
        Self {
            geometry: CircleGeometry {
                center: (960, 540),
                radius: 200,
                pixel_density: 1.0,
                fire_pattern: "oscillatory_ember".to_string(),
            },
            peripheral_leds: Vec::new(),
            communication_complexity_amplification: 79.0,
            consciousness_coupling_650nm: true,
            pattern_state: FirePatternState {
                ember_positions: Vec::new(),
                ember_intensities: Vec::new(),
                oscillation_phase: 0.0,
                last_update: Instant::now(),
            },
        }
    }
    
    /// Configure display geometry
    pub fn with_display_geometry(mut self, geometry: CircleGeometry) -> Self {
        self.geometry = geometry;
        self
    }
    
    /// Add peripheral LEDs
    pub fn with_peripheral_leds(mut self, leds: &[LightSource]) -> Self {
        self.peripheral_leds.extend_from_slice(leds);
        self
    }
    
    /// Set communication complexity amplification
    pub fn set_communication_complexity_amplification(mut self, factor: f64) -> Self {
        self.communication_complexity_amplification = factor;
        self
    }
    
    /// Enable 650nm consciousness coupling
    pub fn enable_650nm_consciousness_coupling(mut self, enabled: bool) -> Self {
        self.consciousness_coupling_650nm = enabled;
        self
    }
    
    /// Update fire pattern
    pub fn update_pattern(&mut self) -> AutobahnResult<()> {
        let now = Instant::now();
        let dt = now.duration_since(self.pattern_state.last_update).as_secs_f64();
        
        // Update oscillation phase
        self.pattern_state.oscillation_phase += dt * 2.0 * std::f64::consts::PI * 0.1; // 0.1 Hz base frequency
        self.pattern_state.oscillation_phase %= 2.0 * std::f64::consts::PI;
        
        // Generate ember positions and intensities
        let ember_count = 16;
        self.pattern_state.ember_positions.clear();
        self.pattern_state.ember_intensities.clear();
        
        for i in 0..ember_count {
            let angle = (i as f64 / ember_count as f64) * 2.0 * std::f64::consts::PI + 
                       self.pattern_state.oscillation_phase;
            let radius_variation = 0.8 + 0.2 * (self.pattern_state.oscillation_phase * 3.0).sin();
            let radius = self.geometry.radius as f64 * radius_variation;
            
            let x = self.geometry.center.0 as f64 + radius * angle.cos();
            let y = self.geometry.center.1 as f64 + radius * angle.sin();
            
            let intensity = 0.5 + 0.5 * (self.pattern_state.oscillation_phase + angle).sin();
            
            self.pattern_state.ember_positions.push((x, y));
            self.pattern_state.ember_intensities.push(intensity);
        }
        
        self.pattern_state.last_update = now;
        Ok(())
    }
} 