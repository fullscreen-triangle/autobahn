//! Environmental Photosynthesis Engine
//! 
//! This module implements visual spectrum ATP conversion and chaos substrate generation
//! from environmental visual complexity. It converts screen color changes and display
//! variations into metabolic energy and environmental noise to create authentic agency
//! illusion through selective attention mechanisms.

use crate::{AutobahnError, AutobahnResult, atp::{ATPManager, MetabolicMode}};
use crate::optical::{DigitalFireProcessor, LightEvent, LightData};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;
use crossbeam_channel::{bounded, Receiver, Sender};
use serde::{Deserialize, Serialize};
use log::{debug, info, warn, error};
use rand::{thread_rng, Rng};

/// Color range for ATP conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColorRange {
    Red(std::ops::Range<u16>),
    Green(std::ops::Range<u16>),
    Blue(std::ops::Range<u16>),
    White(std::ops::Range<u16>),
}

/// ATP generation rate for different color ranges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ATPRate {
    High(f64),
    Medium(f64),
    Low(f64),
    Variable,
}

/// Environmental photosynthesis processor
#[derive(Debug)]
pub struct EnvironmentalPhotosynthesis {
    /// Screen capture region
    capture_region: crate::optical::CaptureRegion,
    /// Color sampling rate in Hz
    color_sampling_rate: f64,
    /// RGB wavelength conversion enabled
    rgb_wavelength_conversion: bool,
    /// ATP conversion rates for different color ranges
    atp_conversion_rates: HashMap<String, ATPRate>,
    /// Environmental noise amplification factor
    environmental_noise_amplification: f64,
    /// Visual ATP manager
    visual_atp_manager: Arc<RwLock<VisualATPManager>>,
    /// Chaos substrate generator
    chaos_generator: Arc<RwLock<ChaosSubstrateGenerator>>,
    /// Processing thread handle
    processing_handle: Option<tokio::task::JoinHandle<()>>,
    /// Event channels
    atp_event_sender: Sender<ATPGenerationEvent>,
    atp_event_receiver: Receiver<ATPGenerationEvent>,
}

/// Visual ATP manager for energy conversion
#[derive(Debug)]
pub struct VisualATPManager {
    /// ATP storage pools by wavelength
    atp_pools: HashMap<u16, f64>,
    /// Total ATP capacity
    total_capacity: f64,
    /// Current ATP level
    current_atp: f64,
    /// Conversion efficiency by wavelength
    conversion_efficiency: HashMap<u16, f64>,
    /// Metabolic mode for ATP usage
    metabolic_mode: MetabolicMode,
}

/// Color metabolism processor
#[derive(Debug)]
pub struct ColorMetabolism {
    /// Associated photosynthesis engine
    photosynthesis_engine: Arc<RwLock<EnvironmentalPhotosynthesis>>,
    /// Chaos substrate generation enabled
    chaos_substrate_generation: bool,
    /// Agency illusion threshold
    agency_illusion_threshold: f64,
    /// Visual complexity buffer
    complexity_buffer: Vec<f64>,
    /// Metabolic state
    metabolic_state: ColorMetabolicState,
}

/// Chaos substrate generator for environmental complexity
#[derive(Debug)]
pub struct ChaosSubstrateGenerator {
    /// Visual entropy source
    visual_entropy_source: HashMap<String, f64>,
    /// Information overload threshold
    overload_threshold: f64,
    /// Attention filtering mechanisms
    attention_filters: Vec<FilterMode>,
    /// Predetermined navigation masking enabled
    predetermined_navigation_masking: bool,
    /// Current chaos level
    chaos_level: f64,
    /// Environmental complexity map
    complexity_map: HashMap<String, f64>,
}

/// Agency illusion engine
#[derive(Debug)]
pub struct AgencyIllusionEngine {
    /// Chaos substrate source
    chaos_substrate: Arc<RwLock<ChaosSubstrateGenerator>>,
    /// Focus allocation complexity
    focus_allocation_complexity: f64,
    /// Choice emergence simulation enabled
    choice_emergence_simulation: bool,
    /// Subjective control feeling optimization level
    subjective_control_optimization: f64,
    /// Attention targets and priorities
    attention_targets: Vec<AttentionTarget>,
    /// Current agency state
    agency_state: AgencyState,
}

/// Filter modes for attention mechanisms
#[derive(Debug, Clone)]
pub enum FilterMode {
    ColorSalience { sensitivity: f64 },
    MotionDetection { threshold: f64 },
    BrightnessGradient { sampling_rate: f64 },
    PatternComplexity { analysis_depth: u8 },
}

/// Attention targets for selective focus
#[derive(Debug, Clone)]
pub enum AttentionTarget {
    HighContrastRegions { priority: f64 },
    MovingElements { priority: f64 },
    ColorTransitions { priority: f64 },
    PatternEmergence { priority: f64 },
}

/// ATP generation event
#[derive(Debug, Clone)]
pub struct ATPGenerationEvent {
    pub wavelength_nm: u16,
    pub atp_generated: f64,
    pub color_rgb: (u8, u8, u8),
    pub conversion_efficiency: f64,
    pub timestamp: Instant,
    pub event_type: ATPEventType,
}

/// Types of ATP generation events
#[derive(Debug, Clone)]
pub enum ATPEventType {
    HighEnergyBurst { duration_ms: u64 },
    SustainedGeneration { rate_per_second: f64 },
    WavelengthOptimization { wavelength: u16, efficiency_gain: f64 },
    EnvironmentalChaosIncrease { chaos_level: f64 },
    AgencyIllusionActivation { strength: f64 },
}

/// Color metabolic state
#[derive(Debug, Clone)]
pub struct ColorMetabolicState {
    /// Active wavelength processing
    pub active_wavelengths: Vec<u16>,
    /// Current metabolic rate
    pub metabolic_rate: f64,
    /// ATP generation efficiency
    pub generation_efficiency: f64,
    /// Visual complexity score
    pub visual_complexity: f64,
    /// Environmental chaos level
    pub chaos_level: f64,
}

/// Agency state for illusion tracking
#[derive(Debug, Clone)]
pub struct AgencyState {
    /// Current agency feeling strength
    pub agency_strength: f64,
    /// Active attention focus areas
    pub focus_areas: Vec<String>,
    /// Choice availability perception
    pub choice_availability: f64,
    /// Control illusion level
    pub control_illusion: f64,
    /// Selective attention active
    pub selective_attention_active: bool,
}

/// Contextual focus engine for choice emergence
#[derive(Debug)]
pub struct ContextualFocusEngine {
    /// Environmental photosynthesis reference
    photosynthesis_engine: Arc<RwLock<EnvironmentalPhotosynthesis>>,
    /// Attention competition configuration
    attention_competition: Vec<AttentionTarget>,
    /// Choice simulation threshold
    choice_simulation_threshold: f64,
    /// Control illusion amplification enabled
    control_illusion_amplification: bool,
    /// Current focus state
    focus_state: FocusState,
}

/// Focus state for contextual processing
#[derive(Debug, Clone)]
pub struct FocusState {
    /// Primary focus region
    pub primary_focus: Option<String>,
    /// Secondary focus areas
    pub secondary_focus: Vec<String>,
    /// Focus intensity
    pub focus_intensity: f64,
    /// Attention allocation map
    pub attention_allocation: HashMap<String, f64>,
    /// Choice emergence detected
    pub choice_emergence: bool,
}

impl EnvironmentalPhotosynthesis {
    /// Create new environmental photosynthesis system
    pub fn new() -> AutobahnResult<Self> {
        let (atp_event_sender, atp_event_receiver) = bounded(1000);
        
        Ok(Self {
            capture_region: crate::optical::CaptureRegion::FullDisplay,
            color_sampling_rate: 120.0,
            rgb_wavelength_conversion: false,
            atp_conversion_rates: HashMap::new(),
            environmental_noise_amplification: 1.0,
            visual_atp_manager: Arc::new(RwLock::new(VisualATPManager::new())),
            chaos_generator: Arc::new(RwLock::new(ChaosSubstrateGenerator::new())),
            processing_handle: None,
            atp_event_sender,
            atp_event_receiver,
        })
    }
    
    /// Configure screen capture region
    pub fn with_screen_capture_region(mut self, region: crate::optical::CaptureRegion) -> Self {
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
    
    /// Configure ATP conversion rates
    pub fn configure_atp_conversion_rates(mut self, rates: &[(ColorRange, ATPRate)]) -> Self {
        for (color_range, atp_rate) in rates {
            let key = match color_range {
                ColorRange::Red(range) => format!("red_{}_{}", range.start, range.end),
                ColorRange::Green(range) => format!("green_{}_{}", range.start, range.end),
                ColorRange::Blue(range) => format!("blue_{}_{}", range.start, range.end),
                ColorRange::White(range) => format!("white_{}_{}", range.start, range.end),
            };
            self.atp_conversion_rates.insert(key, atp_rate.clone());
        }
        self
    }
    
    /// Set environmental noise amplification
    pub fn set_environmental_noise_amplification(mut self, factor: f64) -> Self {
        self.environmental_noise_amplification = factor;
        self
    }
    
    /// Start photosynthesis processing
    pub async fn start_processing(&mut self) -> AutobahnResult<()> {
        if self.processing_handle.is_some() {
            return Err(AutobahnError::PhotosynthesisError("Processing already started".to_string()));
        }
        
        let color_sampling_rate = self.color_sampling_rate;
        let capture_region = self.capture_region.clone();
        let visual_atp_manager = Arc::clone(&self.visual_atp_manager);
        let chaos_generator = Arc::clone(&self.chaos_generator);
        let atp_event_sender = self.atp_event_sender.clone();
        let atp_conversion_rates = self.atp_conversion_rates.clone();
        let environmental_noise_amplification = self.environmental_noise_amplification;
        
        let handle = tokio::spawn(async move {
            Self::photosynthesis_processing_loop(
                color_sampling_rate,
                capture_region,
                visual_atp_manager,
                chaos_generator,
                atp_event_sender,
                atp_conversion_rates,
                environmental_noise_amplification,
            ).await;
        });
        
        self.processing_handle = Some(handle);
        info!("Environmental photosynthesis processing started");
        Ok(())
    }
    
    /// Photosynthesis processing loop
    async fn photosynthesis_processing_loop(
        sampling_rate: f64,
        capture_region: crate::optical::CaptureRegion,
        visual_atp_manager: Arc<RwLock<VisualATPManager>>,
        chaos_generator: Arc<RwLock<ChaosSubstrateGenerator>>,
        event_sender: Sender<ATPGenerationEvent>,
        conversion_rates: HashMap<String, ATPRate>,
        noise_amplification: f64,
    ) {
        let mut interval = interval(Duration::from_millis((1000.0 / sampling_rate) as u64));
        let mut digital_fire_processor = match DigitalFireProcessor::new() {
            Ok(processor) => processor,
            Err(e) => {
                error!("Failed to create digital fire processor: {}", e);
                return;
            }
        };
        
        if let Err(e) = digital_fire_processor.start_processing().await {
            error!("Failed to start digital fire processor: {}", e);
            return;
        }
        
        loop {
            interval.tick().await;
            
            // Get current light data
            let light_data = digital_fire_processor.get_light_data().await;
            
            // Process each light source for ATP generation
            for (source_id, light_info) in &light_data {
                let wavelength = light_info.wavelength_nm;
                let intensity = light_info.intensity;
                let rgb = light_info.color_rgb;
                
                // Calculate ATP generation based on wavelength and intensity
                let atp_generated = Self::calculate_atp_generation(
                    wavelength,
                    intensity,
                    rgb,
                    &conversion_rates,
                );
                
                if atp_generated > 0.001 {
                    // Store ATP in manager
                    {
                        let mut atp_manager = visual_atp_manager.write().await;
                        atp_manager.add_atp(wavelength, atp_generated).await;
                    }
                    
                    // Generate chaos substrate from visual complexity
                    let visual_complexity = Self::calculate_visual_complexity(rgb, intensity);
                    {
                        let mut chaos_gen = chaos_generator.write().await;
                        chaos_gen.add_visual_entropy(source_id.clone(), visual_complexity * noise_amplification);
                    }
                    
                    // Send ATP generation event
                    let event = ATPGenerationEvent {
                        wavelength_nm: wavelength,
                        atp_generated,
                        color_rgb: rgb,
                        conversion_efficiency: Self::get_conversion_efficiency(wavelength),
                        timestamp: Instant::now(),
                        event_type: if intensity > 0.8 {
                            ATPEventType::HighEnergyBurst { duration_ms: 100 }
                        } else {
                            ATPEventType::SustainedGeneration { rate_per_second: atp_generated * sampling_rate }
                        },
                    };
                    
                    if let Err(e) = event_sender.try_send(event) {
                        warn!("Failed to send ATP generation event: {}", e);
                    }
                }
            }
            
            // Check for environmental chaos and agency illusion activation
            {
                let chaos_gen = chaos_generator.read().await;
                if chaos_gen.get_chaos_level() > 0.85 {
                    let agency_event = ATPGenerationEvent {
                        wavelength_nm: 555, // Green wavelength for agency
                        atp_generated: 0.0,
                        color_rgb: (0, 255, 0),
                        conversion_efficiency: 1.0,
                        timestamp: Instant::now(),
                        event_type: ATPEventType::AgencyIllusionActivation { 
                            strength: chaos_gen.get_chaos_level() 
                        },
                    };
                    
                    if let Err(e) = event_sender.try_send(agency_event) {
                        warn!("Failed to send agency illusion event: {}", e);
                    }
                }
            }
            
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }
    
    /// Calculate ATP generation from visual input
    fn calculate_atp_generation(
        wavelength: u16,
        intensity: f64,
        rgb: (u8, u8, u8),
        conversion_rates: &HashMap<String, ATPRate>,
    ) -> f64 {
        let base_efficiency = match wavelength {
            620..=700 => 0.157, // Red range - high efficiency (650nm peak)
            495..=570 => 0.123, // Green range - medium efficiency
            450..=495 => 0.089, // Blue range - low efficiency
            _ => 0.05,          // Other wavelengths - low efficiency
        };
        
        let red_component = rgb.0 as f64 / 255.0;
        let green_component = rgb.1 as f64 / 255.0;
        let blue_component = rgb.2 as f64 / 255.0;
        
        // 650nm optimization - peak efficiency for red light
        let wavelength_factor = if wavelength >= 645 && wavelength <= 655 {
            1.5 // 50% bonus for optimal consciousness coupling wavelength
        } else {
            1.0
        };
        
        // ATP generation formula
        let atp = base_efficiency * intensity * wavelength_factor * 
                 (red_component * 0.5 + green_component * 0.3 + blue_component * 0.2);
        
        atp.max(0.0).min(1.0) // Clamp to valid range
    }
    
    /// Calculate visual complexity for chaos generation
    fn calculate_visual_complexity(rgb: (u8, u8, u8), intensity: f64) -> f64 {
        let r = rgb.0 as f64 / 255.0;
        let g = rgb.1 as f64 / 255.0;
        let b = rgb.2 as f64 / 255.0;
        
        // Complexity based on color variance and intensity
        let color_variance = ((r - g).powf(2.0) + (g - b).powf(2.0) + (b - r).powf(2.0)) / 3.0;
        let intensity_factor = intensity * (1.0 - intensity); // Peak at 0.5 intensity
        
        (color_variance + intensity_factor) / 2.0
    }
    
    /// Get conversion efficiency for wavelength
    fn get_conversion_efficiency(wavelength: u16) -> f64 {
        match wavelength {
            650 => 1.0,         // Peak efficiency at 650nm
            620..=700 => 0.8,   // High efficiency for red
            495..=570 => 0.6,   // Medium efficiency for green
            450..=495 => 0.4,   // Low efficiency for blue
            _ => 0.2,           // Very low for others
        }
    }
    
    /// Get ATP event receiver
    pub fn get_event_receiver(&self) -> Receiver<ATPGenerationEvent> {
        self.atp_event_receiver.clone()
    }
    
    /// Get current ATP levels
    pub async fn get_atp_levels(&self) -> HashMap<u16, f64> {
        self.visual_atp_manager.read().await.get_atp_pools().clone()
    }
    
    /// Get chaos level
    pub async fn get_chaos_level(&self) -> f64 {
        self.chaos_generator.read().await.get_chaos_level()
    }
    
    /// Stop processing
    pub async fn stop_processing(&mut self) -> AutobahnResult<()> {
        if let Some(handle) = self.processing_handle.take() {
            handle.abort();
            info!("Environmental photosynthesis processing stopped");
        }
        Ok(())
    }
}

impl VisualATPManager {
    /// Create new visual ATP manager
    pub fn new() -> Self {
        let mut conversion_efficiency = HashMap::new();
        conversion_efficiency.insert(650, 1.0);   // Peak at 650nm
        conversion_efficiency.insert(620, 0.8);   // High for red range
        conversion_efficiency.insert(550, 0.6);   // Medium for green
        conversion_efficiency.insert(470, 0.4);   // Low for blue
        
        Self {
            atp_pools: HashMap::new(),
            total_capacity: 1000.0,
            current_atp: 0.0,
            conversion_efficiency,
            metabolic_mode: MetabolicMode::Balanced,
        }
    }
    
    /// Add ATP to the specified wavelength pool
    pub async fn add_atp(&mut self, wavelength: u16, amount: f64) {
        let current_pool = self.atp_pools.get(&wavelength).unwrap_or(&0.0);
        let new_amount = (current_pool + amount).min(self.total_capacity / 10.0); // Max 10% per wavelength
        
        self.atp_pools.insert(wavelength, new_amount);
        self.current_atp = self.atp_pools.values().sum::<f64>().min(self.total_capacity);
    }
    
    /// Consume ATP for processing
    pub async fn consume_atp(&mut self, wavelength: u16, amount: f64) -> f64 {
        let available = self.atp_pools.get(&wavelength).unwrap_or(&0.0);
        let consumed = amount.min(*available);
        
        if consumed > 0.0 {
            self.atp_pools.insert(wavelength, available - consumed);
            self.current_atp = self.atp_pools.values().sum::<f64>();
        }
        
        consumed
    }
    
    /// Get ATP pools
    pub fn get_atp_pools(&self) -> &HashMap<u16, f64> {
        &self.atp_pools
    }
    
    /// Get total ATP
    pub fn get_total_atp(&self) -> f64 {
        self.current_atp
    }
    
    /// Set metabolic mode
    pub fn set_metabolic_mode(&mut self, mode: MetabolicMode) {
        self.metabolic_mode = mode;
    }
}

impl ColorMetabolism {
    /// Create new color metabolism processor
    pub fn new() -> Self {
        Self {
            photosynthesis_engine: Arc::new(RwLock::new(
                EnvironmentalPhotosynthesis::new().unwrap()
            )),
            chaos_substrate_generation: false,
            agency_illusion_threshold: 0.7,
            complexity_buffer: Vec::new(),
            metabolic_state: ColorMetabolicState {
                active_wavelengths: vec![650, 550, 470],
                metabolic_rate: 1.0,
                generation_efficiency: 0.8,
                visual_complexity: 0.0,
                chaos_level: 0.0,
            },
        }
    }
    
    /// Set photosynthesis engine
    pub fn with_photosynthesis_engine(mut self, engine: Arc<RwLock<EnvironmentalPhotosynthesis>>) -> Self {
        self.photosynthesis_engine = engine;
        self
    }
    
    /// Enable chaos substrate generation
    pub fn enable_chaos_substrate_generation(mut self, enabled: bool) -> Self {
        self.chaos_substrate_generation = enabled;
        self
    }
    
    /// Set agency illusion threshold
    pub fn set_agency_illusion_threshold(mut self, threshold: f64) -> Self {
        self.agency_illusion_threshold = threshold;
        self
    }
    
    /// Process color metabolism
    pub async fn process_metabolism(&mut self) -> AutobahnResult<ColorMetabolicState> {
        // Get current ATP levels
        let atp_levels = {
            let engine = self.photosynthesis_engine.read().await;
            engine.get_atp_levels().await
        };
        
        // Calculate metabolic rate based on ATP availability
        let total_atp: f64 = atp_levels.values().sum();
        self.metabolic_state.metabolic_rate = (total_atp / 100.0).min(2.0).max(0.1);
        
        // Update active wavelengths based on availability
        self.metabolic_state.active_wavelengths.clear();
        for (wavelength, atp) in atp_levels {
            if atp > 1.0 {
                self.metabolic_state.active_wavelengths.push(wavelength);
            }
        }
        
        // Calculate visual complexity
        let chaos_level = {
            let engine = self.photosynthesis_engine.read().await;
            engine.get_chaos_level().await
        };
        
        self.metabolic_state.chaos_level = chaos_level;
        self.metabolic_state.visual_complexity = chaos_level * 0.8;
        
        // Update generation efficiency based on optimal wavelengths
        if self.metabolic_state.active_wavelengths.contains(&650) {
            self.metabolic_state.generation_efficiency = 0.95;
        } else {
            self.metabolic_state.generation_efficiency = 0.6;
        }
        
        Ok(self.metabolic_state.clone())
    }
    
    /// Get current metabolic state
    pub fn get_metabolic_state(&self) -> &ColorMetabolicState {
        &self.metabolic_state
    }
}

impl ChaosSubstrateGenerator {
    /// Create new chaos substrate generator
    pub fn new() -> Self {
        Self {
            visual_entropy_source: HashMap::new(),
            overload_threshold: 0.85,
            attention_filters: Vec::new(),
            predetermined_navigation_masking: false,
            chaos_level: 0.0,
            complexity_map: HashMap::new(),
        }
    }
    
    /// Set visual entropy source
    pub fn with_visual_entropy_source(&mut self, source: HashMap<String, f64>) {
        self.visual_entropy_source = source;
    }
    
    /// Set overload threshold
    pub fn set_overload_threshold(mut self, threshold: f64) -> Self {
        self.overload_threshold = threshold;
        self
    }
    
    /// Configure attention filtering
    pub fn configure_attention_filtering(mut self, filters: &[FilterMode]) -> Self {
        self.attention_filters.extend_from_slice(filters);
        self
    }
    
    /// Enable predetermined navigation masking
    pub fn enable_predetermined_navigation_masking(mut self, enabled: bool) -> Self {
        self.predetermined_navigation_masking = enabled;
        self
    }
    
    /// Add visual entropy
    pub fn add_visual_entropy(&mut self, source_id: String, entropy: f64) {
        self.visual_entropy_source.insert(source_id.clone(), entropy);
        self.complexity_map.insert(source_id, entropy);
        
        // Update overall chaos level
        let total_entropy: f64 = self.visual_entropy_source.values().sum();
        let entropy_count = self.visual_entropy_source.len() as f64;
        self.chaos_level = if entropy_count > 0.0 {
            (total_entropy / entropy_count).min(1.0)
        } else {
            0.0
        };
    }
    
    /// Get current chaos level
    pub fn get_chaos_level(&self) -> f64 {
        self.chaos_level
    }
    
    /// Check if information overload threshold is exceeded
    pub fn is_overloaded(&self) -> bool {
        self.chaos_level > self.overload_threshold
    }
    
    /// Generate chaos substrate for agency illusion
    pub fn generate_chaos_substrate(&self) -> HashMap<String, f64> {
        let mut substrate = HashMap::new();
        
        for (source_id, complexity) in &self.complexity_map {
            let amplified_complexity = if self.is_overloaded() {
                complexity * 1.5 // Amplify in overload conditions
            } else {
                *complexity
            };
            
            substrate.insert(source_id.clone(), amplified_complexity);
        }
        
        substrate
    }
}

impl AgencyIllusionEngine {
    /// Create new agency illusion engine
    pub fn new() -> Self {
        Self {
            chaos_substrate: Arc::new(RwLock::new(ChaosSubstrateGenerator::new())),
            focus_allocation_complexity: 0.78,
            choice_emergence_simulation: false,
            subjective_control_optimization: 0.84,
            attention_targets: Vec::new(),
            agency_state: AgencyState {
                agency_strength: 0.0,
                focus_areas: Vec::new(),
                choice_availability: 0.0,
                control_illusion: 0.0,
                selective_attention_active: false,
            },
        }
    }
    
    /// Set chaos substrate
    pub fn with_chaos_substrate(mut self, substrate: Arc<RwLock<ChaosSubstrateGenerator>>) -> Self {
        self.chaos_substrate = substrate;
        self
    }
    
    /// Set focus allocation complexity
    pub fn set_focus_allocation_complexity(mut self, complexity: f64) -> Self {
        self.focus_allocation_complexity = complexity;
        self
    }
    
    /// Enable choice emergence simulation
    pub fn enable_choice_emergence_simulation(mut self, enabled: bool) -> Self {
        self.choice_emergence_simulation = enabled;
        self
    }
    
    /// Optimize for subjective control feeling
    pub fn optimize_for_subjective_control_feeling(mut self, level: f64) -> Self {
        self.subjective_control_optimization = level;
        self
    }
    
    /// Process agency illusion
    pub async fn process_agency_illusion(&mut self) -> AutobahnResult<AgencyState> {
        let chaos_level = {
            let substrate = self.chaos_substrate.read().await;
            substrate.get_chaos_level()
        };
        
        // Agency strength increases with chaos level but requires filtering
        self.agency_state.agency_strength = if chaos_level > 0.5 {
            chaos_level * self.focus_allocation_complexity
        } else {
            0.1 // Minimal agency in low-chaos environments
        };
        
        // Choice availability based on information overload
        let is_overloaded = {
            let substrate = self.chaos_substrate.read().await;
            substrate.is_overloaded()
        };
        
        self.agency_state.choice_availability = if is_overloaded {
            0.9 // High choice perception when filtering is required
        } else {
            0.3 // Low choice perception in simple environments
        };
        
        // Control illusion from selective attention
        if chaos_level > 0.7 {
            self.agency_state.selective_attention_active = true;
            self.agency_state.control_illusion = self.subjective_control_optimization;
            
            // Generate focus areas
            self.agency_state.focus_areas = vec![
                "primary_visual_region".to_string(),
                "high_contrast_area".to_string(),
                "color_transition_zone".to_string(),
            ];
        } else {
            self.agency_state.selective_attention_active = false;
            self.agency_state.control_illusion = 0.2;
            self.agency_state.focus_areas.clear();
        }
        
        Ok(self.agency_state.clone())
    }
    
    /// Get current agency state
    pub fn get_agency_state(&self) -> &AgencyState {
        &self.agency_state
    }
}

impl ContextualFocusEngine {
    /// Create new contextual focus engine
    pub fn new() -> Self {
        Self {
            photosynthesis_engine: Arc::new(RwLock::new(
                EnvironmentalPhotosynthesis::new().unwrap()
            )),
            attention_competition: Vec::new(),
            choice_simulation_threshold: 0.67,
            control_illusion_amplification: false,
            focus_state: FocusState {
                primary_focus: None,
                secondary_focus: Vec::new(),
                focus_intensity: 0.0,
                attention_allocation: HashMap::new(),
                choice_emergence: false,
            },
        }
    }
    
    /// Set photosynthesis engine
    pub fn with_environmental_photosynthesis(mut self, engine: Arc<RwLock<EnvironmentalPhotosynthesis>>) -> Self {
        self.photosynthesis_engine = engine;
        self
    }
    
    /// Configure attention competition
    pub fn configure_attention_competition(mut self, targets: &[AttentionTarget]) -> Self {
        self.attention_competition.extend_from_slice(targets);
        self
    }
    
    /// Set choice simulation threshold
    pub fn set_choice_simulation_threshold(mut self, threshold: f64) -> Self {
        self.choice_simulation_threshold = threshold;
        self
    }
    
    /// Enable control illusion amplification
    pub fn enable_control_illusion_amplification(mut self, enabled: bool) -> Self {
        self.control_illusion_amplification = enabled;
        self
    }
    
    /// Process contextual focus
    pub async fn process_contextual_focus(&mut self) -> AutobahnResult<FocusState> {
        let chaos_level = {
            let engine = self.photosynthesis_engine.read().await;
            engine.get_chaos_level().await
        };
        
        // Focus intensity based on chaos level
        self.focus_state.focus_intensity = chaos_level;
        
        // Simulate attention competition
        if !self.attention_competition.is_empty() {
            let mut rng = thread_rng();
            let selected_idx = rng.gen_range(0..self.attention_competition.len());
            
            match &self.attention_competition[selected_idx] {
                AttentionTarget::HighContrastRegions { priority } => {
                    self.focus_state.primary_focus = Some("high_contrast".to_string());
                    self.focus_state.attention_allocation.insert("high_contrast".to_string(), *priority);
                }
                AttentionTarget::MovingElements { priority } => {
                    self.focus_state.primary_focus = Some("moving_elements".to_string());
                    self.focus_state.attention_allocation.insert("moving_elements".to_string(), *priority);
                }
                AttentionTarget::ColorTransitions { priority } => {
                    self.focus_state.primary_focus = Some("color_transitions".to_string());
                    self.focus_state.attention_allocation.insert("color_transitions".to_string(), *priority);
                }
                AttentionTarget::PatternEmergence { priority } => {
                    self.focus_state.primary_focus = Some("pattern_emergence".to_string());
                    self.focus_state.attention_allocation.insert("pattern_emergence".to_string(), *priority);
                }
            }
        }
        
        // Choice emergence detection
        self.focus_state.choice_emergence = chaos_level > self.choice_simulation_threshold;
        
        Ok(self.focus_state.clone())
    }
    
    /// Get current focus state
    pub fn get_focus_state(&self) -> &FocusState {
        &self.focus_state
    }
} 