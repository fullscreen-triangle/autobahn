//! Fire-Consciousness Quantum Framework: Complete Implementation
//! 
//! This module integrates the revolutionary Fire-Consciousness theory
//! with quantum biology, biological Maxwell's demons, and oscillatory bio-metabolic
//! RAG systems to create the first computational model of fire-catalyzed consciousness.

use crate::error::{AutobahnError, AutobahnResult};
use crate::oscillatory::{OscillationProfile, OscillationPhase};
use crate::quantum::{QuantumMembraneState, ENAQTProcessor};
use crate::atp::{MetabolicMode, OscillatoryATPManager};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::{PI, E};
use rand::Rng;
use chrono::{DateTime, Utc};

// ============================================================================
// QUANTUM CONSCIOUSNESS SUBSTRATE
// ============================================================================

/// Represents different ion types involved in neural quantum tunneling
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IonType {
    Hydrogen,    // H+ - Primary quantum tunneling ion (minimal mass)
    Sodium,      // Na+ - Action potential generation
    Potassium,   // K+ - Membrane potential maintenance
    Calcium,     // Ca2+ - Synaptic transmission
    Magnesium,   // Mg2+ - Enzyme cofactor and membrane stability
}

impl IonType {
    /// Returns the mass of the ion in atomic mass units
    pub fn mass(&self) -> f64 {
        match self {
            IonType::Hydrogen => 1.008,   // Lightest - highest tunneling probability
            IonType::Sodium => 22.990,
            IonType::Potassium => 39.098,
            IonType::Calcium => 40.078,
            IonType::Magnesium => 24.305,
        }
    }
    
    /// Quantum tunneling probability based on mass and energy barrier
    pub fn tunneling_probability(&self, barrier_height: f64, barrier_width: f64) -> f64 {
        // Quantum tunneling probability: P = exp(-2 * sqrt(2m(V-E)) * a / ℏ)
        let hbar = 1.054571817e-34; // Reduced Planck constant
        let mass_kg = self.mass() * 1.66053906660e-27; // Convert AMU to kg
        let energy_barrier = barrier_height * 1.602176634e-19; // Convert eV to Joules
        
        let exponent = -2.0 * (2.0 * mass_kg * energy_barrier).sqrt() * barrier_width / hbar;
        exponent.exp().min(1.0) // Cap at 1.0 for probability
    }
}

/// Quantum field state representing collective ion tunneling coherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCoherenceField {
    /// Amplitude of quantum field at each spatial location
    pub field_amplitude: Vec<f64>,
    /// Phase relationships between different field components
    pub phase_coherence: Vec<f64>,
    /// Coherence time in milliseconds
    pub coherence_time: f64,
    /// Energy density of the quantum field
    pub energy_density: f64,
    /// Ion contributions to the field
    pub ion_contributions: HashMap<IonType, f64>,
}

impl QuantumCoherenceField {
    /// Create new quantum coherence field from ion channel activity
    pub fn new(ion_channels: &[IonChannel], fire_light_intensity: f64) -> Self {
        let mut field_amplitude = vec![0.0; 1000]; // 1000 spatial points
        let mut phase_coherence = vec![0.0; 1000];
        let mut ion_contributions = HashMap::new();
        
        // Calculate field contributions from each ion channel
        for channel in ion_channels {
            let contribution = channel.quantum_field_contribution(fire_light_intensity);
            *ion_contributions.entry(channel.ion_type).or_insert(0.0) += contribution;
            
            // Add to spatial field (simplified 1D model)
            for i in 0..field_amplitude.len() {
                let distance = i as f64 / 100.0; // Convert to micrometers
                let amplitude = contribution * (-distance / 10.0).exp(); // Exponential decay
                field_amplitude[i] += amplitude;
                phase_coherence[i] += channel.phase_offset;
            }
        }
        
        // Calculate coherence time based on H+ ion dominance (fire-adapted)
        let h_contribution = ion_contributions.get(&IonType::Hydrogen).unwrap_or(&0.0);
        let coherence_time = 100.0 + (fire_light_intensity * h_contribution * 400.0); // 100-500ms range
        
        // Energy density from field amplitude
        let energy_density: f64 = field_amplitude.iter().map(|a| a * a).sum::<f64>() / field_amplitude.len() as f64;
        
        Self {
            field_amplitude,
            phase_coherence,
            coherence_time,
            energy_density,
            ion_contributions,
        }
    }
    
    /// Check if field meets consciousness threshold (Thermodynamic Consciousness Theorem)
    pub fn meets_consciousness_threshold(&self) -> bool {
        // Consciousness requires energy flux density >0.5 W/kg brain mass
        const CONSCIOUSNESS_THRESHOLD: f64 = 0.5;
        self.energy_density > CONSCIOUSNESS_THRESHOLD && self.coherence_time > 100.0
    }
    
    /// Calculate fire-light optimization factor
    pub fn fire_optimization_factor(&self, fire_wavelength: f64) -> f64 {
        // Fire light (600-700nm peak) optimally stimulates consciousness
        let optimal_wavelength = 650.0; // nanometers
        let wavelength_factor = 1.0 - ((fire_wavelength - optimal_wavelength) / 100.0).abs();
        wavelength_factor.max(0.1) // Minimum 10% efficiency
    }
}

/// Individual ion channel with quantum properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonChannel {
    pub ion_type: IonType,
    pub conductance: f64,           // Channel conductance in siemens
    pub voltage_threshold: f64,     // Activation threshold in mV
    pub phase_offset: f64,          // Quantum phase offset
    pub fire_adaptation: f64,       // Evolutionary fire adaptation factor (0-1)
}

impl IonChannel {
    /// Calculate quantum field contribution from this channel
    pub fn quantum_field_contribution(&self, fire_light_intensity: f64) -> f64 {
        let base_contribution = self.conductance * self.fire_adaptation;
        let fire_enhancement = 1.0 + (fire_light_intensity * 0.3); // Up to 30% enhancement
        let tunneling_prob = self.ion_type.tunneling_probability(0.1, 1e-9); // Typical membrane values
        
        base_contribution * fire_enhancement * tunneling_prob
    }
}

// ============================================================================
// BIOLOGICAL MAXWELL'S DEMONS (BMD) - INFORMATION CATALYSTS
// ============================================================================

/// BMD specialization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BMDSpecialization {
    FireRecognition,
    AgencyDetection,
    SpatialMemory,
    TemporalPlanning,
}

/// Information filter for BMD processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFilter {
    pub filter_weights: Vec<f64>,
    pub threshold: f64,
    pub adaptation_rate: f64,
}

impl InformationFilter {
    pub fn fire_recognition() -> Self {
        Self {
            filter_weights: vec![0.9, 0.8, 0.7, 0.6, 0.5], // Tuned for fire patterns
            threshold: 0.7,
            adaptation_rate: 0.1,
        }
    }
    
    pub fn threat_response() -> Self {
        Self {
            filter_weights: vec![0.8, 0.9, 0.7, 0.8, 0.6],
            threshold: 0.6,
            adaptation_rate: 0.15,
        }
    }
    
    pub fn agency_patterns() -> Self {
        Self {
            filter_weights: vec![0.7, 0.8, 0.9, 0.7, 0.8],
            threshold: 0.65,
            adaptation_rate: 0.12,
        }
    }
    
    pub fn social_response() -> Self {
        Self {
            filter_weights: vec![0.6, 0.7, 0.8, 0.9, 0.7],
            threshold: 0.55,
            adaptation_rate: 0.08,
        }
    }
    
    pub fn spatial_patterns() -> Self {
        Self {
            filter_weights: vec![0.8, 0.6, 0.7, 0.8, 0.9],
            threshold: 0.6,
            adaptation_rate: 0.1,
        }
    }
    
    pub fn navigation_response() -> Self {
        Self {
            filter_weights: vec![0.7, 0.8, 0.6, 0.9, 0.8],
            threshold: 0.65,
            adaptation_rate: 0.09,
        }
    }
    
    pub fn temporal_patterns() -> Self {
        Self {
            filter_weights: vec![0.9, 0.7, 0.8, 0.6, 0.7],
            threshold: 0.7,
            adaptation_rate: 0.11,
        }
    }
    
    pub fn planning_response() -> Self {
        Self {
            filter_weights: vec![0.8, 0.9, 0.7, 0.8, 0.6],
            threshold: 0.68,
            adaptation_rate: 0.13,
        }
    }
}

/// Biological Maxwell's Demon as described by Mizraji
/// Functions as "information catalyst" (iCat) that amplifies processed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalMaxwellDemon {
    /// Input filter - selects specific patterns from environmental input
    pub input_filter: InformationFilter,
    /// Output filter - channels responses toward particular targets
    pub output_filter: InformationFilter,
    /// Catalytic efficiency - how much the BMD amplifies information processing
    pub catalytic_efficiency: f64,
    /// Memory patterns for associative processing
    pub memory_patterns: Vec<(Vec<f64>, Vec<f64>)>, // (input_pattern, output_pattern) pairs
    /// Specialization for fire-related processing
    pub fire_specialization: f64,
}

impl BiologicalMaxwellDemon {
    /// Create new BMD with specified specialization
    pub fn new(specialization: BMDSpecialization) -> Self {
        let (input_filter, output_filter, fire_spec) = match specialization {
            BMDSpecialization::FireRecognition => {
                (InformationFilter::fire_recognition(), InformationFilter::threat_response(), 0.9)
            },
            BMDSpecialization::AgencyDetection => {
                (InformationFilter::agency_patterns(), InformationFilter::social_response(), 0.7)
            },
            BMDSpecialization::SpatialMemory => {
                (InformationFilter::spatial_patterns(), InformationFilter::navigation_response(), 0.5)
            },
            BMDSpecialization::TemporalPlanning => {
                (InformationFilter::temporal_patterns(), InformationFilter::planning_response(), 0.6)
            },
        };
        
        Self {
            input_filter,
            output_filter,
            catalytic_efficiency: 2.5, // Default amplification factor
            memory_patterns: Vec::new(),
            fire_specialization: fire_spec,
        }
    }
    
    /// Process input through BMD filters and amplify information
    pub fn process_information(&mut self, input: &[f64], fire_context: f64) -> Vec<f64> {
        // Apply input filter
        let filtered_input = self.apply_filter(&self.input_filter.clone(), input);
        
        // Apply fire specialization enhancement
        let fire_enhanced = self.apply_fire_enhancement(&filtered_input, fire_context);
        
        // Catalytic amplification
        let amplified = fire_enhanced.iter()
            .map(|&x| x * self.catalytic_efficiency)
            .collect::<Vec<f64>>();
        
        // Apply output filter
        let output = self.apply_filter(&self.output_filter.clone(), &amplified);
        
        // Update memory patterns
        self.update_memory_patterns(&filtered_input, &output);
        
        output
    }
    
    fn apply_filter(&self, filter: &InformationFilter, input: &[f64]) -> Vec<f64> {
        input.iter()
            .zip(filter.filter_weights.iter().cycle())
            .map(|(&x, &w)| if x * w > filter.threshold { x * w } else { 0.0 })
            .collect()
    }
    
    fn apply_fire_enhancement(&self, input: &[f64], fire_context: f64) -> Vec<f64> {
        let enhancement_factor = 1.0 + (self.fire_specialization * fire_context);
        input.iter()
            .map(|&x| x * enhancement_factor)
            .collect()
    }
    
    fn update_memory_patterns(&mut self, input: &[f64], output: &[f64]) {
        // Add new pattern to memory (limit to 1000 patterns)
        if self.memory_patterns.len() < 1000 {
            self.memory_patterns.push((input.to_vec(), output.to_vec()));
        } else {
            // Replace oldest pattern
            let index = rand::thread_rng().gen_range(0..self.memory_patterns.len());
            self.memory_patterns[index] = (input.to_vec(), output.to_vec());
        }
    }
}

// ============================================================================
// FIRE-CIRCLE ENVIRONMENTS AND AGENCY EMERGENCE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireEnvironment {
    /// Light intensity from fire source (0.0 to 1.0)
    pub fire_intensity: f64,
    /// Wavelength distribution of fire light
    pub wavelength_spectrum: Vec<(f64, f64)>, // (wavelength_nm, intensity)
    /// Environmental temperature from fire
    pub temperature_kelvin: f64,
    /// Darkness fear activation level
    pub darkness_level: f64,
    /// Witness space radius for consciousness emergence
    pub witness_space_radius: f64,
}

impl FireEnvironment {
    pub fn new(fire_intensity: f64) -> Self {
        // Generate fire spectrum (peak at 650nm)
        let mut wavelength_spectrum = Vec::new();
        for wavelength in (400..=800).step_by(10) {
            let intensity = fire_intensity * Self::fire_spectrum_intensity(wavelength as f64);
            wavelength_spectrum.push((wavelength as f64, intensity));
        }
        
        Self {
            fire_intensity,
            wavelength_spectrum,
            temperature_kelvin: 300.0 + fire_intensity * 500.0, // 300-800K range
            darkness_level: 1.0 - fire_intensity, // Inverse relationship
            witness_space_radius: 2.0 + fire_intensity * 3.0, // 2-5 meter radius
        }
    }
    
    fn fire_spectrum_intensity(wavelength: f64) -> f64 {
        // Blackbody radiation approximation for fire
        let peak_wavelength = 650.0; // nm
        let width = 100.0; // nm
        let gaussian = (-(wavelength - peak_wavelength).powi(2) / (2.0 * width.powi(2))).exp();
        gaussian
    }
    
    /// Calculate consciousness emergence probability
    pub fn consciousness_emergence_probability(&self) -> f64 {
        // Fire intensity factor
        let intensity_factor = self.fire_intensity.powf(0.5); // Square root relationship
        
        // Temperature factor (optimal around 310K body temperature)
        let temp_factor = if self.temperature_kelvin > 280.0 && self.temperature_kelvin < 340.0 {
            1.0 - ((self.temperature_kelvin - 310.0) / 30.0).abs()
        } else {
            0.1
        };
        
        // Darkness fear factor (consciousness malfunction without light)
        let darkness_factor = 1.0 - self.darkness_level.powi(2);
        
        (intensity_factor * temp_factor * darkness_factor).max(0.0).min(1.0)
    }
}

// ============================================================================
// FIRE CONSCIOUSNESS ENGINE
// ============================================================================

#[derive(Debug)]
pub struct FireConsciousnessEngine {
    /// Quantum coherence field from ion channels
    pub quantum_field: QuantumCoherenceField,
    /// Biological Maxwell's Demons for information processing
    pub bmds: Vec<BiologicalMaxwellDemon>,
    /// Fire environment state
    pub fire_environment: FireEnvironment,
    /// Ion channels for quantum tunneling
    pub ion_channels: Vec<IonChannel>,
    /// Consciousness level (0.0 to 1.0)
    pub consciousness_level: f64,
    /// Fire recognition strength
    pub fire_recognition_strength: f64,
    /// Agency detection capability
    pub agency_detection_strength: f64,
    /// Evolutionary time in millions of years ago
    pub evolutionary_time_mya: f64,
    /// ATP manager for metabolic processes
    pub atp_manager: OscillatoryATPManager,
}

impl FireConsciousnessEngine {
    pub fn new(evolutionary_time_mya: f64) -> AutobahnResult<Self> {
        let mut rng = rand::thread_rng();
        
        // Create ion channels with fire adaptation based on evolutionary time
        let fire_adaptation_factor = Self::calculate_fire_adaptation(evolutionary_time_mya);
        let ion_channels = vec![
            IonChannel {
                ion_type: IonType::Hydrogen,
                conductance: 0.1 + rng.gen::<f64>() * 0.05,
                voltage_threshold: -60.0,
                phase_offset: rng.gen::<f64>() * 2.0 * PI,
                fire_adaptation: fire_adaptation_factor * (0.8 + rng.gen::<f64>() * 0.2),
            },
            IonChannel {
                ion_type: IonType::Sodium,
                conductance: 0.05 + rng.gen::<f64>() * 0.03,
                voltage_threshold: -55.0,
                phase_offset: rng.gen::<f64>() * 2.0 * PI,
                fire_adaptation: fire_adaptation_factor * (0.6 + rng.gen::<f64>() * 0.3),
            },
            IonChannel {
                ion_type: IonType::Potassium,
                conductance: 0.08 + rng.gen::<f64>() * 0.04,
                voltage_threshold: -80.0,
                phase_offset: rng.gen::<f64>() * 2.0 * PI,
                fire_adaptation: fire_adaptation_factor * (0.5 + rng.gen::<f64>() * 0.4),
            },
            IonChannel {
                ion_type: IonType::Calcium,
                conductance: 0.03 + rng.gen::<f64>() * 0.02,
                voltage_threshold: -40.0,
                phase_offset: rng.gen::<f64>() * 2.0 * PI,
                fire_adaptation: fire_adaptation_factor * (0.7 + rng.gen::<f64>() * 0.2),
            },
            IonChannel {
                ion_type: IonType::Magnesium,
                conductance: 0.02 + rng.gen::<f64>() * 0.01,
                voltage_threshold: -70.0,
                phase_offset: rng.gen::<f64>() * 2.0 * PI,
                fire_adaptation: fire_adaptation_factor * (0.4 + rng.gen::<f64>() * 0.3),
            },
        ];
        
        // Create fire environment
        let fire_intensity = 0.7; // Default moderate fire
        let fire_environment = FireEnvironment::new(fire_intensity);
        
        // Create quantum coherence field
        let quantum_field = QuantumCoherenceField::new(&ion_channels, fire_intensity);
        
        // Create specialized BMDs
        let bmds = vec![
            BiologicalMaxwellDemon::new(BMDSpecialization::FireRecognition),
            BiologicalMaxwellDemon::new(BMDSpecialization::AgencyDetection),
            BiologicalMaxwellDemon::new(BMDSpecialization::SpatialMemory),
            BiologicalMaxwellDemon::new(BMDSpecialization::TemporalPlanning),
        ];
        
        // Initialize ATP manager
        let atp_manager = OscillatoryATPManager::new()?;
        
        Ok(Self {
            quantum_field,
            bmds,
            fire_environment,
            ion_channels,
            consciousness_level: fire_environment.consciousness_emergence_probability(),
            fire_recognition_strength: fire_adaptation_factor,
            agency_detection_strength: fire_adaptation_factor * 0.8,
            evolutionary_time_mya,
            atp_manager,
        })
    }
    
    fn calculate_fire_adaptation(evolutionary_time_mya: f64) -> f64 {
        // Fire control emerged ~2 MYA, full adaptation by ~0.5 MYA
        if evolutionary_time_mya >= 2.0 {
            0.1 // Pre-fire minimal adaptation
        } else if evolutionary_time_mya >= 0.5 {
            // Linear increase from 0.1 to 0.9 between 2 MYA and 0.5 MYA
            0.1 + (2.0 - evolutionary_time_mya) * 0.8 / 1.5
        } else {
            // Full fire adaptation
            0.9 + (0.5 - evolutionary_time_mya) * 0.1 / 0.5
        }
    }
    
    /// Process input through fire consciousness system
    pub async fn process_input(&mut self, input_data: &[f64]) -> AutobahnResult<FireConsciousnessResponse> {
        // Update quantum field based on current fire environment
        self.quantum_field = QuantumCoherenceField::new(&self.ion_channels, self.fire_environment.fire_intensity);
        
        // Process through BMDs
        let mut bmd_responses = Vec::new();
        for bmd in &mut self.bmds {
            let response = bmd.process_information(input_data, self.fire_environment.fire_intensity);
            bmd_responses.push(response);
        }
        
        // Update consciousness level
        let emergence_prob = self.fire_environment.consciousness_emergence_probability();
        self.consciousness_level = (self.consciousness_level * 0.9 + emergence_prob * 0.1).min(1.0);
        
        // Fire recognition processing
        let fire_recognition = self.process_fire_recognition(input_data).await?;
        
        // Agency detection processing
        let agency_detection = self.process_agency_detection(input_data).await?;
        
        // ATP consumption for processing
        let atp_cost = self.calculate_processing_cost(input_data.len());
        self.atp_manager.consume_atp(atp_cost)?;
        
        Ok(FireConsciousnessResponse {
            consciousness_level: self.consciousness_level,
            fire_recognition: fire_recognition,
            agency_detection: agency_detection,
            quantum_coherence_time: self.quantum_field.coherence_time,
            quantum_energy_density: self.quantum_field.energy_density,
            bmd_responses,
            meets_consciousness_threshold: self.quantum_field.meets_consciousness_threshold(),
            fire_adaptation_strength: Self::calculate_fire_adaptation(self.evolutionary_time_mya),
            processing_timestamp: Utc::now(),
        })
    }
    
    async fn process_fire_recognition(&mut self, input_data: &[f64]) -> AutobahnResult<FireRecognitionResponse> {
        // Extract fire-like patterns from input
        let fire_patterns = self.extract_fire_patterns(input_data);
        
        // Underwater fireplace paradox test
        let underwater_test = self.test_underwater_fireplace_paradox().await?;
        
        // Update fire recognition strength
        if !fire_patterns.is_empty() {
            let pattern_strength = fire_patterns.iter().sum::<f64>() / fire_patterns.len() as f64;
            self.fire_recognition_strength = (self.fire_recognition_strength * 0.95 + pattern_strength * 0.05).min(1.0);
        }
        
        Ok(FireRecognitionResponse {
            fire_detected: self.fire_recognition_strength > 0.5,
            recognition_strength: self.fire_recognition_strength,
            fire_patterns,
            underwater_paradox_active: underwater_test.hardwired_override_active,
            darkness_fear_level: self.fire_environment.darkness_level,
        })
    }
    
    async fn process_agency_detection(&mut self, input_data: &[f64]) -> AutobahnResult<AgencyDetection> {
        // Extract agency-like patterns from input
        let agency_patterns = self.extract_agency_patterns(input_data);
        
        // Update agency detection strength
        if !agency_patterns.is_empty() {
            let pattern_strength = agency_patterns.iter().sum::<f64>() / agency_patterns.len() as f64;
            self.agency_detection_strength = (self.agency_detection_strength * 0.95 + pattern_strength * 0.05).min(1.0);
        }
        
        Ok(AgencyDetection {
            agency_detected: self.agency_detection_strength > 0.4,
            detection_strength: self.agency_detection_strength,
            agency_patterns,
            individual_consciousness_emergence: self.consciousness_level > 0.6,
        })
    }
    
    fn extract_fire_patterns(&self, input_data: &[f64]) -> Vec<f64> {
        // Simple fire pattern detection (wavelength-like analysis)
        let mut patterns = Vec::new();
        
        for (i, &value) in input_data.iter().enumerate() {
            // Look for fire-like oscillatory patterns
            if i > 2 && i < input_data.len() - 2 {
                let local_pattern = &input_data[i-2..=i+2];
                let fire_score = self.calculate_fire_pattern_score(local_pattern);
                if fire_score > 0.3 {
                    patterns.push(fire_score);
                }
            }
        }
        
        patterns
    }
    
    fn calculate_fire_pattern_score(&self, pattern: &[f64]) -> f64 {
        // Fire patterns: flickering (high variance), warm spectrum (mid-range values)
        let mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
        let variance = pattern.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / pattern.len() as f64;
        
        // Fire score combines variance (flickering) and mid-range values (warmth)
        let flicker_score = (variance * 2.0).min(1.0);
        let warmth_score = if mean > 0.3 && mean < 0.8 { 1.0 - (mean - 0.55).abs() * 2.0 } else { 0.0 };
        
        (flicker_score + warmth_score) / 2.0
    }
    
    fn extract_agency_patterns(&self, input_data: &[f64]) -> Vec<f64> {
        // Simple agency pattern detection (intentional-like behavior)
        let mut patterns = Vec::new();
        
        for i in 3..input_data.len() - 3 {
            let window = &input_data[i-3..=i+3];
            let agency_score = self.calculate_agency_pattern_score(window);
            if agency_score > 0.4 {
                patterns.push(agency_score);
            }
        }
        
        patterns
    }
    
    fn calculate_agency_pattern_score(&self, pattern: &[f64]) -> f64 {
        // Agency patterns: directional trends, purposeful changes
        let mut trend_score = 0.0;
        let mut purpose_score = 0.0;
        
        // Calculate trend (directional movement)
        for i in 1..pattern.len() {
            let diff = pattern[i] - pattern[i-1];
            trend_score += diff.abs();
        }
        trend_score = (trend_score / (pattern.len() - 1) as f64).min(1.0);
        
        // Calculate purpose (non-random structure)
        let mean = pattern.iter().sum::<f64>() / pattern.len() as f64;
        let structure = pattern.iter().enumerate()
            .map(|(i, &x)| (x - mean) * (i as f64 - (pattern.len() - 1) as f64 / 2.0))
            .sum::<f64>().abs();
        purpose_score = (structure / pattern.len() as f64).min(1.0);
        
        (trend_score + purpose_score) / 2.0
    }
    
    fn calculate_processing_cost(&self, input_size: usize) -> f64 {
        // ATP cost scales with input size and consciousness level
        let base_cost = input_size as f64 * 0.1;
        let consciousness_multiplier = 1.0 + self.consciousness_level;
        let quantum_multiplier = 1.0 + (self.quantum_field.energy_density * 0.5);
        
        base_cost * consciousness_multiplier * quantum_multiplier
    }
    
    /// Test the underwater fireplace paradox
    pub async fn test_underwater_fireplace_paradox(&mut self) -> AutobahnResult<UnderwaterFireplaceTest> {
        // Simulate underwater environment (no fire possible)
        let underwater_environment = FireEnvironment {
            fire_intensity: 0.0,
            wavelength_spectrum: vec![(470.0, 0.8)], // Blue underwater light
            temperature_kelvin: 277.0, // Cold water
            darkness_level: 0.8, // Dim underwater
            witness_space_radius: 1.0,
        };
        
        // Test if fire recognition still activates (hardwired override)
        let fake_fire_input = vec![0.6, 0.7, 0.5, 0.8, 0.6]; // Fire-like pattern
        let fire_patterns = self.extract_fire_patterns(&fake_fire_input);
        
        // Hardwired fire recognition should override logic
        let hardwired_override_active = !fire_patterns.is_empty() && self.fire_recognition_strength > 0.3;
        
        Ok(UnderwaterFireplaceTest {
            underwater_environment_active: true,
            fire_logically_impossible: true,
            hardwired_override_active,
            fire_recognition_strength: self.fire_recognition_strength,
            logic_override_strength: if hardwired_override_active { 0.8 } else { 0.0 },
        })
    }
    
    /// Update fire environment
    pub fn update_fire_environment(&mut self, new_intensity: f64) -> AutobahnResult<()> {
        self.fire_environment = FireEnvironment::new(new_intensity);
        self.quantum_field = QuantumCoherenceField::new(&self.ion_channels, new_intensity);
        Ok(())
    }
    
    /// Get current consciousness metrics
    pub fn get_consciousness_metrics(&self) -> ConsciousnessMetrics {
        ConsciousnessMetrics {
            consciousness_level: self.consciousness_level,
            fire_recognition_strength: self.fire_recognition_strength,
            agency_detection_strength: self.agency_detection_strength,
            quantum_coherence_time: self.quantum_field.coherence_time,
            quantum_energy_density: self.quantum_field.energy_density,
            meets_consciousness_threshold: self.quantum_field.meets_consciousness_threshold(),
            fire_adaptation_factor: Self::calculate_fire_adaptation(self.evolutionary_time_mya),
            evolutionary_time_mya: self.evolutionary_time_mya,
        }
    }
}

// ============================================================================
// RESPONSE STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireConsciousnessResponse {
    pub consciousness_level: f64,
    pub fire_recognition: FireRecognitionResponse,
    pub agency_detection: AgencyDetection,
    pub quantum_coherence_time: f64,
    pub quantum_energy_density: f64,
    pub bmd_responses: Vec<Vec<f64>>,
    pub meets_consciousness_threshold: bool,
    pub fire_adaptation_strength: f64,
    pub processing_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireRecognitionResponse {
    pub fire_detected: bool,
    pub recognition_strength: f64,
    pub fire_patterns: Vec<f64>,
    pub underwater_paradox_active: bool,
    pub darkness_fear_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgencyDetection {
    pub agency_detected: bool,
    pub detection_strength: f64,
    pub agency_patterns: Vec<f64>,
    pub individual_consciousness_emergence: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderwaterFireplaceTest {
    pub underwater_environment_active: bool,
    pub fire_logically_impossible: bool,
    pub hardwired_override_active: bool,
    pub fire_recognition_strength: f64,
    pub logic_override_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    pub consciousness_level: f64,
    pub fire_recognition_strength: f64,
    pub agency_detection_strength: f64,
    pub quantum_coherence_time: f64,
    pub quantum_energy_density: f64,
    pub meets_consciousness_threshold: bool,
    pub fire_adaptation_factor: f64,
    pub evolutionary_time_mya: f64,
} 