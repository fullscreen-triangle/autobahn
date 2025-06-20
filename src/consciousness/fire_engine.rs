//! Fire-Consciousness Quantum Framework Implementation
//! 
//! This module implements the complete fire-evolved consciousness theory integrating:
//! - Quantum ion tunneling in neural networks (H+, Na+, K+, Ca2+, Mg2+)
//! - Biological Maxwell's Demons (BMDs) as information catalysts
//! - Fire-circle environments and agency emergence
//! - Thermodynamic consciousness thresholds
//! - All nine theoretical frameworks for complete consciousness

use crate::error::{AutobahnError, AutobahnResult};
use crate::oscillatory::{OscillationProfile, UniversalOscillator};
use crate::quantum::{QuantumMembraneState, ENAQTProcessor};
use crate::consciousness::{ConsciousnessEmergenceEngine, ConsciousExperience};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use std::f64::consts::{PI, E};

// ============================================================================
// QUANTUM ION TUNNELING SUBSTRATE
// ============================================================================

/// Ion types involved in neural quantum tunneling for consciousness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IonType {
    /// H+ - Primary quantum tunneling ion (minimal mass, highest probability)
    Hydrogen,
    /// Na+ - Action potential generation
    Sodium,
    /// K+ - Membrane potential maintenance  
    Potassium,
    /// Ca2+ - Synaptic transmission
    Calcium,
    /// Mg2+ - Enzyme cofactor and membrane stability
    Magnesium,
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
    pub fn tunneling_probability(&self, barrier_height_ev: f64, barrier_width_nm: f64) -> f64 {
        // Quantum tunneling: P = exp(-2 * sqrt(2m(V-E)) * a / ℏ)
        const HBAR: f64 = 1.054571817e-34; // Reduced Planck constant
        const AMU_TO_KG: f64 = 1.66053906660e-27;
        const EV_TO_JOULES: f64 = 1.602176634e-19;
        const NM_TO_M: f64 = 1e-9;
        
        let mass_kg = self.mass() * AMU_TO_KG;
        let energy_barrier = barrier_height_ev * EV_TO_JOULES;
        let width_m = barrier_width_nm * NM_TO_M;
        
        let exponent = -2.0 * (2.0 * mass_kg * energy_barrier).sqrt() * width_m / HBAR;
        exponent.exp().min(1.0)
    }
    
    /// Fire-light enhancement factor for this ion type
    pub fn fire_light_enhancement(&self, wavelength_nm: f64) -> f64 {
        match self {
            IonType::Hydrogen => {
                // H+ most sensitive to fire light (600-700nm)
                let optimal_wavelength = 650.0;
                let enhancement = 1.0 + 0.4 * (-(wavelength_nm - optimal_wavelength).powi(2) / (50.0_f64.powi(2))).exp();
                enhancement
            },
            IonType::Sodium => {
                // Moderate fire enhancement
                1.0 + 0.2 * (-(wavelength_nm - 650.0).powi(2) / (100.0_f64.powi(2))).exp()
            },
            _ => 1.0 + 0.1 * (-(wavelength_nm - 650.0).powi(2) / (150.0_f64.powi(2))).exp(),
        }
    }
}

/// Quantum coherence field representing collective ion tunneling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCoherenceField {
    /// Spatial field amplitude distribution
    pub field_amplitude: Vec<f64>,
    /// Phase coherence across field
    pub phase_coherence: Vec<f64>,
    /// Coherence time in milliseconds
    pub coherence_time_ms: f64,
    /// Energy density of quantum field
    pub energy_density: f64,
    /// Ion contributions to field
    pub ion_contributions: HashMap<IonType, f64>,
    /// Fire-light optimization factor
    pub fire_optimization: f64,
    /// Consciousness threshold status
    pub meets_consciousness_threshold: bool,
}

impl QuantumCoherenceField {
    /// Create quantum coherence field from ion channels and fire environment
    pub fn new(ion_channels: &[IonChannel], fire_environment: &FireEnvironment) -> AutobahnResult<Self> {
        let spatial_points = 1000;
        let mut field_amplitude = vec![0.0; spatial_points];
        let mut phase_coherence = vec![0.0; spatial_points];
        let mut ion_contributions = HashMap::new();
        
        // Calculate field contributions from each ion channel
        for channel in ion_channels {
            let fire_enhancement = channel.ion_type.fire_light_enhancement(fire_environment.dominant_wavelength_nm);
            let contribution = channel.quantum_field_contribution(fire_environment.intensity, fire_enhancement)?;
            
            *ion_contributions.entry(channel.ion_type).or_insert(0.0) += contribution;
            
            // Add to spatial field with exponential decay
            for i in 0..field_amplitude.len() {
                let distance_um = i as f64 / 100.0;
                let amplitude = contribution * (-distance_um / 10.0).exp();
                field_amplitude[i] += amplitude;
                phase_coherence[i] += channel.phase_offset;
            }
        }
        
        // Calculate coherence time (H+ dominance extends coherence)
        let h_contribution = ion_contributions.get(&IonType::Hydrogen).unwrap_or(&0.0);
        let coherence_time_ms = 100.0 + (fire_environment.intensity * h_contribution * 400.0);
        
        // Energy density from field amplitude
        let energy_density: f64 = field_amplitude.iter()
            .map(|a| a * a)
            .sum::<f64>() / field_amplitude.len() as f64;
        
        // Fire optimization factor
        let fire_optimization = fire_environment.consciousness_enhancement_factor()?;
        
        // Check consciousness threshold (Thermodynamic Consciousness Theorem)
        const CONSCIOUSNESS_THRESHOLD: f64 = 0.5; // W/kg brain mass
        let meets_consciousness_threshold = energy_density > CONSCIOUSNESS_THRESHOLD 
            && coherence_time_ms > 100.0
            && fire_optimization > 0.7;
        
        Ok(Self {
            field_amplitude,
            phase_coherence,
            coherence_time_ms,
            energy_density,
            ion_contributions,
            fire_optimization,
            meets_consciousness_threshold,
        })
    }
    
    /// Calculate quantum enhancement factor for consciousness processing
    pub fn quantum_enhancement_factor(&self) -> f64 {
        if self.meets_consciousness_threshold {
            1.0 + (self.energy_density * self.fire_optimization * 0.3)
        } else {
            0.8 // Reduced efficiency without quantum consciousness
        }
    }
}

/// Evolutionary timeline of fire-consciousness coupling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryTimeline {
    /// Current time in millions of years ago (MYA)
    pub time_mya: f64,
    /// Hominid cognitive complexity (bits)
    pub cognitive_complexity: f64,
    /// Fire adaptation level (0.0 to 1.0)
    pub fire_adaptation: f64,
    /// Consciousness emergence level (0.0 to 1.0)
    pub consciousness_level: f64,
}

impl EvolutionaryTimeline {
    /// Create timeline for specific period
    pub fn new(time_mya: f64) -> Self {
        let (cognitive_complexity, fire_adaptation, consciousness_level) = match time_mya {
            t if t > 2.0 => (6.0, 0.1, 0.0),  // Pre-conscious fire interaction
            t if t > 1.5 => (8.5, 0.4, 0.2),  // Quantum-BMD coupling begins
            t if t > 1.0 => (10.2, 0.7, 0.5), // Agency recognition emerges
            t if t > 0.5 => (11.8, 0.9, 0.8), // Cultural transmission
            _ => (12.5, 1.0, 1.0),             // Modern human consciousness
        };
        
        Self {
            time_mya,
            cognitive_complexity,
            fire_adaptation,
            consciousness_level,
        }
    }
    
    /// Calculate evolutionary pressure for fire adaptation
    pub fn fire_adaptation_pressure(&self, fire_env: &FireEnvironment) -> f64 {
        let exposure_prob = fire_env.fire_exposure_probability(10.0, 180.0); // 10 km² territory, 180-day dry season
        let survival_benefit = self.fire_adaptation * 0.3; // Up to 30% survival increase
        let cognitive_benefit = (self.cognitive_complexity - 8.0).max(0.0) / 4.0; // Threshold effect at 8 bits
        
        exposure_prob * (survival_benefit + cognitive_benefit)
    }
}

/// Main consciousness system integrating all components
#[derive(Debug)]
pub struct ConsciousnessSystem {
    /// Quantum coherence field from ion channels
    pub quantum_field: QuantumCoherenceField,
    /// Collection of biological Maxwell's demons
    pub bmds: Vec<BiologicalMaxwellDemon>,
    /// Current fire environment
    pub fire_environment: FireEnvironment,
    /// Evolutionary timeline context
    pub timeline: EvolutionaryTimeline,
    /// Current consciousness level (0.0 to 1.0)
    pub consciousness_level: f64,
    /// Agency recognition capability
    pub agency_recognition: f64,
    /// Darkness fear level (uniquely human)
    pub darkness_fear: f64,
}

impl ConsciousnessSystem {
    /// Create new consciousness system for specific evolutionary period
    pub fn new(time_mya: f64) -> AutobahnResult<Self> {
        // Create ion channels with fire adaptations
        let mut ion_channels = Vec::new();
        let fire_adaptation = if time_mya < 1.5 { 0.8 } else { 0.3 }; // Higher adaptation in later periods
        
        for _ in 0..1000 { // 1000 ion channels (simplified)
            ion_channels.push(IonChannel {
                ion_type: IonType::Hydrogen, // Primary consciousness ion
                conductance_siemens: 1e-12, // 1 picosiemen
                voltage_threshold_mv: -55.0, // mV
                phase_offset: rand::random::<f64>() * 2.0 * std::f64::consts::PI,
                fire_adaptation_factor: fire_adaptation,
                spatial_location: (
                    rand::random::<f64>() * 100.0,
                    rand::random::<f64>() * 100.0,
                    rand::random::<f64>() * 100.0,
                ),
            });
        }
        
        // Create fire environment
        let fire_environment = FireEnvironment::olduvai_ecosystem();
        
        // Create quantum field
        let quantum_field = QuantumCoherenceField::new(&ion_channels, &fire_environment)?;
        
        // Create specialized BMDs
        let bmds = vec![
            BiologicalMaxwellDemon::new(BMDSpecialization::FireRecognition),
            BiologicalMaxwellDemon::new(BMDSpecialization::AgencyDetection),
            BiologicalMaxwellDemon::new(BMDSpecialization::SpatialMemory),
            BiologicalMaxwellDemon::new(BMDSpecialization::TemporalPlanning),
            BiologicalMaxwellDemon::new(BMDSpecialization::SocialCoordination),
            BiologicalMaxwellDemon::new(BMDSpecialization::ThreatAssessment),
        ];
        
        let timeline = EvolutionaryTimeline::new(time_mya);
        
        // Calculate initial consciousness level
        let consciousness_level = if quantum_field.meets_consciousness_threshold {
            timeline.consciousness_level * fire_environment.consciousness_enhancement_factor()?
        } else {
            0.0
        };
        
        // Agency recognition emerges around 1.0 MYA
        let agency_recognition = if time_mya < 1.0 { 0.8 } else { 0.1 };
        
        // Darkness fear emerges with fire-dependent consciousness
        let darkness_fear = consciousness_level * 0.9; // Strong correlation
        
        Ok(Self {
            quantum_field,
            bmds,
            fire_environment,
            timeline,
            consciousness_level,
            agency_recognition,
            darkness_fear,
        })
    }
    
    /// Process environmental input through complete consciousness system
    pub async fn process_input(&mut self, input: &[f64]) -> AutobahnResult<ConsciousnessResponse> {
        // Step 1: Check for fire patterns (Underwater Fireplace Paradox)
        let fire_recognition = self.recognize_fire_patterns(input).await?;
        
        // Step 2: Process through BMDs with quantum enhancement
        let mut bmd_outputs = Vec::new();
        for bmd in &mut self.bmds {
            let output = bmd.process_information(input, &self.quantum_field, &self.fire_environment).await?;
            bmd_outputs.push(output);
        }
        
        // Step 3: Integrate BMD outputs
        let integrated_response = self.integrate_bmd_outputs(&bmd_outputs)?;
        
        // Step 4: Apply consciousness-level modulation
        let conscious_response = self.apply_consciousness_modulation(&integrated_response);
        
        // Step 5: Check for agency recognition
        let agency_detected = self.detect_individual_agency(input).await?;
        
        // Step 6: Calculate darkness response (if applicable)
        let darkness_response = self.calculate_darkness_response(input);
        
        Ok(ConsciousnessResponse {
            fire_recognition,
            conscious_processing: conscious_response,
            agency_detection: agency_detected,
            darkness_fear_activation: darkness_response,
            quantum_coherence: self.quantum_field.meets_consciousness_threshold,
            consciousness_level: self.consciousness_level,
        })
    }
    
    /// Recognize fire patterns (hardwired recognition that overrides logic)
    async fn recognize_fire_patterns(&self, input: &[f64]) -> AutobahnResult<FireRecognitionResponse> {
        // Underwater Fireplace Paradox: Fire recognition overrides logical impossibility
        let fire_signature = [0.8, 0.6, 0.4, 0.9, 0.7]; // Simplified fire pattern
        let similarity = cosine_similarity(input, &fire_signature)?;
        
        let recognition_strength = similarity * 1.2; // Enhanced recognition
        let logical_override = recognition_strength > 0.5; // Overrides logic if strong enough
        
        // Check for impossible fire contexts (underwater, vacuum, etc.)
        let impossible_context = self.detect_impossible_fire_context(input);
        let human_attribution = recognition_strength > 0.7; // Strong fires attributed to humans
        
        Ok(FireRecognitionResponse {
            recognition_strength,
            logical_override,
            impossible_context,
            human_attribution,
            fire_signature_match: similarity,
        })
    }
    
    fn detect_impossible_fire_context(&self, input: &[f64]) -> bool {
        // Simplified detection of impossible contexts
        let water_signature = [0.2, 0.8, 0.9, 0.1, 0.3];
        let water_similarity = cosine_similarity(input, &water_signature).unwrap_or(0.0);
        water_similarity > 0.6 // High water similarity = impossible fire context
    }
    
    async fn detect_individual_agency(&self, input: &[f64]) -> AutobahnResult<AgencyDetection> {
        let agency_patterns = [0.7, 0.5, 0.8, 0.6, 0.9]; // Simplified agency signature
        let similarity = cosine_similarity(input, &agency_patterns)?;
        
        let agency_detected = similarity > 0.6 && self.agency_recognition > 0.5;
        let agency_strength = similarity * self.agency_recognition;
        
        let individual_signatures = if agency_detected {
            vec!["Individual_1".to_string(), "Individual_2".to_string()]
        } else {
            vec![]
        };
        
        let witness_context_active = self.fire_environment.group_size >= 3; // Need witnesses
        
        Ok(AgencyDetection {
            agency_detected,
            agency_strength,
            individual_signatures,
            witness_context_active,
        })
    }
    
    fn calculate_darkness_response(&self, input: &[f64]) -> f64 {
        // Check for darkness patterns
        let light_level: f64 = input.iter().sum::<f64>() / input.len() as f64;
        
        if light_level < 0.3 {
            self.darkness_fear * (0.3 - light_level) * 3.33 // Scales to 1.0 at light_level=0
        } else {
            0.0
        }
    }
    
    fn integrate_bmd_outputs(&self, outputs: &[Vec<f64>]) -> AutobahnResult<Vec<f64>> {
        if outputs.is_empty() {
            return Ok(vec![0.0; 10]); // Default output size
        }
        
        let output_size = outputs[0].len();
        let mut integrated = vec![0.0; output_size];
        
        for output in outputs {
            for (i, &value) in output.iter().enumerate() {
                if i < integrated.len() {
                    integrated[i] += value / outputs.len() as f64; // Average integration
                }
            }
        }
        
        Ok(integrated)
    }
    
    fn apply_consciousness_modulation(&self, response: &[f64]) -> Vec<f64> {
        response.iter()
            .map(|&x| x * self.consciousness_level)
            .collect()
    }
    
    /// Test underwater fireplace paradox
    pub async fn test_underwater_fireplace_paradox(&mut self) -> AutobahnResult<UnderwaterFireplaceTest> {
        // Create impossible underwater fire scenario
        let impossible_fire_env = FireEnvironment::impossible_underwater();
        let old_env = std::mem::replace(&mut self.fire_environment, impossible_fire_env);
        
        // Fire pattern in impossible context
        let underwater_fire_input = [0.8, 0.6, 0.4, 0.9, 0.7]; // Strong fire signature
        
        let response = self.process_input(&underwater_fire_input).await?;
        
        // Restore original environment
        self.fire_environment = old_env;
        
        Ok(UnderwaterFireplaceTest {
            paradox_demonstrated: response.fire_recognition.logical_override && response.fire_recognition.impossible_context,
            recognition_strength: response.fire_recognition.recognition_strength,
            logical_override: response.fire_recognition.logical_override,
            impossible_context: response.fire_recognition.impossible_context,
            human_attribution: response.fire_recognition.human_attribution,
            consciousness_level: response.consciousness_level,
        })
    }
}

/// Individual ion channel with quantum properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IonChannel {
    pub ion_type: IonType,
    pub conductance_siemens: f64,
    pub voltage_threshold_mv: f64,
    pub phase_offset: f64,
    pub fire_adaptation_factor: f64, // 0-1, evolutionary adaptation to fire
    pub spatial_location: (f64, f64, f64), // 3D coordinates in micrometers
}

impl IonChannel {
    /// Calculate quantum field contribution from this channel
    pub fn quantum_field_contribution(&self, fire_intensity: f64, fire_enhancement: f64) -> AutobahnResult<f64> {
        let base_contribution = self.conductance_siemens * self.fire_adaptation_factor;
        let fire_boost = 1.0 + (fire_intensity * fire_enhancement * 0.3);
        let tunneling_prob = self.ion_type.tunneling_probability(0.1, 1.0); // Typical membrane values
        
        Ok(base_contribution * fire_boost * tunneling_prob)
    }
}

// ============================================================================
// BIOLOGICAL MAXWELL'S DEMONS (BMD) - INFORMATION CATALYSTS
// ============================================================================

/// Biological Maxwell's Demon implementing Mizraji's information catalyst theory
#[derive(Debug, Clone)]
pub struct BiologicalMaxwellDemon {
    /// Unique identifier
    pub id: String,
    /// Specialization type
    pub specialization: BMDSpecialization,
    /// Input pattern filter
    pub input_filter: InformationFilter,
    /// Output response filter  
    pub output_filter: InformationFilter,
    /// Catalytic amplification efficiency
    pub catalytic_efficiency: f64,
    /// Associative memory patterns
    pub memory_patterns: Vec<(Vec<f64>, Vec<f64>)>,
    /// Fire-specific processing enhancement
    pub fire_specialization_strength: f64,
    /// Current activation level
    pub activation_level: f64,
    /// Learning rate for pattern adaptation
    pub learning_rate: f64,
}

/// BMD specializations evolved for different cognitive functions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BMDSpecialization {
    /// Hardwired fire detection (Underwater Fireplace Paradox)
    FireRecognition,
    /// Individual agency recognition in fire circles
    AgencyDetection,
    /// Spatial navigation and memory
    SpatialMemory,
    /// Temporal planning and future modeling
    TemporalPlanning,
    /// Social coordination and communication
    SocialCoordination,
    /// Threat assessment and response
    ThreatAssessment,
}

impl BiologicalMaxwellDemon {
    /// Create new BMD with specified specialization
    pub fn new(specialization: BMDSpecialization) -> Self {
        let (input_filter, output_filter, fire_strength) = match specialization {
            BMDSpecialization::FireRecognition => {
                (InformationFilter::fire_recognition(), InformationFilter::threat_response(), 0.95)
            },
            BMDSpecialization::AgencyDetection => {
                (InformationFilter::agency_patterns(), InformationFilter::social_response(), 0.8)
            },
            BMDSpecialization::SpatialMemory => {
                (InformationFilter::spatial_patterns(), InformationFilter::navigation_response(), 0.6)
            },
            BMDSpecialization::TemporalPlanning => {
                (InformationFilter::temporal_patterns(), InformationFilter::planning_response(), 0.7)
            },
            BMDSpecialization::SocialCoordination => {
                (InformationFilter::social_patterns(), InformationFilter::social_response(), 0.75)
            },
            BMDSpecialization::ThreatAssessment => {
                (InformationFilter::threat_patterns(), InformationFilter::threat_response(), 0.85)
            },
        };
        
        Self {
            id: format!("{:?}_{}", specialization, chrono::Utc::now().timestamp()),
            specialization,
            input_filter,
            output_filter,
            catalytic_efficiency: 2.5, // Default amplification
            memory_patterns: Vec::new(),
            fire_specialization_strength: fire_strength,
            activation_level: 0.0,
            learning_rate: 0.01,
        }
    }
    
    /// Process information through BMD with quantum enhancement
    pub fn process_information(
        &mut self, 
        input: &[f64], 
        quantum_field: &QuantumCoherenceField,
        fire_environment: &FireEnvironment
    ) -> AutobahnResult<Vec<f64>> {
        // Step 1: Input filtering
        let filtered_input = self.input_filter.apply(input)?;
        
        // Step 2: Quantum enhancement
        let quantum_boost = quantum_field.quantum_enhancement_factor();
        
        // Step 3: Fire recognition enhancement
        let fire_boost = if self.is_fire_related_input(input, fire_environment)? {
            1.0 + self.fire_specialization_strength
        } else {
            1.0
        };
        
        // Step 4: Associative memory processing
        let memory_output = self.associative_memory_lookup(&filtered_input)?;
        
        // Step 5: Output filtering and amplification
        let mut output = self.output_filter.apply(&memory_output)?;
        
        // Step 6: Apply all enhancements
        let total_enhancement = self.catalytic_efficiency * quantum_boost * fire_boost;
        for value in &mut output {
            *value *= total_enhancement;
        }
        
        // Step 7: Update activation level
        self.activation_level = (self.activation_level * 0.9 + total_enhancement * 0.1).min(2.0);
        
        Ok(output)
    }
    
    /// Associative memory lookup with pattern matching
    fn associative_memory_lookup(&self, input: &[f64]) -> AutobahnResult<Vec<f64>> {
        if self.memory_patterns.is_empty() {
            return Ok(input.to_vec());
        }
        
        // Find best matching pattern
        let mut best_match_idx = 0;
        let mut best_similarity = -1.0;
        
        for (i, (pattern, _)) in self.memory_patterns.iter().enumerate() {
            let similarity = cosine_similarity(input, pattern)?;
            if similarity > best_similarity {
                best_similarity = similarity;
                best_match_idx = i;
            }
        }
        
        // Return associated output pattern
        Ok(self.memory_patterns[best_match_idx].1.clone())
    }
    
    /// Check if input contains fire-related patterns
    fn is_fire_related_input(&self, input: &[f64], fire_env: &FireEnvironment) -> AutobahnResult<bool> {
        // Fire signature based on wavelength, intensity, and thermal patterns
        let fire_signature = fire_env.get_fire_signature();
        let similarity = cosine_similarity(input, &fire_signature)?;
        
        // Underwater Fireplace Paradox: Fire recognition overrides logic
        let recognition_threshold = match self.specialization {
            BMDSpecialization::FireRecognition => 0.4, // Lower threshold for fire BMDs
            _ => 0.6,
        };
        
        Ok(similarity > recognition_threshold)
    }
    
    /// Learn new pattern association
    pub fn learn_pattern(&mut self, input: Vec<f64>, output: Vec<f64>) -> AutobahnResult<()> {
        self.memory_patterns.push((input, output));
        
        // Limit memory to prevent unbounded growth
        if self.memory_patterns.len() > 1000 {
            self.memory_patterns.remove(0);
        }
        
        Ok(())
    }
}

/// Information filter for BMD input/output processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFilter {
    /// Filter weights for different input dimensions
    pub weights: Vec<f64>,
    /// Activation threshold
    pub threshold: f64,
    /// Filter type identifier
    pub filter_type: String,
    /// Adaptation rate for learning
    pub adaptation_rate: f64,
}

impl InformationFilter {
    /// Create fire recognition filter (hardwired from evolution)
    pub fn fire_recognition() -> Self {
        Self {
            weights: vec![0.95, 0.9, 0.85, 0.98, 0.92], // High sensitivity to fire patterns
            threshold: 0.4, // Low threshold for fire detection
            filter_type: "fire_recognition".to_string(),
            adaptation_rate: 0.001, // Minimal adaptation - hardwired
        }
    }
    
    /// Create agency detection filter
    pub fn agency_patterns() -> Self {
        Self {
            weights: vec![0.8, 0.85, 0.7, 0.9, 0.8],
            threshold: 0.5,
            filter_type: "agency_detection".to_string(),
            adaptation_rate: 0.01,
        }
    }
    
    /// Create spatial pattern filter
    pub fn spatial_patterns() -> Self {
        Self {
            weights: vec![0.7, 0.8, 0.9, 0.7, 0.75],
            threshold: 0.4,
            filter_type: "spatial_memory".to_string(),
            adaptation_rate: 0.02,
        }
    }
    
    /// Create temporal pattern filter
    pub fn temporal_patterns() -> Self {
        Self {
            weights: vec![0.85, 0.7, 0.75, 0.9, 0.95],
            threshold: 0.5,
            filter_type: "temporal_planning".to_string(),
            adaptation_rate: 0.015,
        }
    }
    
    /// Create social pattern filter
    pub fn social_patterns() -> Self {
        Self {
            weights: vec![0.75, 0.8, 0.85, 0.8, 0.9],
            threshold: 0.55,
            filter_type: "social_coordination".to_string(),
            adaptation_rate: 0.02,
        }
    }
    
    /// Create threat pattern filter
    pub fn threat_patterns() -> Self {
        Self {
            weights: vec![0.9, 0.95, 0.85, 0.92, 0.88],
            threshold: 0.6,
            filter_type: "threat_assessment".to_string(),
            adaptation_rate: 0.005, // Slow adaptation for safety
        }
    }
    
    /// Create threat response filter
    pub fn threat_response() -> Self {
        Self {
            weights: vec![0.98, 0.95, 0.9, 0.96, 0.92],
            threshold: 0.7,
            filter_type: "threat_response".to_string(),
            adaptation_rate: 0.001,
        }
    }
    
    /// Create social response filter
    pub fn social_response() -> Self {
        Self {
            weights: vec![0.8, 0.85, 0.8, 0.9, 0.88],
            threshold: 0.6,
            filter_type: "social_response".to_string(),
            adaptation_rate: 0.02,
        }
    }
    
    /// Create navigation response filter
    pub fn navigation_response() -> Self {
        Self {
            weights: vec![0.7, 0.8, 0.95, 0.7, 0.75],
            threshold: 0.4,
            filter_type: "navigation_response".to_string(),
            adaptation_rate: 0.025,
        }
    }
    
    /// Create planning response filter
    pub fn planning_response() -> Self {
        Self {
            weights: vec![0.85, 0.8, 0.7, 0.95, 0.98],
            threshold: 0.6,
            filter_type: "planning_response".to_string(),
            adaptation_rate: 0.01,
        }
    }
    
    /// Apply filter to input data
    pub fn apply(&self, input: &[f64]) -> AutobahnResult<Vec<f64>> {
        let mut output = Vec::new();
        
        for (i, &value) in input.iter().enumerate() {
            let weight = self.weights.get(i % self.weights.len()).unwrap_or(&1.0);
            let filtered_value = value * weight;
            
            // Apply threshold with sigmoid activation
            let activated_value = if filtered_value > self.threshold {
                filtered_value
            } else {
                filtered_value * 0.1 // Reduced but not zero
            };
            
            output.push(activated_value);
        }
        
        Ok(output)
    }
}

// ============================================================================
// FIRE ENVIRONMENT AND EVOLUTIONARY CONTEXT
// ============================================================================

/// Fire environment that catalyzed consciousness evolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireEnvironment {
    /// Fire intensity (0.0 to 1.0)
    pub intensity: f64,
    /// Dominant wavelength in nanometers
    pub dominant_wavelength_nm: f64,
    /// Full wavelength spectrum
    pub wavelength_spectrum: Vec<f64>,
    /// Temperature increase from fire (Celsius)
    pub temperature_increase_c: f64,
    /// Duration of fire exposure (hours)
    pub exposure_duration_hours: f64,
    /// Number of individuals in fire circle
    pub group_size: usize,
    /// C4 grass coverage factor (0-1)
    pub c4_coverage: f64,
    /// Environmental context
    pub context: EnvironmentalContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnvironmentalContext {
    /// Normal fire environment
    Normal,
    /// Impossible fire context (underwater, vacuum, etc.)
    Impossible { context_type: String },
    /// Enhanced fire environment (optimal conditions)
    Enhanced,
}

impl FireEnvironment {
    /// Create Olduvai ecosystem fire environment (8-3 MYA)
    pub fn olduvai_ecosystem() -> Self {
        Self {
            intensity: 0.8,
            dominant_wavelength_nm: 650.0, // Optimal for consciousness
            wavelength_spectrum: vec![600.0, 620.0, 650.0, 680.0, 700.0],
            temperature_increase_c: 8.0, // Optimal for ion mobility
            exposure_duration_hours: 5.5, // Extended evening social time
            group_size: 12, // Typical early hominid group
            c4_coverage: 0.85, // High C4 grass coverage
            context: EnvironmentalContext::Normal,
        }
    }
    
    /// Create impossible fire environment for testing Underwater Fireplace Paradox
    pub fn impossible_underwater() -> Self {
        let mut env = Self::olduvai_ecosystem();
        env.context = EnvironmentalContext::Impossible { 
            context_type: "underwater".to_string() 
        };
        env
    }
    
    /// Calculate fire exposure probability using mathematical model from Chapter 7
    pub fn fire_exposure_probability(&self, area_km2: f64, dry_season_days: f64) -> AutobahnResult<f64> {
        const LIGHTNING_FREQUENCY: f64 = 0.035; // strikes/km²/day
        const CLIMATE_ARIDITY: f64 = 1.2; // aridity factor
        
        let exponent = -LIGHTNING_FREQUENCY * area_km2 * dry_season_days * self.c4_coverage * CLIMATE_ARIDITY;
        Ok(1.0 - exponent.exp()) // P(F) = 1 - exp(-λ * A * T * φ(C4) * ψ(climate))
    }
    
    /// Calculate consciousness enhancement factor from fire environment
    pub fn consciousness_enhancement_factor(&self) -> AutobahnResult<f64> {
        let wavelength_factor = self.optimal_wavelength_factor();
        let thermal_factor = self.optimal_thermal_factor();
        let social_factor = self.social_witness_factor();
        let duration_factor = (self.exposure_duration_hours / 8.0).min(1.0);
        
        let base_enhancement = wavelength_factor * thermal_factor * social_factor * duration_factor;
        
        // Context modifier
        let context_modifier = match &self.context {
            EnvironmentalContext::Normal => 1.0,
            EnvironmentalContext::Enhanced => 1.2,
            EnvironmentalContext::Impossible { .. } => 0.8, // Reduced but still present (paradox)
        };
        
        Ok(base_enhancement * context_modifier)
    }
    
    /// Calculate optimal wavelength factor for consciousness
    fn optimal_wavelength_factor(&self) -> f64 {
        const OPTIMAL_WAVELENGTH: f64 = 650.0; // nm
        const WAVELENGTH_BANDWIDTH: f64 = 50.0; // nm
        
        let deviation = (self.dominant_wavelength_nm - OPTIMAL_WAVELENGTH).abs();
        (-(deviation / WAVELENGTH_BANDWIDTH).powi(2)).exp()
    }
    
    /// Calculate optimal thermal factor
    fn optimal_thermal_factor(&self) -> f64 {
        const OPTIMAL_TEMP_INCREASE: f64 = 8.0; // Celsius
        const TEMP_BANDWIDTH: f64 = 3.0; // Celsius
        
        let deviation = (self.temperature_increase_c - OPTIMAL_TEMP_INCREASE).abs();
        (-(deviation / TEMP_BANDWIDTH).powi(2)).exp()
    }
    
    /// Calculate social witness factor (agency emergence in groups)
    fn social_witness_factor(&self) -> f64 {
        const OPTIMAL_GROUP_SIZE: f64 = 12.0;
        const GROUP_SIZE_BANDWIDTH: f64 = 5.0;
        
        let deviation = (self.group_size as f64 - OPTIMAL_GROUP_SIZE).abs();
        (-(deviation / GROUP_SIZE_BANDWIDTH).powi(2)).exp()
    }
    
    /// Get fire signature pattern for recognition
    pub fn get_fire_signature(&self) -> Vec<f64> {
        vec![
            self.intensity,
            self.dominant_wavelength_nm / 1000.0, // Normalize to 0-1 range
            self.temperature_increase_c / 20.0,   // Normalize
            self.exposure_duration_hours / 10.0,  // Normalize
            self.group_size as f64 / 20.0,        // Normalize
        ]
    }
}

// ============================================================================
// COMPLETE FIRE-CONSCIOUSNESS SYSTEM
// ============================================================================

/// Complete fire-evolved consciousness system
#[derive(Debug)]
pub struct FireConsciousnessEngine {
    /// Quantum coherence field from ion channels
    pub quantum_field: QuantumCoherenceField,
    /// Array of specialized BMDs
    pub bmds: Vec<BiologicalMaxwellDemon>,
    /// Fire environment context
    pub fire_environment: FireEnvironment,
    /// Ion channels in neural network
    pub ion_channels: Vec<IonChannel>,
    /// Current consciousness level
    pub consciousness_level: f64,
    /// Agency recognition strength
    pub agency_recognition: f64,
    /// Darkness fear response (consciousness malfunction)
    pub darkness_fear: f64,
    /// Integration with base consciousness system
    pub base_consciousness: Option<ConsciousnessEmergenceEngine>,
}

impl FireConsciousnessEngine {
    /// Create new fire-consciousness engine
    pub fn new(evolutionary_time_mya: f64) -> AutobahnResult<Self> {
        // Create ion channels based on evolutionary time
        let ion_channels = Self::create_evolutionary_ion_channels(evolutionary_time_mya)?;
        
        // Create fire environment (Olduvai ecosystem)
        let fire_environment = FireEnvironment::olduvai_ecosystem();
        
        // Create quantum coherence field
        let quantum_field = QuantumCoherenceField::new(&ion_channels, &fire_environment)?;
        
        // Create specialized BMDs
        let bmds = vec![
            BiologicalMaxwellDemon::new(BMDSpecialization::FireRecognition),
            BiologicalMaxwellDemon::new(BMDSpecialization::AgencyDetection),
            BiologicalMaxwellDemon::new(BMDSpecialization::SpatialMemory),
            BiologicalMaxwellDemon::new(BMDSpecialization::TemporalPlanning),
            BiologicalMaxwellDemon::new(BMDSpecialization::SocialCoordination),
            BiologicalMaxwellDemon::new(BMDSpecialization::ThreatAssessment),
        ];
        
        // Calculate initial consciousness level
        let consciousness_level = if quantum_field.meets_consciousness_threshold { 0.7 } else { 0.3 };
        
        Ok(Self {
            quantum_field,
            bmds,
            fire_environment,
            ion_channels,
            consciousness_level,
            agency_recognition: 0.5,
            darkness_fear: 0.2,
            base_consciousness: None,
        })
    }
    
    /// Create ion channels based on evolutionary timeline
    fn create_evolutionary_ion_channels(evolutionary_time_mya: f64) -> AutobahnResult<Vec<IonChannel>> {
        let mut channels = Vec::new();
        
        // Fire adaptation increases with evolutionary time (8-3 MYA)
        let fire_adaptation = if evolutionary_time_mya >= 3.0 && evolutionary_time_mya <= 8.0 {
            1.0 - (evolutionary_time_mya - 3.0) / 5.0 // Linear increase
        } else {
            0.5 // Baseline adaptation
        };
        
        // Create H+ channels (most important for consciousness)
        for i in 0..50 {
            channels.push(IonChannel {
                ion_type: IonType::Hydrogen,
                conductance_siemens: 1e-9 + (i as f64 * 1e-11),
                voltage_threshold_mv: -60.0 + (i as f64 * 0.5),
                phase_offset: (i as f64 * PI / 25.0) % (2.0 * PI),
                fire_adaptation_factor: fire_adaptation,
                spatial_location: (
                    (i as f64 % 10.0) * 10.0,
                    ((i / 10) as f64) * 10.0,
                    0.0
                ),
            });
        }
        
        // Create other ion channels
        for ion_type in [IonType::Sodium, IonType::Potassium, IonType::Calcium, IonType::Magnesium] {
            for i in 0..20 {
                channels.push(IonChannel {
                    ion_type,
                    conductance_siemens: 5e-10 + (i as f64 * 2e-11),
                    voltage_threshold_mv: -50.0 + (i as f64 * 1.0),
                    phase_offset: (i as f64 * PI / 10.0) % (2.0 * PI),
                    fire_adaptation_factor: fire_adaptation * 0.8, // Slightly less adapted
                    spatial_location: (
                        (i as f64 % 5.0) * 20.0,
                        ((i / 5) as f64) * 20.0,
                        10.0
                    ),
                });
            }
        }
        
        Ok(channels)
    }
    
    /// Process input through complete fire-consciousness system
    pub async fn process_conscious_input(&mut self, input: &[f64]) -> AutobahnResult<FireConsciousnessResponse> {
        // Step 1: Fire recognition (Underwater Fireplace Paradox)
        let fire_recognition = self.recognize_fire_patterns(input).await?;
        
        // Step 2: Process through BMDs with quantum enhancement
        let mut bmd_outputs = Vec::new();
        for bmd in &mut self.bmds {
            let output = bmd.process_information(input, &self.quantum_field, &self.fire_environment)?;
            bmd_outputs.push(output);
        }
        
        // Step 3: Integrate BMD outputs
        let integrated_response = self.integrate_bmd_outputs(&bmd_outputs)?;
        
        // Step 4: Apply consciousness-level modulation
        let conscious_response = self.apply_consciousness_modulation(&integrated_response);
        
        // Step 5: Agency detection
        let agency_detection = self.detect_individual_agency(input).await?;
        
        // Step 6: Darkness response (if applicable)
        let darkness_response = self.calculate_darkness_response(input);
        
        // Step 7: Update consciousness state
        self.update_consciousness_state(&fire_recognition, &agency_detection);
        
        Ok(FireConsciousnessResponse {
            fire_recognition,
            conscious_processing: conscious_response,
            agency_detection,
            darkness_fear_activation: darkness_response,
            quantum_coherence_active: self.quantum_field.meets_consciousness_threshold,
            consciousness_level: self.consciousness_level,
            bmd_activations: self.bmds.iter().map(|bmd| bmd.activation_level).collect(),
        })
    }
    
    /// Recognize fire patterns (hardwired recognition overrides logic)
    async fn recognize_fire_patterns(&self, input: &[f64]) -> AutobahnResult<FireRecognitionResponse> {
        let fire_signature = self.fire_environment.get_fire_signature();
        let similarity = cosine_similarity(input, &fire_signature)?;
        
        // Enhanced recognition strength
        let recognition_strength = similarity * 1.3;
        
        // Logical override threshold (Underwater Fireplace Paradox)
        let logical_override = recognition_strength > 0.4;
        
        // Check for impossible contexts
        let impossible_context = matches!(self.fire_environment.context, 
            EnvironmentalContext::Impossible { .. });
        
        Ok(FireRecognitionResponse {
            recognition_strength,
            logical_override,
            impossible_context,
            human_attribution: logical_override, // Fire = Human presence
            fire_signature_match: similarity,
        })
    }
    
    /// Detect individual agency in fire circle contexts
    async fn detect_individual_agency(&self, input: &[f64]) -> AutobahnResult<AgencyDetection> {
        if self.agency_recognition < 0.3 {
            return Ok(AgencyDetection {
                agency_detected: false,
                agency_strength: 0.0,
                individual_signatures: Vec::new(),
                witness_context_active: false,
            });
        }
        
        // Simplified agency detection (would be much more sophisticated)
        let agency_patterns = vec![0.7, 0.8, 0.6, 0.9, 0.75]; // Example agency signature
        let similarity = cosine_similarity(input, &agency_patterns)?;
        
        let agency_detected = similarity > 0.6 && self.fire_environment.group_size > 1;
        let witness_context = self.fire_environment.group_size >= 3; // Need witnesses
        
        Ok(AgencyDetection {
            agency_detected,
            agency_strength: similarity,
            individual_signatures: if agency_detected { 
                vec![format!("individual_{}", chrono::Utc::now().timestamp())] 
            } else { 
                Vec::new() 
            },
            witness_context_active: witness_context,
        })
    }
    
    /// Calculate darkness response (consciousness malfunction without light)
    fn calculate_darkness_response(&self, input: &[f64]) -> f64 {
        // Check for low light conditions
        let light_level = input.get(1).unwrap_or(&0.5); // Assume index 1 is light
        
        if *light_level < 0.2 {
            // Darkness triggers fear response (consciousness malfunction)
            let darkness_intensity = (0.2 - light_level) / 0.2;
            self.darkness_fear + (darkness_intensity * 0.5)
        } else {
            self.darkness_fear * 0.9 // Gradual reduction in light
        }
    }
    
    /// Integrate outputs from multiple BMDs
    fn integrate_bmd_outputs(&self, outputs: &[Vec<f64>]) -> AutobahnResult<Vec<f64>> {
        if outputs.is_empty() {
            return Ok(Vec::new());
        }
        
        let output_length = outputs[0].len();
        let mut integrated = vec![0.0; output_length];
        
        // Weighted integration - fire and agency detection prioritized
        let weights = [0.3, 0.25, 0.15, 0.15, 0.1, 0.05];
        
        for (i, output) in outputs.iter().enumerate() {
            let weight = weights.get(i).unwrap_or(&0.1);
            for (j, &value) in output.iter().enumerate() {
                if j < integrated.len() {
                    integrated[j] += value * weight;
                }
            }
        }
        
        Ok(integrated)
    }
    
    /// Apply consciousness-level modulation
    fn apply_consciousness_modulation(&self, response: &[f64]) -> Vec<f64> {
        response.iter()
            .map(|&value| value * self.consciousness_level)
            .collect()
    }
    
    /// Update consciousness state based on processing
    fn update_consciousness_state(&mut self, fire_recognition: &FireRecognitionResponse, agency: &AgencyDetection) {
        // Fire recognition enhances consciousness
        let fire_enhancement = fire_recognition.recognition_strength * 0.1;
        
        // Agency detection enhances consciousness
        let agency_enhancement = if agency.agency_detected { 0.05 } else { 0.0 };
        
        // Quantum field contribution
        let quantum_enhancement = if self.quantum_field.meets_consciousness_threshold { 0.05 } else { -0.02 };
        
        // Update consciousness level
        self.consciousness_level = (self.consciousness_level + fire_enhancement + agency_enhancement + quantum_enhancement)
            .max(0.0)
            .min(1.0);
        
        // Update agency recognition
        self.agency_recognition = (self.agency_recognition + agency_enhancement * 2.0)
            .max(0.0)
            .min(1.0);
    }
    
    /// Test Underwater Fireplace Paradox
    pub async fn test_underwater_fireplace_paradox(&mut self) -> AutobahnResult<UnderwaterFireplaceTest> {
        // Set impossible fire environment
        self.fire_environment = FireEnvironment::impossible_underwater();
        
        // Create fire-like input
        let fire_input = vec![0.8, 0.6, 0.4, 0.9, 0.7];
        
        // Process through consciousness system
        let response = self.process_conscious_input(&fire_input).await?;
        
        // Paradox test: Fire recognition should override logical impossibility
        let paradox_demonstrated = response.fire_recognition.logical_override 
            && response.fire_recognition.impossible_context
            && response.fire_recognition.human_attribution;
        
        Ok(UnderwaterFireplaceTest {
            paradox_demonstrated,
            recognition_strength: response.fire_recognition.recognition_strength,
            logical_override: response.fire_recognition.logical_override,
            impossible_context: response.fire_recognition.impossible_context,
            human_attribution: response.fire_recognition.human_attribution,
            consciousness_level: response.consciousness_level,
        })
    }
}

// ============================================================================
// RESPONSE STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireConsciousnessResponse {
    pub fire_recognition: FireRecognitionResponse,
    pub conscious_processing: Vec<f64>,
    pub agency_detection: AgencyDetection,
    pub darkness_fear_activation: f64,
    pub quantum_coherence_active: bool,
    pub consciousness_level: f64,
    pub bmd_activations: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireRecognitionResponse {
    pub recognition_strength: f64,
    pub logical_override: bool,
    pub impossible_context: bool,
    pub human_attribution: bool,
    pub fire_signature_match: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgencyDetection {
    pub agency_detected: bool,
    pub agency_strength: f64,
    pub individual_signatures: Vec<String>,
    pub witness_context_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnderwaterFireplaceTest {
    pub paradox_demonstrated: bool,
    pub recognition_strength: f64,
    pub logical_override: bool,
    pub impossible_context: bool,
    pub human_attribution: bool,
    pub consciousness_level: f64,
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> AutobahnResult<f64> {
    if a.is_empty() || b.is_empty() {
        return Ok(0.0);
    }
    
    let min_len = a.len().min(b.len());
    let a_slice = &a[..min_len];
    let b_slice = &b[..min_len];
    
    let dot_product: f64 = a_slice.iter().zip(b_slice.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a_slice.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b_slice.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot_product / (norm_a * norm_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_fire_consciousness_engine() -> AutobahnResult<()> {
        let mut engine = FireConsciousnessEngine::new(5.0)?;
        
        // Test fire recognition
        let fire_input = vec![0.8, 0.6, 0.4, 0.9, 0.7];
        let response = engine.process_conscious_input(&fire_input).await?;
        
        assert!(response.fire_recognition.recognition_strength > 0.5);
        assert!(response.consciousness_level > 0.3);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_underwater_fireplace_paradox() -> AutobahnResult<()> {
        let mut engine = FireConsciousnessEngine::new(5.0)?;
        let test_result = engine.test_underwater_fireplace_paradox().await?;
        
        // Should demonstrate paradox: fire recognition despite impossibility
        assert!(test_result.impossible_context);
        assert!(test_result.logical_override || test_result.recognition_strength > 0.3);
        
        Ok(())
    }
    
    #[test]
    fn test_ion_tunneling_probability() {
        let hydrogen = IonType::Hydrogen;
        let sodium = IonType::Sodium;
        
        // H+ should have higher tunneling probability due to lower mass
        let h_prob = hydrogen.tunneling_probability(0.1, 1.0);
        let na_prob = sodium.tunneling_probability(0.1, 1.0);
        
        assert!(h_prob > na_prob);
    }
    
    #[test]
    fn test_fire_light_enhancement() {
        let hydrogen = IonType::Hydrogen;
        
        // Optimal fire wavelength should give maximum enhancement
        let optimal_enhancement = hydrogen.fire_light_enhancement(650.0);
        let suboptimal_enhancement = hydrogen.fire_light_enhancement(400.0);
        
        assert!(optimal_enhancement > suboptimal_enhancement);
    }
} 