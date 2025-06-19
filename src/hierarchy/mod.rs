//! Multi-scale hierarchy processing system implementing the 10-level biological hierarchy
//! from quantum oscillations (10⁻⁴⁴ s) to cosmic patterns (10¹³ s).
//!
//! This module demonstrates how consciousness emerges from nested oscillatory dynamics
//! across multiple time and length scales.

use crate::error::{AutobahnError, AutobahnResult};
use crate::oscillatory::{OscillationProfile, OscillationPhase, UniversalOscillator};
use crate::quantum::{QuantumOscillatoryProfile, QuantumMembraneState};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// The 10 hierarchy levels of biological organization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HierarchyLevel {
    /// Level 1: Quantum oscillations (10⁻⁴⁴ s) - Planck scale quantum fluctuations
    QuantumOscillations = 1,
    /// Level 2: Atomic oscillations (10⁻¹⁵ s) - Electronic transitions, nuclear vibrations
    AtomicOscillations = 2,
    /// Level 3: Molecular oscillations (10⁻¹² s) - Molecular vibrations, rotations
    MolecularOscillations = 3,
    /// Level 4: Cellular oscillations (10⁻³ s) - Metabolic cycles, membrane dynamics
    CellularOscillations = 4,
    /// Level 5: Organismal oscillations (10⁰ s) - Heartbeat, breathing, neural firing
    OrganismalOscillations = 5,
    /// Level 6: Cognitive oscillations (10³ s) - Thought processes, decision making
    CognitiveOscillations = 6,
    /// Level 7: Social oscillations (10⁶ s) - Social interactions, group dynamics
    SocialOscillations = 7,
    /// Level 8: Technological oscillations (10⁹ s) - Innovation cycles, technological adoption
    TechnologicalOscillations = 8,
    /// Level 9: Civilizational oscillations (10¹² s) - Rise and fall of civilizations
    CivilizationalOscillations = 9,
    /// Level 10: Cosmic oscillations (10¹³ s) - Stellar evolution, galactic dynamics
    CosmicOscillations = 10,
}

impl HierarchyLevel {
    /// Get the characteristic time scale for this hierarchy level in seconds
    pub fn time_scale_seconds(&self) -> f64 {
        match self {
            HierarchyLevel::QuantumOscillations => 1e-44,      // Planck time
            HierarchyLevel::AtomicOscillations => 1e-15,       // Femtoseconds
            HierarchyLevel::MolecularOscillations => 1e-12,    // Picoseconds
            HierarchyLevel::CellularOscillations => 1e-3,      // Milliseconds
            HierarchyLevel::OrganismalOscillations => 1e0,     // Seconds
            HierarchyLevel::CognitiveOscillations => 1e3,      // ~15 minutes
            HierarchyLevel::SocialOscillations => 1e6,         // ~11 days
            HierarchyLevel::TechnologicalOscillations => 1e9,  // ~30 years
            HierarchyLevel::CivilizationalOscillations => 1e12, // ~30,000 years
            HierarchyLevel::CosmicOscillations => 1e13,        // ~300,000 years
        }
    }
    
    /// Get the characteristic frequency for this hierarchy level in Hz
    pub fn characteristic_frequency(&self) -> f64 {
        1.0 / self.time_scale_seconds()
    }
    
    /// Get typical energy scale for this hierarchy level in Joules
    pub fn energy_scale(&self) -> f64 {
        // E = ℏω = h/2π * 1/T = h/(2πT)
        const PLANCK_CONSTANT: f64 = 6.62607015e-34; // J⋅s
        const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
        
        PLANCK_CONSTANT / (TWO_PI * self.time_scale_seconds())
    }
    
    /// Get typical length scale for this hierarchy level in meters
    pub fn length_scale(&self) -> f64 {
        match self {
            HierarchyLevel::QuantumOscillations => 1.616e-35,      // Planck length
            HierarchyLevel::AtomicOscillations => 1e-10,           // Atomic scale
            HierarchyLevel::MolecularOscillations => 1e-9,         // Molecular scale
            HierarchyLevel::CellularOscillations => 1e-6,          // Cellular scale
            HierarchyLevel::OrganismalOscillations => 1e-1,        // Organ scale
            HierarchyLevel::CognitiveOscillations => 1e0,          // Organism scale
            HierarchyLevel::SocialOscillations => 1e3,             // Social group scale
            HierarchyLevel::TechnologicalOscillations => 1e6,      // City scale
            HierarchyLevel::CivilizationalOscillations => 1e7,     // Continental scale
            HierarchyLevel::CosmicOscillations => 1e21,            // Galactic scale
        }
    }
    
    /// Get information processing capacity at this level (bits/second)
    pub fn information_capacity(&self) -> f64 {
        // Landauer's principle: information capacity scales with available energy
        const BOLTZMANN_CONSTANT: f64 = 1.380649e-23; // J/K
        const ROOM_TEMPERATURE: f64 = 300.0; // K
        
        let thermal_energy = BOLTZMANN_CONSTANT * ROOM_TEMPERATURE;
        let level_energy = self.energy_scale();
        
        // Information capacity proportional to energy ratio
        let capacity_factor = (level_energy / thermal_energy).ln().max(1.0);
        capacity_factor * self.characteristic_frequency()
    }
    
    /// Get all hierarchy levels
    pub fn all_levels() -> Vec<HierarchyLevel> {
        vec![
            HierarchyLevel::QuantumOscillations,
            HierarchyLevel::AtomicOscillations,
            HierarchyLevel::MolecularOscillations,
            HierarchyLevel::CellularOscillations,
            HierarchyLevel::OrganismalOscillations,
            HierarchyLevel::CognitiveOscillations,
            HierarchyLevel::SocialOscillations,
            HierarchyLevel::TechnologicalOscillations,
            HierarchyLevel::CivilizationalOscillations,
            HierarchyLevel::CosmicOscillations,
        ]
    }
    
    /// Get adjacent hierarchy levels for coupling calculations
    pub fn adjacent_levels(&self) -> Vec<HierarchyLevel> {
        let all_levels = Self::all_levels();
        let current_index = *self as usize - 1;
        let mut adjacent = Vec::new();
        
        if current_index > 0 {
            adjacent.push(all_levels[current_index - 1]);
        }
        if current_index < all_levels.len() - 1 {
            adjacent.push(all_levels[current_index + 1]);
        }
        
        adjacent
    }
    
    /// Check if this level can couple with another level
    pub fn can_couple_with(&self, other: HierarchyLevel) -> bool {
        let level_diff = (*self as i32 - other as i32).abs();
        level_diff <= 3 // Allow coupling within 3 levels
    }
    
    /// Calculate coupling strength with another level
    pub fn coupling_strength(&self, other: HierarchyLevel) -> f64 {
        if !self.can_couple_with(other) {
            return 0.0;
        }
        
        let level_diff = (*self as i32 - other as i32).abs() as f64;
        let freq_ratio = (self.characteristic_frequency() / other.characteristic_frequency()).ln().abs();
        
        // Coupling strength decreases exponentially with frequency separation
        0.8 * (-freq_ratio / 5.0).exp() * (1.0 / (1.0 + level_diff))
    }
}

impl fmt::Display for HierarchyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            HierarchyLevel::QuantumOscillations => "Quantum (10⁻⁴⁴s)",
            HierarchyLevel::AtomicOscillations => "Atomic (10⁻¹⁵s)",
            HierarchyLevel::MolecularOscillations => "Molecular (10⁻¹²s)",
            HierarchyLevel::CellularOscillations => "Cellular (10⁻³s)",
            HierarchyLevel::OrganismalOscillations => "Organismal (10⁰s)",
            HierarchyLevel::CognitiveOscillations => "Cognitive (10³s)",
            HierarchyLevel::SocialOscillations => "Social (10⁶s)",
            HierarchyLevel::TechnologicalOscillations => "Technological (10⁹s)",
            HierarchyLevel::CivilizationalOscillations => "Civilizational (10¹²s)",
            HierarchyLevel::CosmicOscillations => "Cosmic (10¹³s)",
        };
        write!(f, "{}", name)
    }
}

/// Result of hierarchy-level processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyResult {
    pub level: HierarchyLevel,
    pub processing_success: bool,
    pub emergence_detected: bool,
    pub coupling_strength: f64,
    pub information_content: f64,
    pub oscillation_phase: OscillationPhase,
    pub cross_scale_interactions: Vec<(HierarchyLevel, f64)>,
    pub computational_cost: f64,
    pub energy_consumption: f64,
    pub coherence_time: f64,
}

/// Emergence detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceResult {
    pub emergent_properties: Vec<String>,
    pub coherence_threshold: f64,
    pub information_integration: f64,
    pub downward_causation_strength: f64,
    pub upward_causation_strength: f64,
    pub emergence_confidence: f64,
}

/// Multi-scale hierarchy processor
#[derive(Debug, Clone)]
pub struct NestedHierarchyProcessor {
    /// Oscillators for each hierarchy level
    level_oscillators: HashMap<HierarchyLevel, UniversalOscillator>,
    /// Cross-level coupling matrix
    coupling_matrix: DMatrix<f64>,
    /// Processing results history
    results_history: Vec<HierarchyResult>,
    /// Emergence detection system
    emergence_detector: EmergenceDetector,
    /// Information integration across scales
    information_integrator: InformationIntegrator,
}

impl NestedHierarchyProcessor {
    /// Create new hierarchy processor
    pub fn new() -> Self {
        let mut level_oscillators = HashMap::new();
        
        // Initialize oscillator for each hierarchy level
        for level in HierarchyLevel::all_levels() {
            let frequency = level.characteristic_frequency();
            let time_scale = level.time_scale_seconds();
            
            // Damping coefficient based on level (higher levels more damped)
            let damping = 0.1 * (level as u8 as f64 / 10.0);
            
            // Create oscillator with level-specific parameters
            let oscillator = UniversalOscillator::new(
                1.0,        // Initial amplitude
                frequency,  // Natural frequency
                damping,    // Damping coefficient
                3,          // 3D dynamics
            ).with_forcing(move |t| {
                // Level-specific forcing function
                match level {
                    HierarchyLevel::QuantumOscillations => {
                        // Quantum vacuum fluctuations
                        0.1 * (t * frequency * 1e6).sin() * (-t / time_scale).exp()
                    },
                    HierarchyLevel::AtomicOscillations => {
                        // Electronic transitions
                        0.3 * (t * frequency).sin()
                    },
                    HierarchyLevel::MolecularOscillations => {
                        // Thermal molecular motion
                        0.5 * (t * frequency).sin() + 0.1 * (t * frequency * 3.0).sin()
                    },
                    HierarchyLevel::CellularOscillations => {
                        // Metabolic rhythms
                        0.7 * (t * frequency).sin() + 0.3 * (t * frequency * 0.5).sin()
                    },
                    HierarchyLevel::OrganismalOscillations => {
                        // Physiological rhythms (heartbeat, breathing)
                        0.8 * (t * frequency).sin() + 0.4 * (t * frequency * 4.0).sin()
                    },
                    HierarchyLevel::CognitiveOscillations => {
                        // Cognitive processing cycles
                        0.6 * (t * frequency).sin() + 0.2 * (t * frequency * 2.0).sin()
                    },
                    HierarchyLevel::SocialOscillations => {
                        // Social interaction patterns
                        0.4 * (t * frequency).sin() + 0.3 * (t * frequency * 7.0).sin()
                    },
                    HierarchyLevel::TechnologicalOscillations => {
                        // Innovation cycles
                        0.3 * (t * frequency).sin() + 0.5 * (t * frequency * 0.1).sin()
                    },
                    HierarchyLevel::CivilizationalOscillations => {
                        // Historical cycles
                        0.2 * (t * frequency).sin() + 0.4 * (t * frequency * 0.01).sin()
                    },
                    HierarchyLevel::CosmicOscillations => {
                        // Cosmic evolution
                        0.1 * (t * frequency).sin()
                    },
                }
            });
            
            level_oscillators.insert(level, oscillator);
        }
        
        // Initialize coupling matrix
        let num_levels = HierarchyLevel::all_levels().len();
        let mut coupling_matrix = DMatrix::zeros(num_levels, num_levels);
        
        for (i, level_i) in HierarchyLevel::all_levels().iter().enumerate() {
            for (j, level_j) in HierarchyLevel::all_levels().iter().enumerate() {
                if i != j {
                    coupling_matrix[(i, j)] = level_i.coupling_strength(*level_j);
                }
            }
        }
        
        Self {
            level_oscillators,
            coupling_matrix,
            results_history: Vec::new(),
            emergence_detector: EmergenceDetector::new(),
            information_integrator: InformationIntegrator::new(),
        }
    }
    
    /// Process information at a specific hierarchy level
    pub fn process_at_level(
        &mut self,
        level: HierarchyLevel,
        input_profile: &OscillationProfile,
        dt: f64,
    ) -> AutobahnResult<HierarchyResult> {
        // Get oscillator for this level
        let oscillator = self.level_oscillators.get_mut(&level)
            .ok_or_else(|| AutobahnError::ProcessingError {
                message: format!("No oscillator found for level {:?}", level),
            })?;
        
        // Evolve oscillator
        oscillator.evolve(dt)?;
        
        // Calculate cross-scale interactions
        let mut cross_scale_interactions = Vec::new();
        for other_level in HierarchyLevel::all_levels() {
            if other_level != level {
                let coupling = level.coupling_strength(other_level);
                if coupling > 0.01 {
                    cross_scale_interactions.push((other_level, coupling));
                }
            }
        }
        
        // Calculate information content at this level
        let information_content = self.calculate_information_content(level, input_profile)?;
        
        // Calculate energy consumption
        let energy_consumption = self.calculate_energy_consumption(level, oscillator)?;
        
        // Calculate computational cost
        let computational_cost = energy_consumption * input_profile.complexity;
        
        // Detect emergence
        let emergence_detected = self.emergence_detector.detect_emergence(
            level,
            oscillator,
            &cross_scale_interactions,
        )?;
        
        let result = HierarchyResult {
            level,
            processing_success: true,
            emergence_detected,
            coupling_strength: input_profile.coupling_strength,
            information_content,
            oscillation_phase: oscillator.calculate_phase(),
            cross_scale_interactions,
            computational_cost,
            energy_consumption,
            coherence_time: self.calculate_coherence_time(level, oscillator)?,
        };
        
        self.results_history.push(result.clone());
        
        // Limit history size
        if self.results_history.len() > 10000 {
            self.results_history.drain(..1000);
        }
        
        Ok(result)
    }
    
    /// Process across multiple hierarchy levels simultaneously
    pub fn process_multi_scale(
        &mut self,
        target_levels: Vec<HierarchyLevel>,
        quantum_profile: &QuantumOscillatoryProfile,
        dt: f64,
    ) -> AutobahnResult<Vec<HierarchyResult>> {
        let mut results = Vec::new();
        
        // Process each level
        for level in target_levels {
            let result = self.process_at_level(level, &quantum_profile.base_oscillation, dt)?;
            results.push(result);
        }
        
        // Apply cross-level coupling
        self.apply_cross_level_coupling(&results, dt)?;
        
        // Integrate information across scales
        let integrated_info = self.information_integrator.integrate_across_scales(&results)?;
        log::debug!("Integrated information: {:.3} bits", integrated_info);
        
        Ok(results)
    }
    
    /// Apply coupling between different hierarchy levels
    fn apply_cross_level_coupling(
        &mut self,
        results: &[HierarchyResult],
        dt: f64,
    ) -> AutobahnResult<()> {
        for result in results {
            let level = result.level;
            let level_oscillator = self.level_oscillators.get_mut(&level)
                .ok_or_else(|| AutobahnError::ProcessingError {
                    message: format!("No oscillator found for level {:?}", level),
                })?;
            
            // Apply coupling forces from other levels
            for (other_level, coupling_strength) in &result.cross_scale_interactions {
                if let Some(other_oscillator) = self.level_oscillators.get(other_level) {
                    // Calculate coupling force
                    let phase_diff = other_oscillator.state.phase - level_oscillator.state.phase;
                    let coupling_force = coupling_strength * phase_diff.sin();
                    
                    // Apply force to level oscillator
                    level_oscillator.state.external_forcing_amplitude += coupling_force * dt;
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate information content at a specific level
    fn calculate_information_content(
        &self,
        level: HierarchyLevel,
        profile: &OscillationProfile,
    ) -> AutobahnResult<f64> {
        // Information content based on level capacity and profile entropy
        let level_capacity = level.information_capacity();
        let profile_entropy = profile.calculate_information_content();
        
        // Scale information by level characteristics
        let scaled_info = profile_entropy * (level_capacity / 1e6).ln().max(1.0);
        
        Ok(scaled_info)
    }
    
    /// Calculate energy consumption for processing at a level
    fn calculate_energy_consumption(
        &self,
        level: HierarchyLevel,
        oscillator: &UniversalOscillator,
    ) -> AutobahnResult<f64> {
        let base_energy = level.energy_scale();
        let oscillator_energy = oscillator.total_energy();
        
        // Energy consumption proportional to oscillator energy and level scale
        let consumption = base_energy * oscillator_energy * 1e-12; // Scale factor
        
        Ok(consumption)
    }
    
    /// Calculate coherence time at a specific level
    fn calculate_coherence_time(
        &self,
        level: HierarchyLevel,
        oscillator: &UniversalOscillator,
    ) -> AutobahnResult<f64> {
        let time_scale = level.time_scale_seconds();
        let quality_factor = oscillator.state.natural_frequency / oscillator.state.damping_coefficient.max(1e-10);
        
        // Coherence time based on level time scale and oscillator quality
        let coherence_time = time_scale * quality_factor * 0.1;
        
        Ok(coherence_time)
    }
    
    /// Get processing results for a specific level
    pub fn get_level_results(&self, level: HierarchyLevel) -> Vec<&HierarchyResult> {
        self.results_history.iter()
            .filter(|result| result.level == level)
            .collect()
    }
    
    /// Get coupling matrix
    pub fn get_coupling_matrix(&self) -> &DMatrix<f64> {
        &self.coupling_matrix
    }
    
    /// Get oscillator state for a level
    pub fn get_level_oscillator(&self, level: HierarchyLevel) -> Option<&UniversalOscillator> {
        self.level_oscillators.get(&level)
    }
}

/// Emergence detection system
#[derive(Debug, Clone)]
pub struct EmergenceDetector {
    coherence_threshold: f64,
    information_threshold: f64,
    coupling_threshold: f64,
}

impl EmergenceDetector {
    pub fn new() -> Self {
        Self {
            coherence_threshold: 0.5,
            information_threshold: 1.0,
            coupling_threshold: 0.1,
        }
    }
    
    pub fn detect_emergence(
        &self,
        level: HierarchyLevel,
        oscillator: &UniversalOscillator,
        cross_interactions: &[(HierarchyLevel, f64)],
    ) -> AutobahnResult<bool> {
        // Check coherence
        let quality_factor = oscillator.state.natural_frequency / oscillator.state.damping_coefficient.max(1e-10);
        let coherence_score = quality_factor / (1.0 + quality_factor);
        
        // Check cross-scale coupling
        let total_coupling: f64 = cross_interactions.iter().map(|(_, strength)| strength).sum();
        
        // Check information integration
        let energy = oscillator.total_energy();
        let information_score = energy.ln().max(0.0);
        
        // Emergence detected if all thresholds exceeded
        let emergence = coherence_score > self.coherence_threshold
            && total_coupling > self.coupling_threshold
            && information_score > self.information_threshold;
        
        log::debug!(
            "Emergence check for {:?}: coherence={:.3}, coupling={:.3}, info={:.3}, emergent={}",
            level, coherence_score, total_coupling, information_score, emergence
        );
        
        Ok(emergence)
    }
}

/// Information integration across scales
#[derive(Debug, Clone)]
pub struct InformationIntegrator {
    integration_window: usize,
}

impl InformationIntegrator {
    pub fn new() -> Self {
        Self {
            integration_window: 100,
        }
    }
    
    pub fn integrate_across_scales(&self, results: &[HierarchyResult]) -> AutobahnResult<f64> {
        if results.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate Φ (phi) - integrated information measure
        let mut phi = 0.0;
        
        for result in results {
            // Information at this level
            let level_info = result.information_content;
            
            // Coupling to other levels
            let coupling_info: f64 = result.cross_scale_interactions
                .iter()
                .map(|(_, strength)| strength * level_info)
                .sum();
            
            // Coherence contribution
            let coherence_factor = result.coherence_time / (1.0 + result.coherence_time);
            
            phi += level_info * coupling_info * coherence_factor;
        }
        
        Ok(phi)
    }
}

// Re-export key types
pub use {
    HierarchyLevel,
    HierarchyResult,
    EmergenceResult,
    NestedHierarchyProcessor,
    EmergenceDetector,
    InformationIntegrator,
}; 