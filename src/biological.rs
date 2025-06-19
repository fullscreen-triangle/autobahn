//! Biological layer processing system implementing the three-layer biological architecture
//! Context -> Reasoning -> Intuition with quantum-enhanced metabolic considerations

use crate::error::{AutobahnError, AutobahnResult};
use crate::oscillatory::{OscillationProfile, OscillationPhase};
use crate::quantum::QuantumOscillatoryProfile;
use crate::atp::{MetabolicMode, ATPState};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The three biological processing layers
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BiologicalLayer {
    /// Context layer - basic information processing and pattern recognition
    Context,
    /// Reasoning layer - logical processing and decision making
    Reasoning,
    /// Intuition layer - highest-level pattern recognition and insight
    Intuition,
}

impl BiologicalLayer {
    /// Get all biological layers in processing order
    pub fn all_layers() -> Vec<BiologicalLayer> {
        vec![
            BiologicalLayer::Context,
            BiologicalLayer::Reasoning,
            BiologicalLayer::Intuition,
        ]
    }
    
    /// Get the metabolic cost multiplier for this layer
    pub fn metabolic_cost_multiplier(&self) -> f64 {
        match self {
            BiologicalLayer::Context => 1.0,    // Base cost
            BiologicalLayer::Reasoning => 2.5,  // Higher reasoning cost
            BiologicalLayer::Intuition => 4.0,  // Highest intuitive cost
        }
    }
    
    /// Get the quantum enhancement factor for this layer
    pub fn quantum_enhancement_factor(&self) -> f64 {
        match self {
            BiologicalLayer::Context => 1.1,    // Minor quantum enhancement
            BiologicalLayer::Reasoning => 1.3,  // Moderate quantum enhancement
            BiologicalLayer::Intuition => 1.5,  // Maximum quantum enhancement
        }
    }
    
    /// Get the oscillation sensitivity for this layer
    pub fn oscillation_sensitivity(&self) -> f64 {
        match self {
            BiologicalLayer::Context => 0.3,    // Low sensitivity to oscillations
            BiologicalLayer::Reasoning => 0.6,  // Moderate sensitivity
            BiologicalLayer::Intuition => 0.9,  // High sensitivity to quantum oscillations
        }
    }
}

/// Result of biological layer processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalProcessingResult {
    pub layer: BiologicalLayer,
    pub processing_success: bool,
    pub output_quality: f64,
    pub metabolic_cost: f64,
    pub quantum_enhancement: f64,
    pub oscillation_coherence: f64,
    pub information_content: f64,
    pub processing_time_ms: f64,
}

/// Biological layer processor
#[derive(Debug, Clone)]
pub struct BiologicalLayerProcessor {
    /// Current oscillation profiles for each layer
    layer_profiles: HashMap<BiologicalLayer, OscillationProfile>,
    /// Quantum profiles for quantum-enhanced processing
    quantum_profiles: HashMap<BiologicalLayer, QuantumOscillatoryProfile>,
    /// Current metabolic mode affecting all layers
    metabolic_mode: MetabolicMode,
    /// Processing history for analysis
    processing_history: HashMap<BiologicalLayer, Vec<BiologicalProcessingResult>>,
}

impl BiologicalLayerProcessor {
    /// Create new biological layer processor
    pub fn new(temperature_k: f64) -> Self {
        let mut layer_profiles = HashMap::new();
        let mut quantum_profiles = HashMap::new();
        let mut processing_history = HashMap::new();
        
        // Initialize profiles for each biological layer
        for layer in BiologicalLayer::all_layers() {
            let complexity = layer.metabolic_cost_multiplier();
            let frequency = match layer {
                BiologicalLayer::Context => 1.0,      // 1 Hz base processing
                BiologicalLayer::Reasoning => 0.1,    // 0.1 Hz reasoning cycles
                BiologicalLayer::Intuition => 0.01,   // 0.01 Hz insight cycles
            };
            
            let oscillation_profile = OscillationProfile::new(complexity, frequency);
            let quantum_profile = QuantumOscillatoryProfile::new(
                oscillation_profile.clone(),
                temperature_k,
            );
            
            layer_profiles.insert(layer, oscillation_profile);
            quantum_profiles.insert(layer, quantum_profile);
            processing_history.insert(layer, Vec::new());
        }
        
        // Default to mammalian metabolism
        let metabolic_mode = MetabolicMode::MammalianBurden {
            quantum_cost_multiplier: 1.2,
            radical_generation_rate: 1e-5,
        };
        
        Self {
            layer_profiles,
            quantum_profiles,
            metabolic_mode,
            processing_history,
        }
    }
    
    /// Process input through a specific biological layer
    pub fn process_at_layer(
        &mut self,
        layer: BiologicalLayer,
        input: &str,
        complexity: f64,
    ) -> AutobahnResult<BiologicalProcessingResult> {
        let start_time = std::time::Instant::now();
        
        // Get current profiles
        let mut oscillation_profile = self.layer_profiles[&layer].clone();
        let mut quantum_profile = self.quantum_profiles[&layer].clone();
        
        // Update profiles based on input complexity
        oscillation_profile.complexity = complexity;
        quantum_profile.base_oscillation.complexity = complexity;
        
        // Calculate processing parameters
        let base_cost = layer.metabolic_cost_multiplier() * complexity;
        let quantum_enhancement = layer.quantum_enhancement_factor();
        let oscillation_sensitivity = layer.oscillation_sensitivity();
        
        // Apply metabolic mode effects
        let metabolic_cost = self.apply_metabolic_mode_effects(base_cost)?;
        
        // Calculate quantum-enhanced processing cost
        let quantum_cost = quantum_profile.calculate_quantum_enhanced_atp_cost(
            metabolic_cost,
            complexity,
        )?;
        
        // Determine oscillation coherence based on layer sensitivity
        let oscillation_coherence = self.calculate_oscillation_coherence(
            &oscillation_profile,
            oscillation_sensitivity,
        );
        
        // Calculate information content
        oscillation_profile.add_termination_point(
            format!("{}:{}", layer as u8, input.len()),
            complexity / 10.0,
        );
        let information_content = oscillation_profile.calculate_information_content();
        
        // Determine output quality based on quantum enhancement and coherence
        let output_quality = self.calculate_output_quality(
            quantum_enhancement,
            oscillation_coherence,
            information_content,
        );
        
        // Check processing success
        let processing_success = output_quality > 0.5 && oscillation_coherence > 0.3;
        
        let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        let result = BiologicalProcessingResult {
            layer,
            processing_success,
            output_quality,
            metabolic_cost: quantum_cost,
            quantum_enhancement,
            oscillation_coherence,
            information_content,
            processing_time_ms: processing_time,
        };
        
        // Update profiles
        self.layer_profiles.insert(layer, oscillation_profile);
        self.quantum_profiles.insert(layer, quantum_profile);
        
        // Store in processing history
        self.processing_history.get_mut(&layer).unwrap().push(result.clone());
        
        // Limit history size
        if self.processing_history[&layer].len() > 1000 {
            self.processing_history.get_mut(&layer).unwrap().drain(..100);
        }
        
        log::debug!("Processed at layer {:?}: success={}, quality={:.2}, cost={:.2}",
                   layer, processing_success, output_quality, quantum_cost);
        
        Ok(result)
    }
    
    /// Apply metabolic mode effects to base cost
    fn apply_metabolic_mode_effects(&self, base_cost: f64) -> AutobahnResult<f64> {
        let modified_cost = match &self.metabolic_mode {
            MetabolicMode::SustainedFlight { efficiency_boost, .. } => {
                base_cost / efficiency_boost
            },
            MetabolicMode::ColdBlooded { temperature_advantage, metabolic_reduction } => {
                base_cost * metabolic_reduction / temperature_advantage
            },
            MetabolicMode::MammalianBurden { quantum_cost_multiplier, .. } => {
                base_cost * quantum_cost_multiplier
            },
            MetabolicMode::AnaerobicEmergency { efficiency_penalty, .. } => {
                base_cost * (1.0 + efficiency_penalty)
            },
        };
        
        Ok(modified_cost)
    }
    
    /// Calculate oscillation coherence for a layer
    fn calculate_oscillation_coherence(
        &self,
        profile: &OscillationProfile,
        sensitivity: f64,
    ) -> f64 {
        let base_coherence = match profile.phase {
            OscillationPhase::Peak => 0.9,         // Maximum coherence
            OscillationPhase::Resonance => 1.0,    // Perfect coherence
            OscillationPhase::Acceleration => 0.7, // Building coherence
            OscillationPhase::Decay => 0.4,        // Losing coherence
            OscillationPhase::Equilibrium => 0.6,  // Moderate coherence
        };
        
        // Apply sensitivity factor
        let coherence = base_coherence * sensitivity;
        
        // Apply quality factor enhancement
        let quality_enhancement = if profile.quality_factor > 10.0 {
            1.2
        } else if profile.quality_factor > 5.0 {
            1.1
        } else {
            1.0
        };
        
        (coherence * quality_enhancement).min(1.0)
    }
    
    /// Calculate output quality based on quantum and oscillation factors
    fn calculate_output_quality(
        &self,
        quantum_enhancement: f64,
        oscillation_coherence: f64,
        information_content: f64,
    ) -> f64 {
        let base_quality = 0.7; // Base processing quality
        
        // Apply quantum enhancement
        let quantum_quality = base_quality * quantum_enhancement;
        
        // Apply oscillation coherence
        let coherent_quality = quantum_quality * oscillation_coherence;
        
        // Apply information content bonus
        let info_bonus = 1.0 + (information_content / 5.0).min(0.3);
        let final_quality = coherent_quality * info_bonus;
        
        final_quality.min(1.0)
    }
    
    /// Process input through all biological layers in sequence
    pub fn process_all_layers(
        &mut self,
        input: &str,
        complexity: f64,
    ) -> AutobahnResult<Vec<BiologicalProcessingResult>> {
        let mut results = Vec::new();
        let mut processed_complexity = complexity;
        
        // Process through each layer in order
        for layer in BiologicalLayer::all_layers() {
            let result = self.process_at_layer(layer, input, processed_complexity)?;
            
            // Adjust complexity for next layer based on output quality
            processed_complexity *= result.output_quality;
            
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Update metabolic mode
    pub fn update_metabolic_mode(&mut self, new_mode: MetabolicMode) {
        if new_mode != self.metabolic_mode {
            log::info!("Biological layer processor: metabolic mode changed from {:?} to {:?}",
                      self.metabolic_mode, new_mode);
            self.metabolic_mode = new_mode;
        }
    }
    
    /// Get processing statistics for all layers
    pub fn get_processing_statistics(&self) -> BiologicalProcessingStatistics {
        let mut layer_stats = HashMap::new();
        
        for layer in BiologicalLayer::all_layers() {
            let history = &self.processing_history[&layer];
            
            let total_processed = history.len();
            let success_count = history.iter().filter(|r| r.processing_success).count();
            let avg_quality = if !history.is_empty() {
                history.iter().map(|r| r.output_quality).sum::<f64>() / history.len() as f64
            } else {
                0.0
            };
            let avg_cost = if !history.is_empty() {
                history.iter().map(|r| r.metabolic_cost).sum::<f64>() / history.len() as f64
            } else {
                0.0
            };
            let avg_coherence = if !history.is_empty() {
                history.iter().map(|r| r.oscillation_coherence).sum::<f64>() / history.len() as f64
            } else {
                0.0
            };
            
            layer_stats.insert(layer, LayerStatistics {
                total_processed,
                success_count,
                success_rate: if total_processed > 0 {
                    success_count as f64 / total_processed as f64
                } else {
                    0.0
                },
                average_output_quality: avg_quality,
                average_metabolic_cost: avg_cost,
                average_oscillation_coherence: avg_coherence,
            });
        }
        
        BiologicalProcessingStatistics {
            layer_statistics: layer_stats,
            current_metabolic_mode: self.metabolic_mode.clone(),
        }
    }
}

/// Statistics for biological processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalProcessingStatistics {
    pub layer_statistics: HashMap<BiologicalLayer, LayerStatistics>,
    pub current_metabolic_mode: MetabolicMode,
}

/// Statistics for a specific biological layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStatistics {
    pub total_processed: usize,
    pub success_count: usize,
    pub success_rate: f64,
    pub average_output_quality: f64,
    pub average_metabolic_cost: f64,
    pub average_oscillation_coherence: f64,
} 