//! ATP resource management system implementing quantum-enhanced metabolic pathways.
//! Based on the principle that ATP synthase functions as a biological quantum computer.

pub mod manager;
pub mod quantum_manager;
pub mod optimization;
pub mod economics;

use crate::error::{AutobahnError, AutobahnResult};
use crate::oscillatory::{OscillationPhase, OscillationProfile};
use crate::biological::BiologicalLayer;
use crate::quantum::QuantumMembraneState;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::sync::Arc;

/// ATP state tracking with quantum considerations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATPState {
    /// Current ATP available
    pub current: f64,
    /// Maximum ATP capacity
    pub maximum: f64,
    /// Reserved ATP for critical functions
    pub reserved: f64,
    /// Emergency threshold for anaerobic activation
    pub emergency_threshold: f64,
    /// ATP regeneration rate (quantum-enhanced)
    pub regeneration_rate: f64,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
    /// Quantum efficiency factor
    pub quantum_efficiency: f64,
    /// Accumulated radical damage
    pub radical_damage: f64,
}

impl ATPState {
    pub fn new(maximum: f64) -> Self {
        Self {
            current: maximum * 0.75,     // Start at 75% capacity
            maximum,
            reserved: maximum * 0.15,    // Reserve 15% for emergencies
            emergency_threshold: maximum * 0.1, // Emergency at 10%
            regeneration_rate: maximum * 0.025,  // 2.5% per minute base rate
            last_update: Utc::now(),
            quantum_efficiency: 0.95,    // Default quantum efficiency
            radical_damage: 0.0,
        }
    }
    
    /// Calculate available ATP (current - reserved)
    pub fn available(&self) -> f64 {
        (self.current - self.reserved).max(0.0)
    }
    
    /// Calculate ATP percentage of maximum
    pub fn percentage(&self) -> f64 {
        (self.current / self.maximum) * 100.0
    }
    
    /// Check if ATP is at critical levels
    pub fn is_critical(&self) -> bool {
        self.current < self.emergency_threshold
    }
    
    /// Check if radical damage is affecting ATP production
    pub fn is_damage_critical(&self) -> bool {
        self.radical_damage > 50.0 // Arbitrary damage threshold
    }
}

/// Metabolic modes based on quantum mechanical principles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetabolicMode {
    /// Sustained flight metabolism - maximum quantum efficiency
    SustainedFlight {
        efficiency_boost: f64,
        radical_suppression: f64,
    },
    /// Cold-blooded metabolism - temperature advantage
    ColdBlooded {
        temperature_advantage: f64,
        metabolic_reduction: f64,
    },
    /// Mammalian burden - high temperature, high metabolic cost
    MammalianBurden {
        quantum_cost_multiplier: f64,
        radical_generation_rate: f64,
    },
    /// Anaerobic emergency - lactate pathway activation
    AnaerobicEmergency {
        lactate_pathway_active: bool,
        efficiency_penalty: f64,
    },
}

/// Layer-specific ATP requirements with quantum considerations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerResourceRequirement {
    pub layer: BiologicalLayer,
    pub base_atp_cost: f64,
    pub complexity_multiplier: f64,
    pub minimum_atp: f64,
    pub maximum_atp: f64,
    pub expected_yield: f64,
    pub oscillation_sensitivity: f64,
    pub quantum_enhancement_factor: f64,
}

/// Basic oscillatory ATP manager
pub struct OscillatoryATPManager {
    state: Arc<RwLock<ATPState>>,
    consumption_history: Arc<RwLock<Vec<(DateTime<Utc>, String, f64, f64)>>>,
    layer_requirements: HashMap<BiologicalLayer, LayerResourceRequirement>,
    cosmic_oscillation_phase: Arc<RwLock<OscillationPhase>>,
}

impl OscillatoryATPManager {
    pub fn new(maximum_atp: f64) -> Self {
        let mut layer_requirements = HashMap::new();
        
        // Context layer - basic information processing
        layer_requirements.insert(
            BiologicalLayer::Context,
            LayerResourceRequirement {
                layer: BiologicalLayer::Context,
                base_atp_cost: 100.0,
                complexity_multiplier: 1.2,
                minimum_atp: 50.0,
                maximum_atp: 300.0,
                expected_yield: 0.7,
                oscillation_sensitivity: 0.3,
                quantum_enhancement_factor: 1.1,
            }
        );
        
        // Reasoning layer - complex logical processing
        layer_requirements.insert(
            BiologicalLayer::Reasoning,
            LayerResourceRequirement {
                layer: BiologicalLayer::Reasoning,
                base_atp_cost: 300.0,
                complexity_multiplier: 1.8,
                minimum_atp: 150.0,
                maximum_atp: 800.0,
                expected_yield: 0.85,
                oscillation_sensitivity: 0.6,
                quantum_enhancement_factor: 1.3,
            }
        );
        
        // Intuition layer - highest-level pattern recognition
        layer_requirements.insert(
            BiologicalLayer::Intuition,
            LayerResourceRequirement {
                layer: BiologicalLayer::Intuition,
                base_atp_cost: 500.0,
                complexity_multiplier: 2.5,
                minimum_atp: 300.0,
                maximum_atp: 1200.0,
                expected_yield: 0.95,
                oscillation_sensitivity: 0.9,
                quantum_enhancement_factor: 1.5,
            }
        );
        
        Self {
            state: Arc::new(RwLock::new(ATPState::new(maximum_atp))),
            consumption_history: Arc::new(RwLock::new(Vec::new())),
            layer_requirements,
            cosmic_oscillation_phase: Arc::new(RwLock::new(OscillationPhase::Equilibrium)),
        }
    }
    
    /// Calculate oscillatory ATP cost with phase and complexity modulation
    pub async fn calculate_oscillatory_atp_cost(
        &self,
        layer: BiologicalLayer,
        query_complexity: f64,
        oscillation_profile: &OscillationProfile,
    ) -> AutobahnResult<f64> {
        let req = self.layer_requirements.get(&layer)
            .ok_or_else(|| AutobahnError::ConfigurationError(
                format!("Unknown biological layer: {:?}", layer)
            ))?;
        
        // Base cost calculation
        let base_cost = req.base_atp_cost;
        let complexity_cost = base_cost * query_complexity.powf(req.complexity_multiplier);
        
        // Oscillatory phase modulation
        let phase_multiplier = match oscillation_profile.phase {
            OscillationPhase::Acceleration => 2.5, // High energy during buildup
            OscillationPhase::Peak => 0.8,         // Maximum efficiency at peak
            OscillationPhase::Decay => 1.5,        // Increased cost during decay
            OscillationPhase::Equilibrium => 1.0,  // Baseline cost
            OscillationPhase::Resonance => 0.6,    // Highly efficient during resonance
        };
        
        // Frequency modulation (biological systems have optimal frequency ranges)
        let frequency_multiplier = if oscillation_profile.frequency > 10.0 {
            1.3 // Higher frequency requires more energy coordination
        } else if oscillation_profile.frequency < 0.1 {
            1.2 // Very low frequency is also less efficient
        } else {
            1.0 // Optimal frequency range
        };
        
        // Coupling strength modulation
        let coupling_multiplier = 1.0 + (oscillation_profile.coupling_strength - 0.5) * 0.4;
        
        // Quality factor consideration (higher Q = more efficient)
        let quality_multiplier = 1.0 / (1.0 + oscillation_profile.quality_factor / 20.0);
        
        let total_cost = complexity_cost * phase_multiplier * frequency_multiplier 
                        * coupling_multiplier * quality_multiplier;
        
        // Apply quantum enhancement
        let quantum_enhanced_cost = total_cost / req.quantum_enhancement_factor;
        
        // Apply bounds
        Ok(quantum_enhanced_cost.clamp(req.minimum_atp, req.maximum_atp))
    }
    
    /// Consume ATP for a specific operation
    pub async fn consume_atp(
        &self,
        layer: BiologicalLayer,
        amount: f64,
        operation: &str,
    ) -> AutobahnResult<bool> {
        let mut state = self.state.write().await;
        
        if state.available() >= amount {
            state.current -= amount;
            
            // Record consumption in history
            let mut history = self.consumption_history.write().await;
            history.push((
                Utc::now(),
                format!("{:?}:{}", layer, operation),
                amount,
                state.current,
            ));
            
            // Limit history size to prevent memory bloat
            if history.len() > 10000 {
                history.drain(..1000);
            }
            
            log::debug!("Consumed {:.1} ATP for {:?}:{}, remaining: {:.1}", 
                       amount, layer, operation, state.current);
            
            Ok(true)
        } else {
            // Handle ATP shortage
            Err(AutobahnError::InsufficientATP {
                required: amount,
                available: state.available(),
            })
        }
    }
    
    /// Regenerate ATP over time
    pub async fn regenerate_atp(&self, duration_minutes: f64) -> AutobahnResult<()> {
        let mut state = self.state.write().await;
        
        let time_elapsed = Utc::now()
            .signed_duration_since(state.last_update)
            .num_minutes() as f64;
        
        // Calculate regeneration amount
        let regeneration_time = duration_minutes.min(time_elapsed);
        let base_regeneration = state.regeneration_rate * regeneration_time;
        
        // Apply quantum efficiency boost
        let quantum_regeneration = base_regeneration * state.quantum_efficiency;
        
        // Apply radical damage penalty
        let damage_penalty = 1.0 / (1.0 + state.radical_damage * 0.01);
        let final_regeneration = quantum_regeneration * damage_penalty;
        
        state.current = (state.current + final_regeneration).min(state.maximum);
        state.last_update = Utc::now();
        
        log::debug!("Regenerated {:.1} ATP (base: {:.1}, quantum: {:.1}, damage penalty: {:.2})",
                   final_regeneration, base_regeneration, quantum_regeneration, damage_penalty);
        
        Ok(())
    }
    
    /// Get current ATP state
    pub async fn get_state(&self) -> ATPState {
        self.state.read().await.clone()
    }
    
    /// Get ATP consumption history
    pub async fn get_consumption_history(&self) -> Vec<(DateTime<Utc>, String, f64, f64)> {
        self.consumption_history.read().await.clone()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATPEfficiencyMetrics {
    pub total_consumed: f64,
    pub average_consumption: f64,
    pub utilization_rate: f64,
    pub quantum_efficiency: f64,
    pub radical_damage_impact: f64,
    pub regeneration_rate: f64,
} 