//! ATP resource management system implementing quantum-enhanced metabolic pathways.
//! Based on the principle that ATP synthase functions as a biological quantum computer.

use crate::error::{AutobahnError, AutobahnResult};
use crate::oscillatory::{OscillationPhase, OscillationProfile, UniversalOscillator};
use crate::quantum::{QuantumOscillatoryProfile, QuantumMembraneState, ENAQTProcessor};
use crate::hierarchy::HierarchyLevel;
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

impl MetabolicMode {
    /// Get efficiency multiplier for this mode
    pub fn efficiency_multiplier(&self) -> f64 {
        match self {
            MetabolicMode::SustainedFlight { efficiency_boost, .. } => *efficiency_boost,
            MetabolicMode::ColdBlooded { temperature_advantage, metabolic_reduction } => {
                temperature_advantage * metabolic_reduction
            },
            MetabolicMode::MammalianBurden { quantum_cost_multiplier, .. } => {
                1.0 / quantum_cost_multiplier
            },
            MetabolicMode::AnaerobicEmergency { efficiency_penalty, .. } => {
                1.0 / (1.0 + efficiency_penalty)
            },
        }
    }
    
    /// Get ATP cost multiplier for this mode
    pub fn cost_multiplier(&self) -> f64 {
        1.0 / self.efficiency_multiplier()
    }
    
    /// Get radical generation factor
    pub fn radical_generation_factor(&self) -> f64 {
        match self {
            MetabolicMode::SustainedFlight { radical_suppression, .. } => *radical_suppression,
            MetabolicMode::ColdBlooded { .. } => 0.5, // Cold reduces radical generation
            MetabolicMode::MammalianBurden { radical_generation_rate, .. } => *radical_generation_rate,
            MetabolicMode::AnaerobicEmergency { .. } => 2.0, // High radical generation in emergency
        }
    }
}

/// Layer-specific ATP requirements with quantum considerations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerResourceRequirement {
    pub hierarchy_level: HierarchyLevel,
    pub base_atp_cost: f64,
    pub complexity_multiplier: f64,
    pub minimum_atp: f64,
    pub maximum_atp: f64,
    pub expected_yield: f64,
    pub oscillation_sensitivity: f64,
    pub quantum_enhancement_factor: f64,
}

impl LayerResourceRequirement {
    pub fn for_level(level: HierarchyLevel) -> Self {
        match level {
            HierarchyLevel::QuantumOscillations => Self {
                hierarchy_level: level,
                base_atp_cost: 10.0,
                complexity_multiplier: 3.0,
                minimum_atp: 5.0,
                maximum_atp: 50.0,
                expected_yield: 0.99,
                oscillation_sensitivity: 0.9,
                quantum_enhancement_factor: 2.0,
            },
            HierarchyLevel::AtomicOscillations => Self {
                hierarchy_level: level,
                base_atp_cost: 25.0,
                complexity_multiplier: 2.5,
                minimum_atp: 10.0,
                maximum_atp: 100.0,
                expected_yield: 0.95,
                oscillation_sensitivity: 0.8,
                quantum_enhancement_factor: 1.8,
            },
            HierarchyLevel::MolecularOscillations => Self {
                hierarchy_level: level,
                base_atp_cost: 50.0,
                complexity_multiplier: 2.0,
                minimum_atp: 25.0,
                maximum_atp: 200.0,
                expected_yield: 0.9,
                oscillation_sensitivity: 0.7,
                quantum_enhancement_factor: 1.5,
            },
            HierarchyLevel::CellularOscillations => Self {
                hierarchy_level: level,
                base_atp_cost: 100.0,
                complexity_multiplier: 1.8,
                minimum_atp: 50.0,
                maximum_atp: 400.0,
                expected_yield: 0.85,
                oscillation_sensitivity: 0.6,
                quantum_enhancement_factor: 1.3,
            },
            HierarchyLevel::OrganismalOscillations => Self {
                hierarchy_level: level,
                base_atp_cost: 200.0,
                complexity_multiplier: 1.5,
                minimum_atp: 100.0,
                maximum_atp: 600.0,
                expected_yield: 0.8,
                oscillation_sensitivity: 0.5,
                quantum_enhancement_factor: 1.2,
            },
            HierarchyLevel::CognitiveOscillations => Self {
                hierarchy_level: level,
                base_atp_cost: 300.0,
                complexity_multiplier: 1.8,
                minimum_atp: 150.0,
                maximum_atp: 800.0,
                expected_yield: 0.85,
                oscillation_sensitivity: 0.6,
                quantum_enhancement_factor: 1.3,
            },
            HierarchyLevel::SocialOscillations => Self {
                hierarchy_level: level,
                base_atp_cost: 400.0,
                complexity_multiplier: 2.0,
                minimum_atp: 200.0,
                maximum_atp: 1000.0,
                expected_yield: 0.9,
                oscillation_sensitivity: 0.7,
                quantum_enhancement_factor: 1.4,
            },
            HierarchyLevel::TechnologicalOscillations => Self {
                hierarchy_level: level,
                base_atp_cost: 600.0,
                complexity_multiplier: 2.5,
                minimum_atp: 300.0,
                maximum_atp: 1500.0,
                expected_yield: 0.95,
                oscillation_sensitivity: 0.8,
                quantum_enhancement_factor: 1.6,
            },
            HierarchyLevel::CivilizationalOscillations => Self {
                hierarchy_level: level,
                base_atp_cost: 800.0,
                complexity_multiplier: 3.0,
                minimum_atp: 400.0,
                maximum_atp: 2000.0,
                expected_yield: 0.98,
                oscillation_sensitivity: 0.9,
                quantum_enhancement_factor: 1.8,
            },
            HierarchyLevel::CosmicOscillations => Self {
                hierarchy_level: level,
                base_atp_cost: 1000.0,
                complexity_multiplier: 4.0,
                minimum_atp: 500.0,
                maximum_atp: 3000.0,
                expected_yield: 0.99,
                oscillation_sensitivity: 1.0,
                quantum_enhancement_factor: 2.0,
            },
        }
    }
}

/// Quantum-enhanced ATP manager with metabolic mode switching
#[derive(Debug)]
pub struct QuantumATPManager {
    /// Current ATP state
    state: Arc<RwLock<ATPState>>,
    /// ATP consumption history
    consumption_history: Arc<RwLock<Vec<(DateTime<Utc>, String, f64, f64)>>>,
    /// Layer requirements for each hierarchy level
    layer_requirements: HashMap<HierarchyLevel, LayerResourceRequirement>,
    /// Current metabolic mode
    metabolic_mode: Arc<RwLock<MetabolicMode>>,
    /// Metabolic mode history
    mode_history: Arc<RwLock<Vec<(DateTime<Utc>, MetabolicMode)>>>,
    /// ATP synthase modeled as quantum oscillator
    atp_synthase_oscillator: Arc<RwLock<UniversalOscillator>>,
    /// Quantum membrane state for ATP production
    quantum_membrane: Arc<RwLock<QuantumMembraneState>>,
    /// ENAQT processor for efficiency calculations
    enaqt_processor: Arc<RwLock<ENAQTProcessor>>,
    /// Accumulated radical damage over time
    radical_damage_accumulator: Arc<RwLock<f64>>,
    /// Temperature controller
    temperature_controller: Arc<RwLock<TemperatureController>>,
}

impl QuantumATPManager {
    /// Create new quantum ATP manager
    pub fn new(maximum_atp: f64, operating_temperature: f64) -> Self {
        let mut layer_requirements = HashMap::new();
        for level in HierarchyLevel::all_levels() {
            layer_requirements.insert(level, LayerResourceRequirement::for_level(level));
        }
        
        // ATP synthase as quantum oscillator (rotational frequency ~100 Hz)
        let atp_synthase_oscillator = UniversalOscillator::new(
            1.0,    // Initial amplitude
            100.0,  // Natural frequency (Hz) - ATP synthase rotation
            0.1,    // Damping coefficient
            3,      // 3D rotational dynamics
        ).with_forcing(|t| {
            // Proton gradient forcing function
            0.5 * (2.0 * std::f64::consts::PI * 10.0 * t).sin() // 10 Hz proton oscillation
        });
        
        let quantum_membrane = QuantumMembraneState::new(operating_temperature);
        let enaqt_processor = ENAQTProcessor::new(10);
        
        // Determine initial metabolic mode based on temperature
        let initial_mode = if operating_temperature < 290.0 {
            MetabolicMode::ColdBlooded {
                temperature_advantage: Self::calculate_temperature_advantage(operating_temperature),
                metabolic_reduction: 0.5,
            }
        } else if operating_temperature > 310.0 {
            MetabolicMode::MammalianBurden {
                quantum_cost_multiplier: 1.5,
                radical_generation_rate: 1e-5,
            }
        } else {
            MetabolicMode::MammalianBurden {
                quantum_cost_multiplier: 1.2,
                radical_generation_rate: 8e-6,
            }
        };
        
        Self {
            state: Arc::new(RwLock::new(ATPState::new(maximum_atp))),
            consumption_history: Arc::new(RwLock::new(Vec::new())),
            layer_requirements,
            metabolic_mode: Arc::new(RwLock::new(initial_mode)),
            mode_history: Arc::new(RwLock::new(Vec::new())),
            atp_synthase_oscillator: Arc::new(RwLock::new(atp_synthase_oscillator)),
            quantum_membrane: Arc::new(RwLock::new(quantum_membrane)),
            enaqt_processor: Arc::new(RwLock::new(enaqt_processor)),
            radical_damage_accumulator: Arc::new(RwLock::new(0.0)),
            temperature_controller: Arc::new(RwLock::new(TemperatureController::new(operating_temperature))),
        }
    }
    
    /// Calculate quantum-enhanced ATP cost with full metabolic modeling
    pub async fn calculate_quantum_atp_cost(
        &self,
        level: HierarchyLevel,
        query_complexity: f64,
        quantum_profile: &QuantumOscillatoryProfile,
    ) -> AutobahnResult<f64> {
        let req = self.layer_requirements.get(&level)
            .ok_or_else(|| AutobahnError::ProcessingError {
                message: format!("No requirements found for level {:?}", level),
            })?;
        
        // Base cost calculation
        let base_cost = req.base_atp_cost;
        let complexity_cost = base_cost * query_complexity.powf(req.complexity_multiplier);
        
        // Quantum membrane efficiency modulation
        let enaqt = self.enaqt_processor.read().await;
        let membrane = self.quantum_membrane.read().await;
        let quantum_efficiency = enaqt.calculate_transport_efficiency(
            membrane.enaqt_coupling_strength,
            membrane.temperature_k,
        )?;
        drop(enaqt);
        drop(membrane);
        
        // Current metabolic mode modulation
        let mode = self.metabolic_mode.read().await;
        let mode_factor = mode.cost_multiplier();
        drop(mode);
        
        // ATP synthase quantum oscillation efficiency
        let synthase = self.atp_synthase_oscillator.read().await;
        let synthase_phase = synthase.calculate_phase();
        let synthase_efficiency = match synthase_phase {
            OscillationPhase::Peak => 0.6,      // Maximum quantum efficiency
            OscillationPhase::Resonance => 0.5, // Optimal ENAQT coupling
            OscillationPhase::Acceleration => 0.8,
            OscillationPhase::Decay => 1.2,
            OscillationPhase::Equilibrium => 1.0,
        };
        drop(synthase);
        
        // Oscillatory phase modulation
        let phase_factor = match quantum_profile.base_oscillation.phase {
            OscillationPhase::Peak => 0.7,         // Maximum quantum coherence
            OscillationPhase::Resonance => 0.5,    // Optimal ENAQT coupling
            OscillationPhase::Acceleration => 1.3, // Building coherence costs energy
            OscillationPhase::Decay => 1.8,        // Losing coherence is inefficient
            OscillationPhase::Equilibrium => 1.0,  // Baseline efficiency
        };
        
        // Radical damage penalty
        let damage = *self.radical_damage_accumulator.read().await;
        let damage_penalty = 1.0 + (damage * 0.001);
        
        // Temperature optimization
        let temp_controller = self.temperature_controller.read().await;
        let temp_factor = temp_controller.calculate_efficiency_factor();
        drop(temp_controller);
        
        let quantum_cost = complexity_cost * mode_factor * synthase_efficiency * damage_penalty * phase_factor
                          / (quantum_efficiency * temp_factor * req.quantum_enhancement_factor);
        
        Ok(quantum_cost.clamp(req.minimum_atp, req.maximum_atp))
    }
    
    /// Process operation with full quantum membrane modeling
    pub async fn process_with_quantum_membrane(
        &mut self,
        operation: &str,
        energy_demand: f64,
        dt: f64,
    ) -> AutobahnResult<QuantumProcessingResult> {
        // Evolve ATP synthase oscillator
        {
            let mut synthase = self.atp_synthase_oscillator.write().await;
            synthase.evolve(dt)?;
        }
        
        // Update quantum membrane state
        {
            let mut membrane = self.quantum_membrane.write().await;
            membrane.update_state(dt)?;
        }
        
        // Calculate quantum tunneling and radical generation
        let (tunneling_prob, radical_rate, quantum_efficiency) = {
            let enaqt = self.enaqt_processor.read().await;
            let membrane = self.quantum_membrane.read().await;
            
            let tunneling_prob = enaqt.calculate_tunneling_probability(
                1.0, // 1 eV barrier height (typical for biological systems)
                membrane.membrane_thickness_nm,
            )?;
            
            let radical_rate = membrane.radical_generation_rate;
            
            let quantum_efficiency = enaqt.calculate_transport_efficiency(
                membrane.enaqt_coupling_strength,
                membrane.temperature_k,
            )?;
            
            (tunneling_prob, radical_rate, quantum_efficiency)
        };
        
        // Accumulate radical damage over time
        {
            let mut damage = self.radical_damage_accumulator.write().await;
            *damage += radical_rate * dt;
        }
        
        // Check for metabolic mode transitions
        self.update_metabolic_mode(energy_demand).await?;
        
        // Calculate actual ATP generated with quantum enhancement
        let base_atp = energy_demand;
        let quantum_enhanced_atp = base_atp * quantum_efficiency;
        
        // Apply metabolic mode efficiency
        let mode = self.metabolic_mode.read().await;
        let final_atp = quantum_enhanced_atp * mode.efficiency_multiplier();
        let current_mode = mode.clone();
        drop(mode);
        
        // Consume ATP
        let consumption_success = self.consume_atp(
            HierarchyLevel::CognitiveOscillations, // Default level for quantum processing
            final_atp,
            operation,
        ).await?;
        
        let synthase_phase = {
            let synthase = self.atp_synthase_oscillator.read().await;
            synthase.calculate_phase()
        };
        
        let temperature_k = {
            let membrane = self.quantum_membrane.read().await;
            membrane.temperature_k
        };
        
        let coherence_time_fs = {
            let membrane = self.quantum_membrane.read().await;
            membrane.coherence_time_fs
        };
        
        Ok(QuantumProcessingResult {
            atp_generated: final_atp,
            quantum_efficiency,
            radical_damage_rate: radical_rate,
            tunneling_probability: tunneling_prob,
            metabolic_mode: current_mode,
            consumption_success,
            coherence_time_fs,
            synthase_phase,
            temperature_k,
        })
    }
    
    /// Update metabolic mode based on energy demand and system state
    async fn update_metabolic_mode(&mut self, energy_demand: f64) -> AutobahnResult<()> {
        let current_atp_state = self.get_state().await;
        let current_mode = self.metabolic_mode.read().await.clone();
        let temperature_k = {
            let membrane = self.quantum_membrane.read().await;
            membrane.temperature_k
        };
        
        let new_mode = if energy_demand > 15.0 && current_atp_state.percentage() > 70.0 {
            // High energy demand with sufficient ATP - switch to sustained flight
            MetabolicMode::SustainedFlight {
                efficiency_boost: 2.5,
                radical_suppression: 0.3,
            }
        } else if current_atp_state.is_critical() {
            // ATP critical - emergency anaerobic mode
            MetabolicMode::AnaerobicEmergency {
                lactate_pathway_active: true,
                efficiency_penalty: 1.8,
            }
        } else if energy_demand < 3.0 && temperature_k < 290.0 {
            // Low demand, low temperature - cold-blooded advantage
            MetabolicMode::ColdBlooded {
                temperature_advantage: Self::calculate_temperature_advantage(temperature_k),
                metabolic_reduction: 0.6,
            }
        } else {
            // Default mammalian metabolism
            let damage = *self.radical_damage_accumulator.read().await;
            let radical_rate = if damage > 10.0 { 1.5e-5 } else { 1e-5 };
            MetabolicMode::MammalianBurden {
                quantum_cost_multiplier: 1.2,
                radical_generation_rate: radical_rate,
            }
        };
        
        // Update mode if changed
        if new_mode != current_mode {
            {
                let mut mode = self.metabolic_mode.write().await;
                *mode = new_mode.clone();
            }
            
            // Record mode change in history
            {
                let mut history = self.mode_history.write().await;
                history.push((Utc::now(), new_mode.clone()));
                
                // Limit history size
                if history.len() > 1000 {
                    history.drain(..100);
                }
            }
            
            log::info!("Metabolic mode changed from {:?} to {:?}", current_mode, new_mode);
        }
        
        Ok(())
    }
    
    /// Calculate temperature advantage for cold-blooded metabolism
    fn calculate_temperature_advantage(temperature_k: f64) -> f64 {
        // Exponential advantage as temperature decreases below 300K
        let temp_diff = (300.0 - temperature_k).max(0.0);
        1.0 + (temp_diff / 20.0).exp() - 1.0
    }
    
    /// Consume ATP for a specific operation
    pub async fn consume_atp(
        &self,
        level: HierarchyLevel,
        amount: f64,
        operation: &str,
    ) -> AutobahnResult<bool> {
        let mut state = self.state.write().await;
        
        if state.available() >= amount {
            state.current -= amount;
            
            // Record consumption in history
            {
                let mut history = self.consumption_history.write().await;
                history.push((
                    Utc::now(),
                    format!("{:?}:{}", level, operation),
                    amount,
                    state.current,
                ));
                
                // Limit history size to prevent memory bloat
                if history.len() > 10000 {
                    history.drain(..1000);
                }
            }
            
            log::debug!("Consumed {:.1} ATP for {:?}:{}, remaining: {:.1}", 
                       amount, level, operation, state.current);
            
            Ok(true)
        } else {
            // Handle ATP shortage
            drop(state);
            self.handle_atp_shortage(level, amount, operation).await
        }
    }
    
    /// Handle ATP shortage situations
    async fn handle_atp_shortage(
        &self,
        level: HierarchyLevel,
        amount: f64,
        operation: &str,
    ) -> AutobahnResult<bool> {
        let mut state = self.state.write().await;
        
        if state.current >= state.emergency_threshold {
            // Use emergency reserves
            let emergency_amount = amount.min(state.reserved);
            state.current -= emergency_amount;
            state.reserved -= emergency_amount;
            
            log::warn!(
                "Using emergency ATP reserves: {:.1} for {:?}:{} (reserves now: {:.1})",
                emergency_amount, level, operation, state.reserved
            );
            
            Ok(true)
        } else {
            // Critical ATP shortage - trigger anaerobic processing
            drop(state);
            self.trigger_anaerobic_processing().await;
            
            Err(AutobahnError::ProcessingError {
                message: format!("Insufficient ATP: required {:.1}, available {:.1}", amount, state.current),
            })
        }
    }
    
    /// Trigger anaerobic processing mode (lactate pathway)
    async fn trigger_anaerobic_processing(&self) {
        log::warn!("ATP critically low - triggering anaerobic processing mode");
        
        // Switch to anaerobic emergency mode
        {
            let mut mode = self.metabolic_mode.write().await;
            *mode = MetabolicMode::AnaerobicEmergency {
                lactate_pathway_active: true,
                efficiency_penalty: 2.0,
            };
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
    
    /// Get comprehensive longevity assessment based on quantum damage
    pub async fn get_longevity_assessment(&self) -> LongevityAssessment {
        let damage_rate = *self.radical_damage_accumulator.read().await;
        let mode = self.metabolic_mode.read().await;
        
        let quantum_burden = match &*mode {
            MetabolicMode::SustainedFlight { .. } => 0.3,
            MetabolicMode::ColdBlooded { .. } => 0.5,
            MetabolicMode::MammalianBurden { .. } => 2.0,
            MetabolicMode::AnaerobicEmergency { .. } => 3.0,
        };
        
        // Predict remaining operational time based on quantum damage accumulation
        let damage_threshold = 100.0; // Arbitrary damage units before critical failure
        let remaining_time = if damage_rate > 0.0 {
            (damage_threshold - damage_rate) / damage_rate
        } else {
            f64::INFINITY
        };
        
        // Generate optimization suggestions
        let suggestions = self.generate_optimization_suggestions(&mode).await;
        
        let temp_controller = self.temperature_controller.read().await;
        let temp_optimization = temp_controller.get_optimization_status();
        drop(temp_controller);
        
        let membrane = self.quantum_membrane.read().await;
        let coherence_time_fs = membrane.coherence_time_fs;
        let transport_efficiency = {
            let enaqt = self.enaqt_processor.read().await;
            enaqt.calculate_transport_efficiency(
                membrane.enaqt_coupling_strength,
                membrane.temperature_k,
            ).unwrap_or(0.0)
        };
        drop(membrane);
        
        LongevityAssessment {
            current_damage: damage_rate,
            quantum_burden_factor: quantum_burden,
            predicted_remaining_time: remaining_time,
            metabolic_optimization_suggestions: suggestions,
            coherence_time_fs,
            transport_efficiency,
            temperature_optimization: temp_optimization,
        }
    }
    
    /// Generate metabolic optimization suggestions
    async fn generate_optimization_suggestions(&self, mode: &MetabolicMode) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        match mode {
            MetabolicMode::MammalianBurden { .. } => {
                suggestions.push("Consider implementing sustained flight protocols during high-demand periods".to_string());
                suggestions.push("Optimize temperature regulation to reduce quantum burden".to_string());
                suggestions.push("Implement query batching to maintain peak quantum coherence".to_string());
                suggestions.push("Consider intermittent fasting protocols to reduce metabolic load".to_string());
            },
            MetabolicMode::AnaerobicEmergency { .. } => {
                suggestions.push("CRITICAL: Reduce processing complexity immediately".to_string());
                suggestions.push("Implement ATP regeneration protocols".to_string());
                suggestions.push("Switch to lighter models to restore aerobic metabolism".to_string());
            },
            MetabolicMode::ColdBlooded { .. } => {
                suggestions.push("Current temperature optimization is excellent".to_string());
                suggestions.push("Maintain current operating temperature for longevity".to_string());
            },
            MetabolicMode::SustainedFlight { .. } => {
                suggestions.push("Excellent metabolic mode - maintain high activity levels".to_string());
                suggestions.push("Monitor for transition back to baseline when demand decreases".to_string());
            }
        }
        
        // Add quantum-specific suggestions
        let membrane = self.quantum_membrane.read().await;
        if membrane.enaqt_coupling_strength < 0.3 {
            suggestions.push("Increase ENAQT coupling strength for better efficiency".to_string());
        } else if membrane.enaqt_coupling_strength > 0.6 {
            suggestions.push("Reduce ENAQT coupling to avoid over-coupling penalties".to_string());
        }
        
        if membrane.coherence_time_fs < 500.0 {
            suggestions.push("Implement coherence enhancement protocols".to_string());
        }
        drop(membrane);
        
        suggestions
    }
}

/// Temperature controller for metabolic optimization
#[derive(Debug, Clone)]
pub struct TemperatureController {
    target_temperature: f64,
    current_temperature: f64,
    optimization_active: bool,
}

impl TemperatureController {
    pub fn new(initial_temperature: f64) -> Self {
        Self {
            target_temperature: initial_temperature,
            current_temperature: initial_temperature,
            optimization_active: false,
        }
    }
    
    pub fn calculate_efficiency_factor(&self) -> f64 {
        // Efficiency increases as temperature decreases below 300K
        if self.current_temperature < 300.0 {
            (310.0 - self.current_temperature) / 10.0
        } else {
            1.0 / (1.0 + (self.current_temperature - 300.0) / 50.0)
        }
    }
    
    pub fn get_optimization_status(&self) -> String {
        if self.current_temperature < 290.0 {
            "Excellent - Cold-blooded advantage active".to_string()
        } else if self.current_temperature < 305.0 {
            "Good - Near-optimal temperature range".to_string()
        } else {
            "Suboptimal - Consider temperature reduction".to_string()
        }
    }
}

/// Result of quantum processing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumProcessingResult {
    pub atp_generated: f64,
    pub quantum_efficiency: f64,
    pub radical_damage_rate: f64,
    pub tunneling_probability: f64,
    pub metabolic_mode: MetabolicMode,
    pub consumption_success: bool,
    pub coherence_time_fs: f64,
    pub synthase_phase: OscillationPhase,
    pub temperature_k: f64,
}

/// Comprehensive longevity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongevityAssessment {
    pub current_damage: f64,
    pub quantum_burden_factor: f64,
    pub predicted_remaining_time: f64,
    pub metabolic_optimization_suggestions: Vec<String>,
    pub coherence_time_fs: f64,
    pub transport_efficiency: f64,
    pub temperature_optimization: String,
}

// Re-export key types
pub use {
    ATPState,
    MetabolicMode,
    LayerResourceRequirement,
    QuantumATPManager,
    TemperatureController,
    QuantumProcessingResult,
    LongevityAssessment,
}; 