//! Quantum-enhanced ATP manager implementing the Membrane Quantum Computation Theorem
//! with metabolic mode switching based on energy demand and environmental conditions.

use crate::atp::{OscillatoryATPManager, ATPState, MetabolicMode, LayerResourceRequirement};
use crate::quantum::{QuantumMembraneState, ENAQTProcessor, QuantumOscillatoryProfile};
use crate::oscillatory::{OscillationPhase, UniversalOscillator};
use crate::biological::BiologicalLayer;
use crate::error::{AutobahnError, AutobahnResult};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Advanced ATP manager with quantum membrane computation
pub struct QuantumATPManager {
    /// Base oscillatory ATP manager
    base_manager: OscillatoryATPManager,
    /// Quantum membrane state
    quantum_membrane: QuantumMembraneState,
    /// ENAQT processor for quantum calculations
    enaqt_processor: ENAQTProcessor,
    /// ATP synthase modeled as quantum oscillator
    atp_synthase_oscillator: UniversalOscillator,
    /// Accumulated radical damage over time
    radical_damage_accumulator: f64,
    /// Current metabolic mode
    metabolic_mode: Arc<RwLock<MetabolicMode>>,
    /// Metabolic mode history for analysis
    mode_history: Arc<RwLock<Vec<(DateTime<Utc>, MetabolicMode)>>>,
    /// Temperature regulation system
    temperature_controller: TemperatureController,
}

impl QuantumATPManager {
    pub fn new(maximum_atp: f64, operating_temperature: f64) -> Self {
        let quantum_membrane = QuantumMembraneState::new(operating_temperature);
        let enaqt_processor = ENAQTProcessor::new(10);
        
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
            base_manager: OscillatoryATPManager::new(maximum_atp),
            quantum_membrane,
            enaqt_processor,
            atp_synthase_oscillator,
            radical_damage_accumulator: 0.0,
            metabolic_mode: Arc::new(RwLock::new(initial_mode)),
            mode_history: Arc::new(RwLock::new(Vec::new())),
            temperature_controller: TemperatureController::new(operating_temperature),
        }
    }
    
    /// Calculate quantum-enhanced ATP cost with full metabolic modeling
    pub async fn calculate_quantum_atp_cost(
        &self,
        layer: BiologicalLayer,
        query_complexity: f64,
        quantum_profile: &QuantumOscillatoryProfile,
    ) -> AutobahnResult<f64> {
        // Base oscillatory cost
        let base_cost = self.base_manager.calculate_oscillatory_atp_cost(
            layer,
            query_complexity,
            &quantum_profile.base_oscillation,
        ).await?;
        
        // Quantum membrane efficiency modulation
        let quantum_efficiency = self.enaqt_processor.calculate_transport_efficiency(
            self.quantum_membrane.enaqt_coupling_strength,
            self.quantum_membrane.temperature_k,
        )?;
        
        // Current metabolic mode modulation
        let mode = self.metabolic_mode.read().await;
        let mode_factor = match &*mode {
            MetabolicMode::SustainedFlight { efficiency_boost, .. } => {
                0.4 * efficiency_boost // Sustained flight provides massive efficiency
            },
            MetabolicMode::ColdBlooded { temperature_advantage, metabolic_reduction } => {
                metabolic_reduction / temperature_advantage // Cold-blooded efficiency
            },
            MetabolicMode::MammalianBurden { quantum_cost_multiplier, .. } => {
                *quantum_cost_multiplier // Mammalian quantum burden
            },
            MetabolicMode::AnaerobicEmergency { efficiency_penalty, .. } => {
                2.0 + efficiency_penalty // Emergency metabolism is expensive
            },
        };
        drop(mode);
        
        // ATP synthase quantum oscillation efficiency
        let synthase_phase = self.atp_synthase_oscillator.calculate_phase();
        let synthase_efficiency = match synthase_phase {
            OscillationPhase::Peak => 0.6,      // Maximum quantum efficiency
            OscillationPhase::Resonance => 0.5, // Optimal ENAQT coupling
            OscillationPhase::Acceleration => 0.8,
            OscillationPhase::Decay => 1.2,
            OscillationPhase::Equilibrium => 1.0,
        };
        
        // Radical damage penalty
        let damage_penalty = 1.0 + (self.radical_damage_accumulator * 0.001);
        
        // Temperature optimization
        let temp_factor = self.temperature_controller.calculate_efficiency_factor();
        
        let quantum_cost = base_cost * mode_factor * synthase_efficiency * damage_penalty 
                          / (quantum_efficiency * temp_factor);
        
        Ok(quantum_cost)
    }
    
    /// Process operation with full quantum membrane modeling
    pub async fn process_with_quantum_membrane(
        &mut self,
        operation: &str,
        energy_demand: f64,
        dt: f64,
    ) -> AutobahnResult<QuantumProcessingResult> {
        // Evolve ATP synthase oscillator
        self.atp_synthase_oscillator.evolve(dt)?;
        
        // Calculate quantum tunneling and radical generation
        let tunneling_prob = self.enaqt_processor.calculate_quantum_tunneling_probability(
            1.0, // 1 eV barrier height (typical for biological systems)
            self.quantum_membrane.membrane_thickness_nm,
            0.3, // 0.3 eV electron energy
        )?;
        
        let radical_rate = self.enaqt_processor.calculate_radical_generation_rate(
            1e-3, // Electron density (M)
            2e-4, // Oxygen concentration (M)
            tunneling_prob,
        )?;
        
        // Accumulate radical damage over time
        self.radical_damage_accumulator += radical_rate * dt;
        
        // Check for metabolic mode transitions
        self.update_metabolic_mode(energy_demand).await?;
        
        // Calculate ATP efficiency with quantum effects
        let quantum_efficiency = self.enaqt_processor.calculate_transport_efficiency(
            self.quantum_membrane.enaqt_coupling_strength,
            self.quantum_membrane.temperature_k,
        )?;
        
        // Calculate actual ATP generated with quantum enhancement
        let base_atp = energy_demand;
        let quantum_enhanced_atp = base_atp * quantum_efficiency;
        
        // Apply metabolic mode efficiency
        let mode = self.metabolic_mode.read().await;
        let final_atp = match &*mode {
            MetabolicMode::SustainedFlight { efficiency_boost, .. } => {
                quantum_enhanced_atp * efficiency_boost
            },
            MetabolicMode::ColdBlooded { temperature_advantage, .. } => {
                quantum_enhanced_atp * temperature_advantage
            },
            MetabolicMode::MammalianBurden { .. } => {
                quantum_enhanced_atp // Standard efficiency
            },
            MetabolicMode::AnaerobicEmergency { efficiency_penalty, .. } => {
                quantum_enhanced_atp / (1.0 + efficiency_penalty)
            },
        };
        drop(mode);
        
        // Consume ATP through base manager
        let consumption_success = self.base_manager.consume_atp(
            BiologicalLayer::Context, // Default layer for quantum processing
            final_atp,
            operation,
        ).await?;
        
        // Update quantum membrane state
        self.update_quantum_membrane_state(energy_demand, dt).await?;
        
        Ok(QuantumProcessingResult {
            atp_generated: final_atp,
            quantum_efficiency,
            radical_damage_rate: radical_rate,
            tunneling_probability: tunneling_prob,
            metabolic_mode: self.metabolic_mode.read().await.clone(),
            consumption_success,
            coherence_time_fs: self.quantum_membrane.coherence_time_fs,
            synthase_phase: self.atp_synthase_oscillator.calculate_phase(),
            temperature_k: self.quantum_membrane.temperature_k,
        })
    }
    
    /// Update metabolic mode based on energy demand and system state
    async fn update_metabolic_mode(&mut self, energy_demand: f64) -> AutobahnResult<()> {
        let current_atp_state = self.base_manager.get_state().await;
        let current_mode = self.metabolic_mode.read().await.clone();
        
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
        } else if energy_demand < 3.0 && self.quantum_membrane.temperature_k < 290.0 {
            // Low demand, low temperature - cold-blooded advantage
            MetabolicMode::ColdBlooded {
                temperature_advantage: Self::calculate_temperature_advantage(self.quantum_membrane.temperature_k),
                metabolic_reduction: 0.6,
            }
        } else {
            // Default mammalian metabolism
            let radical_rate = if self.radical_damage_accumulator > 10.0 { 1.5e-5 } else { 1e-5 };
            MetabolicMode::MammalianBurden {
                quantum_cost_multiplier: 1.2,
                radical_generation_rate: radical_rate,
            }
        };
        
        // Only update if mode changed
        if !std::mem::discriminant(&new_mode).eq(&std::mem::discriminant(&current_mode)) {
            let mut mode_guard = self.metabolic_mode.write().await;
            *mode_guard = new_mode.clone();
            
            // Record mode change
            let mut history = self.mode_history.write().await;
            history.push((Utc::now(), new_mode));
            
            // Keep only last 100 mode changes
            if history.len() > 100 {
                history.remove(0);
            }
        }
        
        Ok(())
    }
    
    /// Update quantum membrane state based on processing
    async fn update_quantum_membrane_state(&mut self, energy_demand: f64, dt: f64) -> AutobahnResult<()> {
        // Membrane thickness changes with energy demand
        let thickness_change = (energy_demand - 5.0) * 0.01 * dt;
        self.quantum_membrane.membrane_thickness_nm += thickness_change;
        
        // Ensure thickness stays within biological bounds
        self.quantum_membrane.membrane_thickness_nm = self.quantum_membrane.membrane_thickness_nm
            .clamp(3.0, 10.0); // 3-10 nm typical range
        
        // ENAQT coupling strength evolves with radical damage
        let coupling_degradation = self.radical_damage_accumulator * 0.001 * dt;
        self.quantum_membrane.enaqt_coupling_strength -= coupling_degradation;
        self.quantum_membrane.enaqt_coupling_strength = self.quantum_membrane.enaqt_coupling_strength
            .max(0.1); // Minimum coupling
        
        // Coherence time decreases with temperature and damage
        let temp_factor = (self.quantum_membrane.temperature_k - 273.0) / 50.0; // Normalized
        let damage_factor = self.radical_damage_accumulator / 100.0;
        
        self.quantum_membrane.coherence_time_fs = (100.0 / (1.0 + temp_factor + damage_factor))
            .max(10.0); // Minimum 10 fs
        
        Ok(())
    }
    
    fn calculate_temperature_advantage(temperature: f64) -> f64 {
        // Cold-blooded advantage peaks around 285K (12°C)
        let optimal_temp = 285.0;
        let temp_diff = (temperature - optimal_temp).abs();
        
        if temp_diff < 5.0 {
            1.8 // Maximum advantage
        } else {
            1.0 + (10.0 - temp_diff).max(0.0) / 20.0
        }
    }
    
    pub async fn get_quantum_state(&self) -> QuantumATPState {
        QuantumATPState {
            base_atp_state: self.base_manager.get_state().await,
            quantum_membrane: self.quantum_membrane.clone(),
            metabolic_mode: self.metabolic_mode.read().await.clone(),
            radical_damage: self.radical_damage_accumulator,
            synthase_phase: self.atp_synthase_oscillator.calculate_phase(),
            temperature_efficiency: self.temperature_controller.calculate_efficiency_factor(),
        }
    }
}

/// Temperature controller for optimal ATP synthesis
#[derive(Debug, Clone)]
pub struct TemperatureController {
    target_temperature: f64,
    current_temperature: f64,
    thermal_mass: f64,
    cooling_rate: f64,
}

impl TemperatureController {
    pub fn new(initial_temperature: f64) -> Self {
        Self {
            target_temperature: 295.0, // Optimal for quantum efficiency
            current_temperature: initial_temperature,
            thermal_mass: 1.0,
            cooling_rate: 0.1,
        }
    }
    
    pub fn calculate_efficiency_factor(&self) -> f64 {
        let temp_diff = (self.current_temperature - self.target_temperature).abs();
        
        if temp_diff < 2.0 {
            1.0 // Perfect efficiency
        } else if temp_diff < 10.0 {
            1.0 - (temp_diff - 2.0) / 20.0 // Linear decrease
        } else {
            0.6 // Minimum efficiency
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

/// Complete quantum ATP state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumATPState {
    pub base_atp_state: ATPState,
    pub quantum_membrane: QuantumMembraneState,
    pub metabolic_mode: MetabolicMode,
    pub radical_damage: f64,
    pub synthase_phase: OscillationPhase,
    pub temperature_efficiency: f64,
} 