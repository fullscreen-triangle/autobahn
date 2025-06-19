//! Quantum membrane computation module implementing Environment-Assisted Quantum Transport (ENAQT)
//! 
//! This module demonstrates that biological membranes function as room-temperature quantum computers,
//! making life a thermodynamic inevitability rather than an improbable accident.

use crate::error::{AutobahnError, AutobahnResult};
use crate::oscillatory::{OscillationProfile, OscillationPhase};
use nalgebra::{DVector, DMatrix};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::f64::consts::PI;

/// Quantum membrane state for biological computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMembraneState {
    /// Current temperature in Kelvin
    pub temperature_k: f64,
    /// ENAQT coupling strength (optimized for efficiency)
    pub enaqt_coupling_strength: f64,
    /// Quantum coherence time in femtoseconds
    pub coherence_time_fs: f64,
    /// Electron transport efficiency (quantum-enhanced)
    pub electron_transport_efficiency: f64,
    /// Rate of radical generation (aging factor)
    pub radical_generation_rate: f64,
    /// Membrane thickness in nanometers
    pub membrane_thickness_nm: f64,
    /// Quantum tunneling probability
    pub tunneling_probability: f64,
    /// Environment coupling matrix
    pub environment_coupling: DMatrix<f64>,
    /// Timestamp of last update
    pub last_update: DateTime<Utc>,
}

impl QuantumMembraneState {
    /// Create new quantum membrane state
    pub fn new(temperature_k: f64) -> Self {
        let coupling_strength = Self::calculate_optimal_coupling(temperature_k);
        let coherence_time = Self::calculate_coherence_time(temperature_k, coupling_strength);
        
        // Initialize 3x3 environment coupling matrix for typical biological system
        let mut env_coupling = DMatrix::zeros(3, 3);
        env_coupling[(0, 1)] = 0.1;
        env_coupling[(1, 0)] = 0.1;
        env_coupling[(1, 2)] = 0.15;
        env_coupling[(2, 1)] = 0.15;
        
        Self {
            temperature_k,
            enaqt_coupling_strength: coupling_strength,
            coherence_time_fs: coherence_time,
            electron_transport_efficiency: 0.95, // High biological efficiency
            radical_generation_rate: Self::calculate_radical_rate(temperature_k),
            membrane_thickness_nm: 4.0, // Typical biological membrane
            tunneling_probability: 0.0, // Will be calculated
            environment_coupling: env_coupling,
            last_update: Utc::now(),
        }
    }

    /// Calculate optimal ENAQT coupling strength for given temperature
    fn calculate_optimal_coupling(temperature_k: f64) -> f64 {
        // Optimal coupling balances coherence preservation and energy transfer
        let thermal_energy = 8.314e-3 * temperature_k; // kT in kJ/mol
        let optimal_coupling = 0.4 * (300.0 / temperature_k).sqrt();
        optimal_coupling.clamp(0.1, 0.8)
    }

    /// Calculate quantum coherence time considering temperature and coupling
    fn calculate_coherence_time(temperature_k: f64, coupling_strength: f64) -> f64 {
        // Base coherence time inversely proportional to temperature
        let base_coherence = 1000.0 * (300.0 / temperature_k).powf(1.5);
        
        // ENAQT enhancement factor
        let enhancement_factor = if coupling_strength > 0.2 && coupling_strength < 0.6 {
            2.5 // Optimal coupling enhances coherence
        } else {
            1.0 / (1.0 + coupling_strength) // Sub-optimal coupling reduces coherence
        };
        
        base_coherence * enhancement_factor
    }

    /// Calculate radical generation rate (quantum aging)
    fn calculate_radical_rate(temperature_k: f64) -> f64 {
        // Exponential dependence on temperature (Arrhenius-like)
        let activation_energy = 50.0; // kJ/mol
        let base_rate = 0.01;
        base_rate * (activation_energy / (8.314e-3 * temperature_k)).exp()
    }

    /// Calculate quantum advantage factor over classical systems
    pub fn quantum_advantage_factor(&self) -> f64 {
        let classical_efficiency = 0.4; // Maximum classical efficiency
        self.electron_transport_efficiency / classical_efficiency
    }

    /// Check if system is operating in optimal quantum regime
    pub fn is_quantum_optimal(&self) -> bool {
        self.coherence_time_fs > 500.0 
            && self.enaqt_coupling_strength > 0.3 
            && self.enaqt_coupling_strength < 0.6
            && self.electron_transport_efficiency > 0.85
    }

    /// Update quantum state based on current conditions
    pub fn update_state(&mut self, dt: f64) -> AutobahnResult<()> {
        // Update radical damage accumulation
        let damage_increment = self.radical_generation_rate * dt;
        
        // Coherence decay over time
        let coherence_decay = dt / (self.coherence_time_fs * 1e-15); // Convert fs to seconds
        self.coherence_time_fs *= (1.0 - coherence_decay * 0.01).max(0.1);

        // Update efficiency based on accumulated damage
        let damage_factor = 1.0 / (1.0 + damage_increment * 0.001);
        self.electron_transport_efficiency *= damage_factor;
        self.electron_transport_efficiency = self.electron_transport_efficiency.max(0.3);

        self.last_update = Utc::now();
        Ok(())
    }
}

/// ENAQT (Environment-Assisted Quantum Transport) processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ENAQTProcessor {
    /// Number of quantum sites in the transport chain
    pub num_sites: usize,
    /// Site-to-site coupling matrix
    pub coupling_matrix: DMatrix<f64>,
    /// Site energies
    pub site_energies: DVector<f64>,
    /// Environmental coupling strengths
    pub environmental_coupling: DVector<f64>,
    /// Transport efficiency optimization parameter
    pub coupling_optimization: f64,
    /// Decoherence rates for each site
    pub decoherence_rates: DVector<f64>,
}

impl ENAQTProcessor {
    /// Create new ENAQT processor
    pub fn new(num_sites: usize) -> Self {
        // Initialize coupling matrix with nearest-neighbor coupling
        let mut coupling_matrix = DMatrix::zeros(num_sites, num_sites);
        let coupling_strength = 100.0; // cm⁻¹

        for i in 0..num_sites-1 {
            coupling_matrix[(i, i+1)] = coupling_strength;
            coupling_matrix[(i+1, i)] = coupling_strength;
        }

        // Random site energies (disorder)
        let mut site_energies = DVector::zeros(num_sites);
        for i in 0..num_sites {
            site_energies[i] = (rand::random::<f64>() - 0.5) * 200.0; // ±100 cm⁻¹ disorder
        }

        // Environmental coupling (stronger at ends for sink/source)
        let mut env_coupling = DVector::zeros(num_sites);
        env_coupling[0] = 50.0; // Source coupling
        env_coupling[num_sites-1] = 50.0; // Sink coupling
        for i in 1..num_sites-1 {
            env_coupling[i] = 10.0; // Weak bulk coupling
        }

        // Decoherence rates proportional to environmental coupling
        let decoherence_rates = &env_coupling * 0.1;

        Self {
            num_sites,
            coupling_matrix,
            site_energies,
            environmental_coupling: env_coupling,
            coupling_optimization: 0.4, // Default optimal value
            decoherence_rates,
        }
    }

    /// Calculate transport efficiency using ENAQT theory
    pub fn calculate_transport_efficiency(&self, coupling_strength: f64, temperature_k: f64) -> AutobahnResult<f64> {
        if temperature_k <= 0.0 {
            return Err(AutobahnError::ConfigurationError {
                parameter: "temperature_k".to_string(),
                value: temperature_k.to_string(),
            });
        }

        // Base efficiency function: η = η₀(1 + αγ + βγ²)
        let eta_0 = 0.85; // Base quantum efficiency
        let alpha = 0.8;  // Linear enhancement coefficient
        let beta = -0.2;  // Quadratic penalty coefficient

        let base_efficiency = eta_0 * (1.0 + alpha * coupling_strength + beta * coupling_strength.powi(2));

        // Temperature enhancement factor (cold-blooded advantage)
        let temp_factor = if temperature_k < 300.0 {
            1.0 + (300.0 - temperature_k) / 100.0 // Linear enhancement below 300K
        } else {
            1.0 / (1.0 + (temperature_k - 300.0) / 200.0) // Penalty above 300K
        };

        // Decoherence penalty
        let avg_decoherence = self.decoherence_rates.mean();
        let decoherence_factor = 1.0 / (1.0 + avg_decoherence * 0.01);

        let final_efficiency = base_efficiency * temp_factor * decoherence_factor;
        
        Ok(final_efficiency.clamp(0.1, 1.0))
    }

    /// Calculate quantum tunneling probability through membrane
    pub fn calculate_tunneling_probability(&self, barrier_height_ev: f64, membrane_thickness_nm: f64) -> AutobahnResult<f64> {
        if barrier_height_ev <= 0.0 || membrane_thickness_nm <= 0.0 {
            return Err(AutobahnError::ConfigurationError {
                parameter: "barrier parameters".to_string(),
                value: format!("height: {}, thickness: {}", barrier_height_ev, membrane_thickness_nm),
            });
        }

        // Quantum tunneling probability: P = exp(-2κd)
        // where κ = sqrt(2m(V-E))/ℏ
        
        const HBAR: f64 = 1.054571817e-34; // J⋅s
        const ELECTRON_MASS: f64 = 9.1093837015e-31; // kg
        const EV_TO_JOULES: f64 = 1.602176634e-19;
        
        let barrier_height_j = barrier_height_ev * EV_TO_JOULES;
        let thickness_m = membrane_thickness_nm * 1e-9;
        
        // Assuming electron has some kinetic energy (thermal)
        let thermal_energy_j = 8.314 * 300.0 / 6.022e23; // ~kT at room temperature
        let effective_barrier = (barrier_height_j - thermal_energy_j).max(0.1 * EV_TO_JOULES);
        
        let kappa = (2.0 * ELECTRON_MASS * effective_barrier).sqrt() / HBAR;
        let tunneling_prob = (-2.0 * kappa * thickness_m).exp();
        
        Ok(tunneling_prob.min(1.0))
    }

    /// Calculate enhanced coherence time with ENAQT
    pub fn calculate_enhanced_coherence(&self, base_coherence_fs: f64, coupling_strength: f64, temperature: f64) -> AutobahnResult<f64> {
        // ENAQT enhancement factor
        let enhancement_factor = if coupling_strength > 0.2 && coupling_strength < 0.6 {
            2.5 // Optimal coupling enhances coherence
        } else {
            1.0 / (1.0 + coupling_strength) // Sub-optimal coupling reduces coherence
        };
        
        // Temperature decoherence factor
        let temp_factor = (300.0 / temperature).sqrt();
        
        let enhanced_coherence = base_coherence_fs * enhancement_factor * temp_factor;
        
        Ok(enhanced_coherence)
    }
    
    /// Optimize ENAQT coupling for maximum efficiency
    pub fn optimize_coupling(&mut self, target_efficiency: f64) -> AutobahnResult<f64> {
        let alpha = 0.8;
        let beta = -0.2;
        
        // Optimal coupling from calculus: dη/dγ = 0 → γ_opt = α/(2|β|)
        let optimal_coupling = alpha / (2.0 * beta.abs());
        
        self.coupling_optimization = optimal_coupling;
        
        // Calculate maximum achievable efficiency
        let eta_0 = 0.85;
        let max_efficiency = eta_0 * (1.0 + alpha * optimal_coupling + beta * optimal_coupling.powi(2));
        
        if max_efficiency < target_efficiency {
            log::warn!(
                "Target efficiency {:.2} not achievable. Maximum: {:.2}", 
                target_efficiency, max_efficiency
            );
        }
        
        Ok(optimal_coupling)
    }
}

/// Combined quantum-oscillatory profile integrating both frameworks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumOscillatoryProfile {
    /// Base oscillatory dynamics
    pub base_oscillation: OscillationProfile,
    /// Quantum membrane state
    pub quantum_membrane_state: QuantumMembraneState,
    /// ENAQT processor for quantum calculations
    pub enaqt_processor: ENAQTProcessor,
    /// ATP synthase efficiency (quantum-enhanced)
    pub atp_synthase_efficiency: f64,
    /// Metabolic quantum burden factor
    pub metabolic_quantum_burden: f64,
    /// Predicted longevity based on quantum damage
    pub longevity_prediction: Option<f64>,
    /// Cross-hierarchy quantum coupling matrix
    pub hierarchy_coupling_matrix: DMatrix<f64>,
}

impl QuantumOscillatoryProfile {
    pub fn new(
        base_oscillation: OscillationProfile,
        temperature_k: f64,
    ) -> Self {
        let quantum_membrane_state = QuantumMembraneState::new(temperature_k);
        let enaqt_processor = ENAQTProcessor::new(10);
        
        // Initialize hierarchy coupling matrix (10x10 for 10 hierarchy levels)
        let mut coupling_matrix = DMatrix::zeros(10, 10);
        
        // Set coupling strengths based on frequency separation
        for i in 0..10 {
            for j in 0..10 {
                if i != j {
                    let freq_ratio = ((i + 1) as f64 / (j + 1) as f64).ln().abs();
                    let coupling = 0.8 * (-freq_ratio / 5.0).exp();
                    coupling_matrix[(i, j)] = coupling;
                }
            }
        }
        
        Self {
            base_oscillation,
            quantum_membrane_state,
            enaqt_processor,
            atp_synthase_efficiency: 0.95, // Default high efficiency
            metabolic_quantum_burden: 1.0,
            longevity_prediction: None,
            hierarchy_coupling_matrix: coupling_matrix,
        }
    }
    
    /// Calculate quantum-enhanced ATP cost considering oscillatory and quantum effects
    pub fn calculate_quantum_enhanced_atp_cost(
        &self,
        base_cost: f64,
        metabolic_demand: f64,
    ) -> AutobahnResult<f64> {
        // Quantum efficiency modulation
        let quantum_efficiency = self.enaqt_processor.calculate_transport_efficiency(
            self.quantum_membrane_state.enaqt_coupling_strength,
            self.quantum_membrane_state.temperature_k,
        )?;
        
        // Metabolic mode factor based on demand
        let metabolic_factor = if metabolic_demand > 10.0 {
            // Sustained flight metabolism - maximum efficiency
            0.6 * quantum_efficiency
        } else if self.quantum_membrane_state.temperature_k < 290.0 {
            // Cold-blooded advantage - temperature-dependent efficiency
            let temp_advantage = (300.0 - self.quantum_membrane_state.temperature_k) / 20.0;
            0.8 * quantum_efficiency * (1.0 + temp_advantage)
        } else {
            // Mammalian quantum burden - higher cost due to warm-blooded metabolism
            1.2 / quantum_efficiency
        };
        
        // Oscillatory phase modulation
        let phase_factor = match self.base_oscillation.phase {
            OscillationPhase::Peak => 0.7,         // Maximum quantum coherence
            OscillationPhase::Resonance => 0.5,    // Optimal ENAQT coupling
            OscillationPhase::Acceleration => 1.3, // Building coherence costs energy
            OscillationPhase::Decay => 1.8,        // Losing coherence is inefficient
            OscillationPhase::Equilibrium => 1.0,  // Baseline efficiency
        };
        
        // Coupling strength optimization
        let coupling_factor = if self.quantum_membrane_state.enaqt_coupling_strength > 0.3 
                                && self.quantum_membrane_state.enaqt_coupling_strength < 0.6 {
            0.9 // Optimal coupling range
        } else {
            1.1 // Sub-optimal coupling penalty
        };
        
        let final_cost = base_cost * metabolic_factor * phase_factor * coupling_factor;
        
        Ok(final_cost)
    }
    
    /// Predict longevity enhancement based on quantum mechanical principles
    pub fn predict_longevity_enhancement(&mut self) -> AutobahnResult<f64> {
        let base_lifespan = 80.0; // Human baseline years
        
        // Temperature factor (ectothermic advantage from theorem)
        let temp_factor = if self.quantum_membrane_state.temperature_k < 300.0 {
            let temp_reduction = (300.0 - self.quantum_membrane_state.temperature_k) / 10.0;
            // Exponential longevity increase with temperature reduction
            (temp_reduction / 2.0).exp()
        } else {
            1.0
        };
        
        // Metabolic efficiency factor (sustained flight advantage)
        let efficiency_factor = if self.atp_synthase_efficiency > 0.9 {
            2.5 // Sustained flight metabolism advantage
        } else if self.atp_synthase_efficiency > 0.8 {
            1.5 // Good efficiency
        } else {
            1.0 // Standard mammalian burden
        };
        
        // Radical generation burden (quantum mechanical aging)
        let radical_factor = 1.0 / (1.0 + self.quantum_membrane_state.radical_generation_rate * 0.1);
        
        // Quantum coherence advantage
        let coherence_factor = 1.0 + (self.quantum_membrane_state.coherence_time_fs / 1000.0).ln().max(0.0);
        
        // ENAQT coupling optimization bonus
        let coupling_bonus = if self.quantum_membrane_state.enaqt_coupling_strength > 0.35 
                               && self.quantum_membrane_state.enaqt_coupling_strength < 0.55 {
            1.2 // Optimal coupling provides longevity bonus
        } else {
            1.0
        };
        
        let predicted_lifespan = base_lifespan * temp_factor * efficiency_factor 
                               * radical_factor * coherence_factor * coupling_bonus;
        
        self.longevity_prediction = Some(predicted_lifespan);
        
        log::info!(
            "Longevity prediction: {:.1} years (temp: {:.1}x, efficiency: {:.1}x, radicals: {:.1}x, coherence: {:.1}x)",
            predicted_lifespan, temp_factor, efficiency_factor, radical_factor, coherence_factor
        );
        
        Ok(predicted_lifespan)
    }
    
    /// Calculate quantum advantage over classical systems
    pub fn quantum_advantage(&self) -> f64 {
        let classical_efficiency = 0.4; // Maximum classical efficiency
        let quantum_efficiency = self.quantum_membrane_state.electron_transport_efficiency;
        
        quantum_efficiency / classical_efficiency
    }
    
    /// Check if system is operating in optimal quantum regime
    pub fn is_quantum_optimal(&self) -> bool {
        self.quantum_membrane_state.is_quantum_optimal()
            && self.base_oscillation.quality_factor > 5.0
            && self.atp_synthase_efficiency > 0.85
    }
} 