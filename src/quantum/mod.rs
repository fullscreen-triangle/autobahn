//! Quantum membrane computation implementation based on the Membrane Quantum Computation Theorem.
//! This module implements Environment-Assisted Quantum Transport (ENAQT) and demonstrates
//! how biological membranes function as room-temperature quantum computers.

use crate::error::{AutobahnError, AutobahnResult};
use crate::oscillatory::{OscillationProfile, OscillationPhase};
use nalgebra::{Complex, DVector, DMatrix};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantum state of a biological membrane system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMembraneState {
    /// Quantum coherence time in femtoseconds (FMO complex: ~660 fs)
    pub coherence_time_fs: f64,
    /// Electron tunneling probability through membrane
    pub tunneling_probability: f64,
    /// ENAQT coupling strength (optimal ~0.4)
    pub enaqt_coupling_strength: f64,
    /// Operating temperature in Kelvin
    pub temperature_k: f64,
    /// Electron transport efficiency (biological: >90%)
    pub electron_transport_efficiency: f64,
    /// Rate of oxygen radical generation (quantum leakage)
    pub radical_generation_rate: f64,
    /// Membrane thickness in nanometers (typical: 3-5 nm)
    pub membrane_thickness_nm: f64,
    /// Complex quantum oscillation amplitudes
    pub quantum_oscillations: Vec<Complex<f64>>,
    /// Membrane potential in millivolts
    pub membrane_potential_mv: f64,
    /// Proton gradient strength
    pub proton_gradient: f64,
}

impl QuantumMembraneState {
    pub fn new(temperature_k: f64) -> Self {
        Self {
            coherence_time_fs: 660.0, // FMO complex experimental value
            tunneling_probability: 0.1,
            enaqt_coupling_strength: 0.4, // Optimal coupling from theorem
            temperature_k,
            electron_transport_efficiency: 0.95,
            radical_generation_rate: 1e-6, // Base quantum leakage rate
            membrane_thickness_nm: 4.0,
            quantum_oscillations: vec![Complex::new(1.0, 0.0); 10],
            membrane_potential_mv: -70.0, // Typical resting potential
            proton_gradient: 1.0,
        }
    }
    
    /// Calculate quantum advantage factor over classical systems
    pub fn quantum_advantage_factor(&self) -> f64 {
        // κ = k_quantum / k_classical = 1 / (1 + exp((E_a - ΔG) / k_B T))
        let activation_energy_ev = 0.5; // Typical activation barrier
        let delta_g_ev = 0.3; // Free energy change
        let kb_t_ev = 8.617e-5 * self.temperature_k; // Boltzmann constant × temperature
        
        let exponent = (activation_energy_ev - delta_g_ev) / kb_t_ev;
        1.0 / (1.0 + exponent.exp())
    }
    
    /// Check if membrane is in optimal quantum regime
    pub fn is_quantum_optimal(&self) -> bool {
        self.coherence_time_fs > 500.0 
            && self.electron_transport_efficiency > 0.9
            && self.enaqt_coupling_strength > 0.3
            && self.enaqt_coupling_strength < 0.6
    }
}

/// Environment-Assisted Quantum Transport processor
/// Implements the core theorem: environmental coupling enhances rather than destroys coherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ENAQTProcessor {
    /// System Hamiltonian (H_system)
    pub system_hamiltonian: DMatrix<Complex<f64>>,
    /// Environment Hamiltonian (H_environment)  
    pub environment_hamiltonian: DMatrix<Complex<f64>>,
    /// Interaction Hamiltonian (H_interaction)
    pub interaction_hamiltonian: DMatrix<Complex<f64>>,
    /// Optimized coupling strength
    pub coupling_optimization: f64,
    /// Coherence enhancement factor from environmental coupling
    pub coherence_enhancement_factor: f64,
    /// Spectral density of environmental modes
    pub environmental_spectral_density: Vec<f64>,
}

impl ENAQTProcessor {
    pub fn new(dimensions: usize) -> Self {
        Self {
            system_hamiltonian: DMatrix::zeros(dimensions, dimensions),
            environment_hamiltonian: DMatrix::zeros(dimensions, dimensions),
            interaction_hamiltonian: DMatrix::zeros(dimensions, dimensions),
            coupling_optimization: 0.0,
            coherence_enhancement_factor: 1.0,
            environmental_spectral_density: vec![0.0; dimensions],
        }
    }
    
    /// Calculate transport efficiency using ENAQT theorem: η = η₀ × (1 + αγ + βγ²)
    pub fn calculate_transport_efficiency(
        &self,
        coupling_strength: f64,
        temperature: f64,
    ) -> AutobahnResult<f64> {
        // ENAQT enhancement coefficients (from biological measurements)
        let alpha = 0.8;  // Linear enhancement coefficient 
        let beta = -0.2;  // Quadratic optimization coefficient
        let eta_0 = 0.85; // Base efficiency without environmental coupling
        
        let gamma = coupling_strength;
        
        // Verify coupling is in valid range
        if gamma < 0.0 || gamma > 1.0 {
            return Err(AutobahnError::PhysicsError { 
                message: format!("Invalid ENAQT coupling: {}", gamma)
            });
        }
        
        // ENAQT efficiency formula
        let efficiency = eta_0 * (1.0 + alpha * gamma + beta * gamma.powi(2));
        
        // Calculate optimal coupling: γ_optimal = α/(2|β|)
        let gamma_optimal = alpha / (2.0 * beta.abs());
        
        // Temperature correction factor (biological systems maintain efficiency at 300K)
        let temperature_factor = if temperature > 0.0 {
            (-0.01 * (temperature - 300.0).abs()).exp()
        } else {
            return Err(AutobahnError::ConfigurationError(
                format!("Invalid temperature: {}", temperature)
            ));
        };
        
        // Store optimization information
        let final_efficiency = efficiency * temperature_factor;
        
        // Biological systems achieve >90% efficiency, artificial systems <40%
        Ok(final_efficiency.min(0.98)) // Cap at 98% (thermodynamic limit)
    }
    
    /// Calculate quantum tunneling probability: P = (16E(V₀-E)/V₀²) × exp(-2κa)
    pub fn calculate_quantum_tunneling_probability(
        &self,
        barrier_height_ev: f64,
        barrier_width_nm: f64,
        electron_energy_ev: f64,
    ) -> AutobahnResult<f64> {
        if barrier_height_ev <= 0.0 || barrier_width_nm <= 0.0 {
            return Err(AutobahnError::PhysicsError { 
                message: format!("Invalid barrier parameters: height={}, width={}", 
                               barrier_height_ev, barrier_width_nm)
            });
        }
        
        let e = electron_energy_ev;
        let v0 = barrier_height_ev;
        let a = barrier_width_nm * 1e-9; // Convert to meters
        
        // Over-barrier transport (classical regime)
        if e >= v0 {
            return Ok(1.0);
        }
        
        // Quantum tunneling calculation
        // κ = √(2m(V₀-E)/ℏ²)
        let m_electron = 9.109e-31; // kg
        let hbar = 1.055e-34; // J⋅s
        let ev_to_joule = 1.602e-19;
        
        let energy_diff = (v0 - e) * ev_to_joule;
        let kappa = ((2.0 * m_electron * energy_diff) / (hbar * hbar)).sqrt();
        
        // Transmission coefficient
        let pre_factor = (16.0 * e * (v0 - e)) / (v0 * v0);
        let exponential_factor = (-2.0 * kappa * a).exp();
        
        let tunneling_prob = pre_factor * exponential_factor;
        
        Ok(tunneling_prob.min(1.0))
    }
    
    /// Calculate oxygen radical generation rate from quantum leakage
    /// d[O₂⁻]/dt = k_leak × [e⁻] × [O₂] × P_quantum
    pub fn calculate_radical_generation_rate(
        &self,
        electron_density: f64,
        oxygen_concentration: f64,
        quantum_leakage_probability: f64,
    ) -> AutobahnResult<f64> {
        if electron_density < 0.0 || oxygen_concentration < 0.0 || quantum_leakage_probability < 0.0 {
            return Err(AutobahnError::PhysicsError { 
                message: format!("Negative values not allowed: e_density={}, O2={}, leakage={}", 
                               electron_density, oxygen_concentration, quantum_leakage_probability)
            });
        }
        
        // Rate constant for electron-oxygen interaction (typical biological value)
        let k_leak = 1e6; // s⁻¹ M⁻¹
        
        let radical_rate = k_leak * electron_density * oxygen_concentration * quantum_leakage_probability;
        
        Ok(radical_rate)
    }
    
    /// Calculate coherence time with environmental enhancement
    pub fn calculate_enhanced_coherence_time(
        &self,
        base_coherence_fs: f64,
        coupling_strength: f64,
        temperature: f64,
    ) -> AutobahnResult<f64> {
        // Environmental enhancement of coherence (counter-intuitive but experimentally verified)
        let enhancement_factor = if coupling_strength > 0.2 && coupling_strength < 0.6 {
            1.0 + 0.5 * coupling_strength // Optimal coupling enhances coherence
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