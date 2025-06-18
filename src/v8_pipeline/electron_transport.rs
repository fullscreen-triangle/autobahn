//! Electron Transport Chain Module - Final Stage of Biological Metabolism
//!
//! This module implements the electron transport chain and oxidative phosphorylation,
//! the final stage of cellular respiration that produces the majority of ATP.

use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::traits::EnergyManager;
use crate::v8_pipeline::krebs_cycle::CycleOutput;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Electron transport chain processor
#[derive(Debug, Clone)]
pub struct ElectronTransportProcessor {
    /// Current processing state
    state: ElectronTransportState,
    /// NADH input buffer
    nadh_buffer: Vec<NADHCarrier>,
    /// FADH2 input buffer
    fadh2_buffer: Vec<FADH2Carrier>,
    /// ATP synthase complex
    atp_synthase: ATPSynthaseComplex,
    /// Proton gradient
    proton_gradient: ProtonGradient,
    /// Final ATP output
    atp_output: Vec<ATPMolecule>,
    /// Processing configuration
    config: ElectronTransportConfig,
}

/// State of the electron transport chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronTransportState {
    /// Current complex being processed
    pub current_complex: ElectronTransportComplex,
    /// Processing progress (0.0 - 1.0)
    pub progress: f64,
    /// Proton pumping active
    pub proton_pumping_active: bool,
    /// ATP synthesis active
    pub atp_synthesis_active: bool,
    /// Total electrons processed
    pub electrons_processed: u32,
    /// Processing time
    pub processing_time_ms: u64,
}

/// Electron transport complexes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElectronTransportComplex {
    /// Complex I (NADH dehydrogenase)
    ComplexI,
    /// Complex II (Succinate dehydrogenase)
    ComplexII,
    /// Complex III (Cytochrome bc1)
    ComplexIII,
    /// Complex IV (Cytochrome c oxidase)
    ComplexIV,
    /// ATP Synthase (Complex V)
    ATPSynthase,
    /// Complete
    Complete,
}

/// NADH carrier molecule
#[derive(Debug, Clone)]
pub struct NADHCarrier {
    /// Energy content
    pub energy_content: f64,
    /// Electron pair
    pub electron_pair: ElectronPair,
    /// Source information
    pub source_content: String,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// FADH2 carrier molecule
#[derive(Debug, Clone)]
pub struct FADH2Carrier {
    /// Energy content
    pub energy_content: f64,
    /// Electron pair
    pub electron_pair: ElectronPair,
    /// Source information
    pub source_content: String,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Electron pair for transport
#[derive(Debug, Clone)]
pub struct ElectronPair {
    /// High energy state
    pub high_energy: bool,
    /// Reduction potential
    pub reduction_potential: f64,
    /// Transport efficiency
    pub transport_efficiency: f64,
}

/// ATP Synthase complex for ATP production
#[derive(Debug, Clone)]
pub struct ATPSynthaseComplex {
    /// Rotor position
    pub rotor_position: f64,
    /// Protons per ATP
    pub protons_per_atp: f64,
    /// Synthesis efficiency
    pub synthesis_efficiency: f64,
    /// Active sites
    pub active_sites: u32,
    /// Current synthesis rate
    pub synthesis_rate: f64,
}

/// Proton gradient across membrane
#[derive(Debug, Clone)]
pub struct ProtonGradient {
    /// Proton concentration difference
    pub concentration_gradient: f64,
    /// Electrical potential difference
    pub electrical_gradient: f64,
    /// Total proton-motive force
    pub proton_motive_force: f64,
    /// Gradient stability
    pub stability: f64,
}

/// ATP molecule output
#[derive(Debug, Clone)]
pub struct ATPMolecule {
    /// Energy content
    pub energy_content: f64,
    /// Synthesis quality
    pub synthesis_quality: f64,
    /// Source pathway
    pub source_pathway: String,
    /// Ready for use
    pub ready_for_use: bool,
}

/// Configuration for electron transport chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronTransportConfig {
    /// Maximum NADH buffer size
    pub max_nadh_buffer: usize,
    /// Maximum FADH2 buffer size
    pub max_fadh2_buffer: usize,
    /// Proton pumping efficiency
    pub proton_pumping_efficiency: f64,
    /// ATP synthesis efficiency
    pub atp_synthesis_efficiency: f64,
    /// Oxygen availability (affects final step)
    pub oxygen_availability: f64,
    /// Processing timeout per complex
    pub complex_timeout_ms: u64,
    /// Enable parallel processing
    pub parallel_processing: bool,
}

impl ElectronTransportProcessor {
    /// Create new electron transport processor
    pub fn new() -> Self {
        Self {
            state: ElectronTransportState::new(),
            nadh_buffer: Vec::new(),
            fadh2_buffer: Vec::new(),
            atp_synthase: ATPSynthaseComplex::new(),
            proton_gradient: ProtonGradient::new(),
            atp_output: Vec::new(),
            config: ElectronTransportConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ElectronTransportConfig) -> Self {
        Self {
            state: ElectronTransportState::new(),
            nadh_buffer: Vec::new(),
            fadh2_buffer: Vec::new(),
            atp_synthase: ATPSynthaseComplex::new(),
            proton_gradient: ProtonGradient::new(),
            atp_output: Vec::new(),
            config,
        }
    }

    /// Add NADH from Krebs cycle
    pub fn add_nadh(&mut self, cycle_output: &CycleOutput) -> AutobahnResult<()> {
        if self.nadh_buffer.len() >= self.config.max_nadh_buffer {
            return Err(AutobahnError::ProcessingError {
                layer: "electron_transport".to_string(),
                reason: "NADH buffer full".to_string(),
            });
        }

        for _ in 0..(cycle_output.nadh_produced as u32) {
            let nadh = NADHCarrier {
                energy_content: 52.0, // ~52 kJ/mol for NADH
                electron_pair: ElectronPair {
                    high_energy: true,
                    reduction_potential: -0.32, // Standard reduction potential
                    transport_efficiency: 0.9,
                },
                source_content: cycle_output.processed_content.clone(),
                metadata: HashMap::new(),
            };
            self.nadh_buffer.push(nadh);
        }

        Ok(())
    }

    /// Add FADH2 from Krebs cycle
    pub fn add_fadh2(&mut self, cycle_output: &CycleOutput) -> AutobahnResult<()> {
        if self.fadh2_buffer.len() >= self.config.max_fadh2_buffer {
            return Err(AutobahnError::ProcessingError {
                layer: "electron_transport".to_string(),
                reason: "FADH2 buffer full".to_string(),
            });
        }

        for _ in 0..(cycle_output.fadh2_produced as u32) {
            let fadh2 = FADH2Carrier {
                energy_content: 38.0, // ~38 kJ/mol for FADH2
                electron_pair: ElectronPair {
                    high_energy: true,
                    reduction_potential: -0.22, // Standard reduction potential
                    transport_efficiency: 0.85,
                },
                source_content: cycle_output.processed_content.clone(),
                metadata: HashMap::new(),
            };
            self.fadh2_buffer.push(fadh2);
        }

        Ok(())
    }

    /// Process electron transport and ATP synthesis
    pub async fn process_electron_transport(&mut self, energy_manager: &mut dyn EnergyManager) -> AutobahnResult<Vec<ATPMolecule>> {
        if self.nadh_buffer.is_empty() && self.fadh2_buffer.is_empty() {
            return Ok(Vec::new());
        }

        // Process NADH through Complex I
        self.process_complex_i().await?;

        // Process FADH2 through Complex II
        self.process_complex_ii().await?;

        // Process through Complex III
        self.process_complex_iii().await?;

        // Process through Complex IV
        self.process_complex_iv().await?;

        // ATP synthesis through Complex V (ATP Synthase)
        let atp_molecules = self.process_atp_synthase().await?;

        self.atp_output.extend(atp_molecules.clone());
        Ok(atp_molecules)
    }

    /// Process Complex I (NADH dehydrogenase)
    async fn process_complex_i(&mut self) -> AutobahnResult<()> {
        self.state.current_complex = ElectronTransportComplex::ComplexI;
        self.state.progress = 0.1;

        for nadh in self.nadh_buffer.drain(..) {
            // NADH → NAD+ + H+ + 2e-
            // Pump 4 protons across membrane
            self.pump_protons(4.0 * self.config.proton_pumping_efficiency);
            self.state.electrons_processed += 2;
        }

        self.state.progress = 0.2;
        Ok(())
    }

    /// Process Complex II (Succinate dehydrogenase)
    async fn process_complex_ii(&mut self) -> AutobahnResult<()> {
        self.state.current_complex = ElectronTransportComplex::ComplexII;
        self.state.progress = 0.3;

        for fadh2 in self.fadh2_buffer.drain(..) {
            // FADH2 → FAD + 2H+ + 2e-
            // No proton pumping in Complex II
            self.state.electrons_processed += 2;
        }

        self.state.progress = 0.4;
        Ok(())
    }

    /// Process Complex III (Cytochrome bc1)
    async fn process_complex_iii(&mut self) -> AutobahnResult<()> {
        self.state.current_complex = ElectronTransportComplex::ComplexIII;
        self.state.progress = 0.5;

        // Process electrons through cytochrome bc1 complex
        // Pump 2 protons per electron pair
        let electron_pairs = self.state.electrons_processed / 2;
        self.pump_protons((electron_pairs as f64) * 2.0 * self.config.proton_pumping_efficiency);

        self.state.progress = 0.7;
        Ok(())
    }

    /// Process Complex IV (Cytochrome c oxidase)
    async fn process_complex_iv(&mut self) -> AutobahnResult<()> {
        self.state.current_complex = ElectronTransportComplex::ComplexIV;
        self.state.progress = 0.8;

        // Check oxygen availability for final electron acceptor
        if self.config.oxygen_availability < 0.1 {
            return Err(AutobahnError::ProcessingError {
                layer: "electron_transport".to_string(),
                reason: "Insufficient oxygen for electron transport".to_string(),
            });
        }

        // Process electrons through cytochrome c oxidase
        // 4e- + 4H+ + O2 → 2H2O
        // Pump 2 protons per electron pair
        let electron_pairs = self.state.electrons_processed / 2;
        self.pump_protons((electron_pairs as f64) * 2.0 * self.config.proton_pumping_efficiency);

        self.state.progress = 0.9;
        Ok(())
    }

    /// Process ATP Synthase (Complex V)
    async fn process_atp_synthase(&mut self) -> AutobahnResult<Vec<ATPMolecule>> {
        self.state.current_complex = ElectronTransportComplex::ATPSynthase;
        self.state.atp_synthesis_active = true;

        let mut atp_molecules = Vec::new();

        // Calculate ATP yield from proton gradient
        let available_protons = self.proton_gradient.concentration_gradient;
        let atp_possible = (available_protons / self.atp_synthase.protons_per_atp).floor();

        for i in 0..(atp_possible as u32) {
            // Consume protons for ATP synthesis
            self.consume_protons(self.atp_synthase.protons_per_atp);

            // Synthesize ATP
            let atp = ATPMolecule {
                energy_content: 30.5, // ~30.5 kJ/mol for ATP hydrolysis
                synthesis_quality: self.atp_synthase.synthesis_efficiency,
                source_pathway: "electron_transport".to_string(),
                ready_for_use: true,
            };

            atp_molecules.push(atp);

            // Rotate ATP synthase
            self.atp_synthase.rotor_position += 120.0; // 120° per ATP
            if self.atp_synthase.rotor_position >= 360.0 {
                self.atp_synthase.rotor_position -= 360.0;
            }
        }

        self.state.current_complex = ElectronTransportComplex::Complete;
        self.state.progress = 1.0;

        Ok(atp_molecules)
    }

    /// Pump protons across membrane
    fn pump_protons(&mut self, proton_count: f64) {
        self.state.proton_pumping_active = true;
        self.proton_gradient.concentration_gradient += proton_count;
        self.proton_gradient.electrical_gradient += proton_count * 0.1; // Electrical component
        self.update_proton_motive_force();
    }

    /// Consume protons for ATP synthesis
    fn consume_protons(&mut self, proton_count: f64) {
        self.proton_gradient.concentration_gradient -= proton_count;
        self.proton_gradient.electrical_gradient -= proton_count * 0.1;
        self.update_proton_motive_force();
    }

    /// Update proton-motive force
    fn update_proton_motive_force(&mut self) {
        self.proton_gradient.proton_motive_force = 
            self.proton_gradient.concentration_gradient * 0.059 + // Chemical gradient
            self.proton_gradient.electrical_gradient; // Electrical gradient
        
        // Update stability
        self.proton_gradient.stability = if self.proton_gradient.proton_motive_force > 0.0 {
            (self.proton_gradient.proton_motive_force / 200.0).min(1.0)
        } else {
            0.0
        };
    }

    /// Calculate theoretical ATP yield
    pub fn calculate_theoretical_atp_yield(&self) -> f64 {
        let nadh_atp = (self.nadh_buffer.len() as f64) * 2.5; // ~2.5 ATP per NADH
        let fadh2_atp = (self.fadh2_buffer.len() as f64) * 1.5; // ~1.5 ATP per FADH2
        nadh_atp + fadh2_atp
    }

    /// Get current processing state
    pub fn get_state(&self) -> &ElectronTransportState {
        &self.state
    }

    /// Get proton gradient information
    pub fn get_proton_gradient(&self) -> &ProtonGradient {
        &self.proton_gradient
    }

    /// Get ATP output
    pub fn get_atp_output(&self) -> &Vec<ATPMolecule> {
        &self.atp_output
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.state = ElectronTransportState::new();
        self.nadh_buffer.clear();
        self.fadh2_buffer.clear();
        self.atp_synthase = ATPSynthaseComplex::new();
        self.proton_gradient = ProtonGradient::new();
        self.atp_output.clear();
    }
}

impl ElectronTransportState {
    fn new() -> Self {
        Self {
            current_complex: ElectronTransportComplex::ComplexI,
            progress: 0.0,
            proton_pumping_active: false,
            atp_synthesis_active: false,
            electrons_processed: 0,
            processing_time_ms: 0,
        }
    }
}

impl ATPSynthaseComplex {
    fn new() -> Self {
        Self {
            rotor_position: 0.0,
            protons_per_atp: 3.0, // Approximately 3 protons per ATP
            synthesis_efficiency: 0.9,
            active_sites: 3,
            synthesis_rate: 100.0, // ATP per second
        }
    }
}

impl ProtonGradient {
    fn new() -> Self {
        Self {
            concentration_gradient: 0.0,
            electrical_gradient: 0.0,
            proton_motive_force: 0.0,
            stability: 0.0,
        }
    }
}

impl Default for ElectronTransportConfig {
    fn default() -> Self {
        Self {
            max_nadh_buffer: 100,
            max_fadh2_buffer: 50,
            proton_pumping_efficiency: 0.9,
            atp_synthesis_efficiency: 0.85,
            oxygen_availability: 1.0, // Full oxygen availability
            complex_timeout_ms: 2000,  // 2 seconds per complex
            parallel_processing: true,
        }
    }
}

impl Default for ElectronTransportProcessor {
    fn default() -> Self {
        Self::new()
    }
} 