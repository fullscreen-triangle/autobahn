//! Krebs Cycle Module - Second Stage of Biological Metabolism
//!
//! This module implements the citric acid cycle (Krebs cycle), the second stage of
//! cellular respiration that processes pyruvate into CO2 with significant ATP production.

use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::traits::EnergyManager;
use crate::v8_pipeline::glycolysis::PyruvateUnit;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Krebs cycle processor implementing the citric acid cycle
#[derive(Debug, Clone)]
pub struct KrebsCycleProcessor {
    /// Current processing state
    state: KrebsState,
    /// Pyruvate input buffer
    pyruvate_buffer: Vec<PyruvateUnit>,
    /// Acetyl-CoA intermediate buffer
    acetyl_coa_buffer: Vec<AcetylCoAUnit>,
    /// Processing output
    cycle_output: Vec<CycleOutput>,
    /// ATP yield tracking
    atp_yield: KrebsATPYield,
    /// Processing configuration
    config: KrebsConfig,
}

/// State of the Krebs cycle process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrebsState {
    /// Current step in the cycle
    pub current_step: KrebsStep,
    /// Processing progress (0.0 - 1.0)
    pub progress: f64,
    /// Number of complete cycles
    pub cycles_completed: u32,
    /// Current cycle turn
    pub current_turn: u32,
    /// Processing time
    pub processing_time_ms: u64,
}

/// Steps in the Krebs cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KrebsStep {
    /// Pyruvate to Acetyl-CoA conversion
    PyruvateDecarboxylation,
    /// Acetyl-CoA + Oxaloacetate → Citrate
    CitrateSynthesis,
    /// Citrate → Isocitrate
    Aconitase,
    /// Isocitrate → α-Ketoglutarate (NADH production)
    IsocitrateDehydrogenase,
    /// α-Ketoglutarate → Succinyl-CoA (NADH production)
    AlphaKetoglutarateDehydrogenase,
    /// Succinyl-CoA → Succinate (GTP/ATP production)
    SuccinylCoASynthetase,
    /// Succinate → Fumarate (FADH2 production)
    SuccinateDehydrogenase,
    /// Fumarate → Malate
    Fumarase,
    /// Malate → Oxaloacetate (NADH production)
    MalateDehydrogenase,
    /// Cycle complete
    CycleComplete,
}

/// Acetyl-CoA unit for Krebs cycle processing
#[derive(Debug, Clone)]
pub struct AcetylCoAUnit {
    /// Processed content from pyruvate
    pub content: String,
    /// Energy potential
    pub energy_potential: f64,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
    /// Ready for cycle entry
    pub ready_for_cycle: bool,
}

/// Output from a complete Krebs cycle
#[derive(Debug, Clone)]
pub struct CycleOutput {
    /// Processed information
    pub processed_content: String,
    /// CO2 equivalent (processed waste)
    pub co2_output: Vec<String>,
    /// NADH produced (high-energy carriers)
    pub nadh_produced: f64,
    /// FADH2 produced (energy carriers)
    pub fadh2_produced: f64,
    /// GTP/ATP directly produced
    pub gtp_produced: f64,
    /// Total energy yield
    pub total_energy_yield: f64,
    /// Processing confidence
    pub confidence: f64,
}

/// ATP yield tracking for Krebs cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrebsATPYield {
    /// Direct ATP/GTP produced
    pub direct_atp: f64,
    /// NADH produced (worth ~2.5 ATP each)
    pub nadh_count: f64,
    /// FADH2 produced (worth ~1.5 ATP each)
    pub fadh2_count: f64,
    /// Total theoretical ATP yield
    pub theoretical_atp: f64,
    /// Actual ATP yield (with efficiency)
    pub actual_atp: f64,
    /// Cycle efficiency
    pub efficiency: f64,
}

/// Configuration for Krebs cycle processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrebsConfig {
    /// Maximum pyruvate buffer size
    pub max_pyruvate_buffer: usize,
    /// NADH to ATP conversion ratio
    pub nadh_to_atp_ratio: f64,
    /// FADH2 to ATP conversion ratio
    pub fadh2_to_atp_ratio: f64,
    /// Processing timeout per step
    pub step_timeout_ms: u64,
    /// Enable parallel cycle processing
    pub parallel_cycles: bool,
    /// Minimum efficiency threshold
    pub min_efficiency: f64,
}

impl KrebsCycleProcessor {
    /// Create new Krebs cycle processor
    pub fn new() -> Self {
        Self {
            state: KrebsState::new(),
            pyruvate_buffer: Vec::new(),
            acetyl_coa_buffer: Vec::new(),
            cycle_output: Vec::new(),
            atp_yield: KrebsATPYield::new(),
            config: KrebsConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: KrebsConfig) -> Self {
        Self {
            state: KrebsState::new(),
            pyruvate_buffer: Vec::new(),
            acetyl_coa_buffer: Vec::new(),
            cycle_output: Vec::new(),
            atp_yield: KrebsATPYield::new(),
            config,
        }
    }

    /// Add pyruvate from glycolysis
    pub fn add_pyruvate(&mut self, pyruvate: PyruvateUnit) -> AutobahnResult<()> {
        if self.pyruvate_buffer.len() >= self.config.max_pyruvate_buffer {
            return Err(AutobahnError::ProcessingError {
                layer: "krebs_cycle".to_string(),
                reason: "Pyruvate buffer full".to_string(),
            });
        }

        if !pyruvate.ready_for_krebs {
            return Err(AutobahnError::ProcessingError {
                layer: "krebs_cycle".to_string(),
                reason: "Pyruvate not ready for Krebs cycle".to_string(),
            });
        }

        self.pyruvate_buffer.push(pyruvate);
        Ok(())
    }

    /// Process pyruvate through Krebs cycle
    pub async fn process_pyruvate(&mut self, energy_manager: &mut dyn EnergyManager) -> AutobahnResult<Vec<CycleOutput>> {
        if self.pyruvate_buffer.is_empty() {
            return Ok(Vec::new());
        }

        let mut outputs = Vec::new();

        // Convert pyruvate to Acetyl-CoA
        self.convert_pyruvate_to_acetyl_coa().await?;

        // Process through Krebs cycles
        for acetyl_coa in self.acetyl_coa_buffer.drain(..) {
            let output = self.run_krebs_cycle(acetyl_coa, energy_manager).await?;
            outputs.push(output);
        }

        self.cycle_output.extend(outputs.clone());
        Ok(outputs)
    }

    /// Convert pyruvate to Acetyl-CoA
    async fn convert_pyruvate_to_acetyl_coa(&mut self) -> AutobahnResult<()> {
        self.state.current_step = KrebsStep::PyruvateDecarboxylation;

        for pyruvate in self.pyruvate_buffer.drain(..) {
            // Pyruvate decarboxylation: Pyruvate → Acetyl-CoA + CO2 + NADH
            let acetyl_coa = AcetylCoAUnit {
                content: format!("ACETYL_COA[{}]", pyruvate.content),
                energy_potential: pyruvate.energy_yield * 1.5, // Increased energy potential
                metadata: pyruvate.metadata,
                ready_for_cycle: true,
            };

            self.acetyl_coa_buffer.push(acetyl_coa);
            
            // Produce NADH from decarboxylation
            self.atp_yield.nadh_count += 1.0;
        }

        Ok(())
    }

    /// Run a complete Krebs cycle
    async fn run_krebs_cycle(
        &mut self,
        acetyl_coa: AcetylCoAUnit,
        energy_manager: &mut dyn EnergyManager,
    ) -> AutobahnResult<CycleOutput> {
        self.state.current_turn += 1;
        let mut co2_output = Vec::new();
        let mut processed_content = acetyl_coa.content.clone();

        // Step 1: Citrate synthesis
        self.state.current_step = KrebsStep::CitrateSynthesis;
        processed_content = format!("CITRATE[{}]", processed_content);
        self.state.progress = 0.1;

        // Step 2: Aconitase (Citrate → Isocitrate)
        self.state.current_step = KrebsStep::Aconitase;
        processed_content = format!("ISOCITRATE[{}]", processed_content);
        self.state.progress = 0.2;

        // Step 3: Isocitrate dehydrogenase (produces NADH + CO2)
        self.state.current_step = KrebsStep::IsocitrateDehydrogenase;
        co2_output.push("CO2_1".to_string());
        self.atp_yield.nadh_count += 1.0;
        processed_content = format!("ALPHA_KETOGLUTARATE[{}]", processed_content);
        self.state.progress = 0.4;

        // Step 4: α-Ketoglutarate dehydrogenase (produces NADH + CO2)
        self.state.current_step = KrebsStep::AlphaKetoglutarateDehydrogenase;
        co2_output.push("CO2_2".to_string());
        self.atp_yield.nadh_count += 1.0;
        processed_content = format!("SUCCINYL_COA[{}]", processed_content);
        self.state.progress = 0.5;

        // Step 5: Succinyl-CoA synthetase (produces GTP/ATP)
        self.state.current_step = KrebsStep::SuccinylCoASynthetase;
        self.atp_yield.direct_atp += 1.0;
        processed_content = format!("SUCCINATE[{}]", processed_content);
        self.state.progress = 0.6;

        // Step 6: Succinate dehydrogenase (produces FADH2)
        self.state.current_step = KrebsStep::SuccinateDehydrogenase;
        self.atp_yield.fadh2_count += 1.0;
        processed_content = format!("FUMARATE[{}]", processed_content);
        self.state.progress = 0.8;

        // Step 7: Fumarase (Fumarate → Malate)
        self.state.current_step = KrebsStep::Fumarase;
        processed_content = format!("MALATE[{}]", processed_content);
        self.state.progress = 0.9;

        // Step 8: Malate dehydrogenase (produces NADH)
        self.state.current_step = KrebsStep::MalateDehydrogenase;
        self.atp_yield.nadh_count += 1.0;
        processed_content = format!("OXALOACETATE[{}]", processed_content);
        self.state.progress = 1.0;

        // Complete cycle
        self.state.current_step = KrebsStep::CycleComplete;
        self.state.cycles_completed += 1;

        // Calculate energy yields for this cycle
        let nadh_produced = 3.0; // Per cycle: 1 from pyruvate decarboxylation + 3 from cycle
        let fadh2_produced = 1.0; // Per cycle
        let gtp_produced = 1.0; // Per cycle

        let total_energy_yield = 
            (nadh_produced * self.config.nadh_to_atp_ratio) +
            (fadh2_produced * self.config.fadh2_to_atp_ratio) +
            gtp_produced;

        Ok(CycleOutput {
            processed_content,
            co2_output,
            nadh_produced,
            fadh2_produced,
            gtp_produced,
            total_energy_yield,
            confidence: 0.9, // High confidence from complete cycle
        })
    }

    /// Calculate total ATP yield
    pub fn calculate_total_atp_yield(&mut self) -> f64 {
        self.atp_yield.theoretical_atp = 
            self.atp_yield.direct_atp +
            (self.atp_yield.nadh_count * self.config.nadh_to_atp_ratio) +
            (self.atp_yield.fadh2_count * self.config.fadh2_to_atp_ratio);

        // Apply efficiency factor
        self.atp_yield.efficiency = if self.state.cycles_completed > 0 {
            0.85 // Typical Krebs cycle efficiency
        } else {
            0.0
        };

        self.atp_yield.actual_atp = self.atp_yield.theoretical_atp * self.atp_yield.efficiency;
        self.atp_yield.actual_atp
    }

    /// Get current processing state
    pub fn get_state(&self) -> &KrebsState {
        &self.state
    }

    /// Get ATP yield information
    pub fn get_atp_yield(&self) -> &KrebsATPYield {
        &self.atp_yield
    }

    /// Get cycle outputs
    pub fn get_cycle_outputs(&self) -> &Vec<CycleOutput> {
        &self.cycle_output
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.state = KrebsState::new();
        self.pyruvate_buffer.clear();
        self.acetyl_coa_buffer.clear();
        self.cycle_output.clear();
        self.atp_yield = KrebsATPYield::new();
    }
}

impl KrebsState {
    fn new() -> Self {
        Self {
            current_step: KrebsStep::PyruvateDecarboxylation,
            progress: 0.0,
            cycles_completed: 0,
            current_turn: 0,
            processing_time_ms: 0,
        }
    }
}

impl KrebsATPYield {
    fn new() -> Self {
        Self {
            direct_atp: 0.0,
            nadh_count: 0.0,
            fadh2_count: 0.0,
            theoretical_atp: 0.0,
            actual_atp: 0.0,
            efficiency: 0.0,
        }
    }
}

impl Default for KrebsConfig {
    fn default() -> Self {
        Self {
            max_pyruvate_buffer: 50,
            nadh_to_atp_ratio: 2.5,    // ~2.5 ATP per NADH
            fadh2_to_atp_ratio: 1.5,   // ~1.5 ATP per FADH2
            step_timeout_ms: 1500,     // 1.5 seconds per step
            parallel_cycles: true,
            min_efficiency: 0.7,       // 70% minimum efficiency
        }
    }
}

impl Default for KrebsCycleProcessor {
    fn default() -> Self {
        Self::new()
    }
} 