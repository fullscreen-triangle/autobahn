//! Glycolysis Module - First Stage of Biological Metabolism
//!
//! This module implements the glycolysis pathway, the first stage of cellular respiration
//! that breaks down glucose (information) into pyruvate with ATP production.

use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::traits::EnergyManager;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Glycolysis processor implementing the first metabolic pathway
#[derive(Debug, Clone)]
pub struct GlycolysisProcessor {
    /// Current processing state
    state: GlycolysisState,
    /// Glucose (information) buffer
    glucose_buffer: Vec<GlucoseUnit>,
    /// Pyruvate (processed information) output
    pyruvate_output: Vec<PyruvateUnit>,
    /// ATP yield tracking
    atp_yield: ATPYield,
    /// Processing configuration
    config: GlycolysisConfig,
}

/// State of the glycolysis process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlycolysisState {
    /// Current step in glycolysis pathway
    pub current_step: GlycolysisStep,
    /// Processing progress (0.0 - 1.0)
    pub progress: f64,
    /// Energy investment phase complete
    pub investment_complete: bool,
    /// Energy payoff phase active
    pub payoff_active: bool,
    /// Total processing time
    pub processing_time_ms: u64,
}

/// Steps in the glycolysis pathway
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GlycolysisStep {
    /// Glucose phosphorylation (information preparation)
    GlucosePhosphorylation,
    /// Glucose-6-phosphate isomerization
    Isomerization,
    /// Fructose-6-phosphate phosphorylation
    SecondPhosphorylation,
    /// Fructose-1,6-bisphosphate cleavage
    Cleavage,
    /// Triose phosphate isomerization
    TrioseIsomerization,
    /// Glyceraldehyde-3-phosphate oxidation
    Oxidation,
    /// 1,3-bisphosphoglycerate to 3-phosphoglycerate
    FirstATPGeneration,
    /// 3-phosphoglycerate to 2-phosphoglycerate
    Mutase,
    /// 2-phosphoglycerate to phosphoenolpyruvate
    Enolase,
    /// Phosphoenolpyruvate to pyruvate
    FinalATPGeneration,
    /// Complete
    Complete,
}

/// Glucose unit (information input)
#[derive(Debug, Clone)]
pub struct GlucoseUnit {
    /// Original information content
    pub content: String,
    /// Information complexity
    pub complexity: f64,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
    /// Energy content
    pub energy_content: f64,
}

/// Pyruvate unit (processed information output)
#[derive(Debug, Clone)]
pub struct PyruvateUnit {
    /// Processed content
    pub content: String,
    /// Processing confidence
    pub confidence: f64,
    /// Energy yield
    pub energy_yield: f64,
    /// Ready for Krebs cycle
    pub ready_for_krebs: bool,
    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// ATP yield tracking for glycolysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATPYield {
    /// ATP invested (glucose activation)
    pub atp_invested: f64,
    /// ATP generated (energy payoff)
    pub atp_generated: f64,
    /// Net ATP yield
    pub net_atp: f64,
    /// Efficiency ratio
    pub efficiency: f64,
}

/// Configuration for glycolysis processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlycolysisConfig {
    /// Maximum glucose buffer size
    pub max_glucose_buffer: usize,
    /// ATP investment per glucose unit
    pub atp_investment: f64,
    /// Expected ATP yield per glucose
    pub expected_atp_yield: f64,
    /// Processing timeout per step
    pub step_timeout_ms: u64,
    /// Enable parallel processing
    pub parallel_processing: bool,
}

impl GlycolysisProcessor {
    /// Create new glycolysis processor
    pub fn new() -> Self {
        Self {
            state: GlycolysisState::new(),
            glucose_buffer: Vec::new(),
            pyruvate_output: Vec::new(),
            atp_yield: ATPYield::new(),
            config: GlycolysisConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: GlycolysisConfig) -> Self {
        Self {
            state: GlycolysisState::new(),
            glucose_buffer: Vec::new(),
            pyruvate_output: Vec::new(),
            atp_yield: ATPYield::new(),
            config,
        }
    }

    /// Add glucose (information) to buffer
    pub fn add_glucose(&mut self, content: String, complexity: f64) -> AutobahnResult<()> {
        if self.glucose_buffer.len() >= self.config.max_glucose_buffer {
            return Err(AutobahnError::ProcessingError {
                layer: "glycolysis".to_string(),
                reason: "Glucose buffer full".to_string(),
            });
        }

        let glucose_unit = GlucoseUnit {
            content,
            complexity,
            metadata: HashMap::new(),
            energy_content: complexity * 10.0, // Base energy content
        };

        self.glucose_buffer.push(glucose_unit);
        Ok(())
    }

    /// Process glucose through glycolysis pathway
    pub async fn process_glucose(&mut self, energy_manager: &mut dyn EnergyManager) -> AutobahnResult<Vec<PyruvateUnit>> {
        if self.glucose_buffer.is_empty() {
            return Ok(Vec::new());
        }

        let mut pyruvate_units = Vec::new();

        for glucose in self.glucose_buffer.drain(..) {
            let pyruvate = self.process_single_glucose(glucose, energy_manager).await?;
            pyruvate_units.push(pyruvate);
        }

        self.pyruvate_output.extend(pyruvate_units.clone());
        Ok(pyruvate_units)
    }

    /// Process a single glucose unit through the glycolysis pathway
    async fn process_single_glucose(
        &mut self,
        glucose: GlucoseUnit,
        energy_manager: &mut dyn EnergyManager,
    ) -> AutobahnResult<PyruvateUnit> {
        // Energy investment phase (steps 1-3)
        self.energy_investment_phase(&glucose, energy_manager).await?;

        // Cleavage phase (steps 4-5)
        let triose_phosphates = self.cleavage_phase(&glucose).await?;

        // Energy payoff phase (steps 6-10)
        let pyruvate = self.energy_payoff_phase(triose_phosphates, energy_manager).await?;

        Ok(pyruvate)
    }

    /// Energy investment phase (ATP consumption)
    async fn energy_investment_phase(
        &mut self,
        glucose: &GlucoseUnit,
        energy_manager: &mut dyn EnergyManager,
    ) -> AutobahnResult<()> {
        // Step 1: Glucose phosphorylation (ATP investment)
        self.state.current_step = GlycolysisStep::GlucosePhosphorylation;
        energy_manager.consume_atp(self.config.atp_investment)?;
        self.atp_yield.atp_invested += self.config.atp_investment;

        // Step 2: Isomerization (no ATP cost)
        self.state.current_step = GlycolysisStep::Isomerization;
        
        // Step 3: Second phosphorylation (ATP investment)
        self.state.current_step = GlycolysisStep::SecondPhosphorylation;
        energy_manager.consume_atp(self.config.atp_investment)?;
        self.atp_yield.atp_invested += self.config.atp_investment;

        self.state.investment_complete = true;
        self.state.progress = 0.3;

        Ok(())
    }

    /// Cleavage phase (glucose splitting)
    async fn cleavage_phase(&mut self, glucose: &GlucoseUnit) -> AutobahnResult<Vec<String>> {
        // Step 4: Fructose-1,6-bisphosphate cleavage
        self.state.current_step = GlycolysisStep::Cleavage;
        
        // Split glucose content into two triose phosphates
        let content_parts = self.split_content(&glucose.content);
        
        // Step 5: Triose phosphate isomerization
        self.state.current_step = GlycolysisStep::TrioseIsomerization;
        
        self.state.progress = 0.5;
        Ok(content_parts)
    }

    /// Energy payoff phase (ATP generation)
    async fn energy_payoff_phase(
        &mut self,
        triose_phosphates: Vec<String>,
        energy_manager: &mut dyn EnergyManager,
    ) -> AutobahnResult<PyruvateUnit> {
        self.state.payoff_active = true;
        
        let mut processed_content = String::new();
        let mut total_atp_generated = 0.0;

        for triose in triose_phosphates {
            // Step 6: Oxidation and phosphorylation
            self.state.current_step = GlycolysisStep::Oxidation;
            let oxidized_content = self.oxidize_content(&triose);
            
            // Step 7: First ATP generation
            self.state.current_step = GlycolysisStep::FirstATPGeneration;
            let atp_generated = self.config.expected_atp_yield / 2.0; // Per triose
            total_atp_generated += atp_generated;
            
            // Step 8: Mutase reaction
            self.state.current_step = GlycolysisStep::Mutase;
            
            // Step 9: Enolase reaction
            self.state.current_step = GlycolysisStep::Enolase;
            
            // Step 10: Final ATP generation
            self.state.current_step = GlycolysisStep::FinalATPGeneration;
            total_atp_generated += atp_generated;
            
            processed_content.push_str(&oxidized_content);
        }

        // Update ATP yield
        self.atp_yield.atp_generated = total_atp_generated;
        self.atp_yield.net_atp = self.atp_yield.atp_generated - self.atp_yield.atp_invested;
        self.atp_yield.efficiency = if self.atp_yield.atp_invested > 0.0 {
            self.atp_yield.net_atp / self.atp_yield.atp_invested
        } else {
            0.0
        };

        // Complete glycolysis
        self.state.current_step = GlycolysisStep::Complete;
        self.state.progress = 1.0;

        Ok(PyruvateUnit {
            content: processed_content,
            confidence: 0.8, // High confidence from glycolysis
            energy_yield: self.atp_yield.net_atp,
            ready_for_krebs: true,
            metadata: HashMap::new(),
        })
    }

    /// Split content for cleavage phase
    fn split_content(&self, content: &str) -> Vec<String> {
        let mid_point = content.len() / 2;
        vec![
            content[..mid_point].to_string(),
            content[mid_point..].to_string(),
        ]
    }

    /// Oxidize content for energy extraction
    fn oxidize_content(&self, content: &str) -> String {
        // Simulate oxidation by adding processing markers
        format!("OXIDIZED[{}]", content)
    }

    /// Get current processing state
    pub fn get_state(&self) -> &GlycolysisState {
        &self.state
    }

    /// Get ATP yield information
    pub fn get_atp_yield(&self) -> &ATPYield {
        &self.atp_yield
    }

    /// Get pyruvate output buffer
    pub fn get_pyruvate_output(&self) -> &Vec<PyruvateUnit> {
        &self.pyruvate_output
    }

    /// Reset processor state
    pub fn reset(&mut self) {
        self.state = GlycolysisState::new();
        self.glucose_buffer.clear();
        self.pyruvate_output.clear();
        self.atp_yield = ATPYield::new();
    }
}

impl GlycolysisState {
    fn new() -> Self {
        Self {
            current_step: GlycolysisStep::GlucosePhosphorylation,
            progress: 0.0,
            investment_complete: false,
            payoff_active: false,
            processing_time_ms: 0,
        }
    }
}

impl ATPYield {
    fn new() -> Self {
        Self {
            atp_invested: 0.0,
            atp_generated: 0.0,
            net_atp: 0.0,
            efficiency: 0.0,
        }
    }
}

impl Default for GlycolysisConfig {
    fn default() -> Self {
        Self {
            max_glucose_buffer: 100,
            atp_investment: 1.0,     // 1 ATP per phosphorylation
            expected_atp_yield: 4.0, // 4 ATP generated per glucose
            step_timeout_ms: 1000,   // 1 second per step
            parallel_processing: true,
        }
    }
}

impl Default for GlycolysisProcessor {
    fn default() -> Self {
        Self::new()
    }
} 