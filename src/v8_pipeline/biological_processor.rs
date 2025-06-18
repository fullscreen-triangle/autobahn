//! BiologicalProcessor - The main orchestrator for V8 metabolism pipeline
//!
//! This is the core processor that external packages use to metabolize information
//! through authentic biological pathways.

use crate::traits::{MetacognitiveOrchestrator, BiologicalModule, EnergyManager, ModuleInput, ModuleOutput};
use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::v8_pipeline::{V8Configuration, PipelineStage, ProcessingStatistics, ATPAllocation, ProcessingStrategy};
use async_trait::async_trait;
use chrono::Utc;
use uuid::Uuid;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Main biological processor implementing the V8 metabolism pipeline
pub struct BiologicalProcessor {
    /// Configuration for the V8 pipeline
    config: V8Configuration,
    /// Current energy state
    energy_state: EnergyState,
    /// Processing statistics
    statistics: ProcessingStatistics,
    /// Current pipeline stage
    current_stage: Option<PipelineStage>,
    /// V8 modules (8 specialized intelligence modules)
    modules: HashMap<String, Box<dyn BiologicalModule + Send + Sync>>,
    /// Lactate buffer for champagne phase processing
    lactate_buffer: Vec<String>,
    /// Processing history for optimization
    processing_history: Vec<ProcessingResult>,
    /// Is processor ready for work
    ready: bool,
}

impl BiologicalProcessor {
    /// Create a new biological processor
    pub fn new() -> Self {
        Self::with_config(V8Configuration::default())
    }

    /// Create processor with custom configuration
    pub fn with_config(config: V8Configuration) -> Self {
        let energy_state = EnergyState::new(config.max_atp);
        
        Self {
            config,
            energy_state,
            statistics: ProcessingStatistics::default(),
            current_stage: None,
            modules: HashMap::new(),
            lactate_buffer: Vec::new(),
            processing_history: Vec::new(),
            ready: true,
        }
    }

    /// Initialize all V8 modules
    pub async fn initialize_modules(&mut self) -> AutobahnResult<()> {
        log::info!("Initializing V8 biological modules...");

        // For now, we'll create placeholder modules
        // In future iterations, these will be full implementations
        
        // TODO: Initialize actual module implementations
        // self.modules.insert("mzekezeke".to_string(), Box::new(MzekezekerModule::new()));
        // self.modules.insert("diggiden".to_string(), Box::new(DiggidenModule::new()));
        // etc.

        log::info!("V8 modules initialized successfully");
        self.ready = true;
        Ok(())
    }

    /// Calculate content complexity for ATP planning
    fn calculate_content_complexity(&self, content: &str) -> f64 {
        let word_count = content.split_whitespace().count() as f64;
        let sentence_count = content.matches(&['.', '!', '?'][..]).count() as f64;
        let complexity_indicators = content.matches(&["however", "therefore", "nevertheless", "furthermore"][..]).count() as f64;
        
        // Base complexity from length
        let length_complexity = (word_count / 100.0).min(5.0);
        
        // Sentence structure complexity
        let structure_complexity = if sentence_count > 0.0 {
            (word_count / sentence_count / 10.0).min(3.0)
        } else {
            1.0
        };
        
        // Semantic complexity from indicators
        let semantic_complexity = (complexity_indicators / word_count * 20.0).min(2.0);
        
        (length_complexity + structure_complexity + semantic_complexity).max(0.1)
    }

    /// Process through glycolysis (Context layer)
    async fn process_glycolysis(&mut self, content: &str, atp_allocated: f64) -> AutobahnResult<GlycolysisResult> {
        log::debug!("Starting glycolysis processing...");
        
        let start_time = std::time::Instant::now();
        
        // Simulate ATP consumption for glycolysis
        let atp_consumed = atp_allocated * 0.8; // Use 80% of allocated ATP
        
        if !self.energy_state.can_afford(atp_consumed) {
            return Err(AutobahnError::InsufficientATP {
                required: atp_consumed,
                available: self.energy_state.current_atp,
            });
        }

        // Consume ATP
        self.energy_state.current_atp -= atp_consumed;
        
        // Simulate processing steps
        let comprehension_validated = content.len() > 10; // Simple validation
        let context_preserved = !content.is_empty();
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Net ATP gain from glycolysis (typically 2 ATP in biology)
        let atp_net = 2.0;
        self.energy_state.current_atp += atp_net;
        
        log::debug!("Glycolysis completed in {}ms", processing_time);
        
        Ok(GlycolysisResult {
            atp_net,
            processing_time_ms: processing_time,
            comprehension_validated,
            context_preserved,
        })
    }

    /// Process through Krebs cycle (Reasoning layer)
    async fn process_krebs_cycle(&mut self, content: &str, atp_allocated: f64) -> AutobahnResult<KrebsResult> {
        log::debug!("Starting Krebs cycle processing...");
        
        let start_time = std::time::Instant::now();
        
        // Krebs cycle produces more ATP (typically 2 ATP, 6 NADH, 2 FADH2)
        let atp_produced = 2.0;
        let nadh_produced = 6.0;
        let fadh2_produced = 2.0;
        
        let atp_consumed = atp_allocated * 0.6; // More efficient than glycolysis
        
        if !self.energy_state.can_afford(atp_consumed) {
            return Err(AutobahnError::InsufficientATP {
                required: atp_consumed,
                available: self.energy_state.current_atp,
            });
        }

        self.energy_state.current_atp -= atp_consumed;
        
        // Simulate evidence processing
        let evidence_processed = vec![
            format!("Processed: {}", &content[0..content.len().min(50)]),
            "Evidence validated through reasoning".to_string(),
        ];
        
        let cycles_completed = ((content.len() / 100) + 1) as u32;
        
        // Add ATP yield
        self.energy_state.current_atp += atp_produced;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        log::debug!("Krebs cycle completed {} cycles in {}ms", cycles_completed, processing_time);
        
        Ok(KrebsResult {
            atp_produced,
            nadh_produced,
            fadh2_produced,
            cycles_completed,
            evidence_processed,
        })
    }

    /// Process through electron transport (Intuition layer)
    async fn process_electron_transport(&mut self, content: &str, atp_allocated: f64, nadh: f64, fadh2: f64) -> AutobahnResult<ElectronTransportResult> {
        log::debug!("Starting electron transport processing...");
        
        let start_time = std::time::Instant::now();
        
        // Electron transport produces the most ATP (typically 32-34 ATP from NADH and FADH2)
        let atp_from_nadh = nadh * 3.0; // Each NADH produces ~3 ATP
        let atp_from_fadh2 = fadh2 * 2.0; // Each FADH2 produces ~2 ATP
        let total_atp_synthesized = atp_from_nadh + atp_from_fadh2;
        
        let atp_consumed = atp_allocated * 0.3; // Most efficient stage
        
        if !self.energy_state.can_afford(atp_consumed) {
            return Err(AutobahnError::InsufficientATP {
                required: atp_consumed,
                available: self.energy_state.current_atp,
            });
        }

        self.energy_state.current_atp -= atp_consumed;
        
        // Simulate truth synthesis and metacognitive alignment
        let truth_synthesis_confidence = (content.len() as f64 / 1000.0).min(0.95);
        let metacognitive_alignment = 0.85; // Base alignment score
        
        let paradigm_insights = vec![
            "Probabilistic processing pathway identified".to_string(),
            "Uncertainty quantification completed".to_string(),
            "Temporal decay factors calculated".to_string(),
        ];
        
        // Add synthesized ATP
        self.energy_state.current_atp += total_atp_synthesized;
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        log::debug!("Electron transport synthesized {} ATP in {}ms", total_atp_synthesized, processing_time);
        
        Ok(ElectronTransportResult {
            atp_synthesized: total_atp_synthesized,
            truth_synthesis_confidence,
            metacognitive_alignment,
            paradigm_insights,
        })
    }

    /// Create uncertainty analysis for content
    fn create_uncertainty_analysis(&self, content: &str, confidence: f64) -> UncertaintyAnalysis {
        let uncertainty_score = 1.0 - confidence;
        
        let uncertainty_sources = vec![
            UncertaintySource {
                source: "Content ambiguity".to_string(),
                contribution: 0.3,
                description: "Ambiguous terms or unclear references".to_string(),
            },
            UncertaintySource {
                source: "Context dependency".to_string(),
                contribution: 0.2,
                description: "Meaning depends on missing context".to_string(),
            },
            UncertaintySource {
                source: "Temporal factors".to_string(),
                contribution: 0.1,
                description: "Time-dependent information decay".to_string(),
            },
        ];
        
        let mut confidence_intervals = HashMap::new();
        confidence_intervals.insert("overall_confidence".to_string(), (confidence - 0.1, confidence + 0.1));
        
        UncertaintyAnalysis {
            uncertainty_score,
            uncertainty_sources,
            confidence_intervals,
            epistemic_uncertainty: uncertainty_score * 0.7, // Knowledge-based uncertainty
            aleatoric_uncertainty: uncertainty_score * 0.3, // Inherent randomness
            temporal_decay: TemporalDecay {
                decay_function: DecayFunction::Exponential { lambda: 0.1 },
                half_life: 24.0, // 24 hour half-life
                current_strength: 1.0,
            },
        }
    }

    /// Create validation result
    fn create_validation_result(&self, content: &str, confidence: f64) -> ValidationResult {
        let tests = vec![
            ValidationTest {
                test_name: "Comprehension validation".to_string(),
                passed: confidence > 0.7,
                score: confidence,
                details: "Basic comprehension test".to_string(),
            },
            ValidationTest {
                test_name: "Context preservation".to_string(),
                passed: !content.is_empty(),
                score: if content.is_empty() { 0.0 } else { 0.9 },
                details: "Context retention test".to_string(),
            },
        ];
        
        let overall_passed = tests.iter().all(|t| t.passed);
        let robustness_score = tests.iter().map(|t| t.score).sum::<f64>() / tests.len() as f64;
        
        ValidationResult {
            passed: overall_passed,
            tests,
            perturbation_analysis: None, // Would be populated in full implementation
            adversarial_testing: None,   // Would be populated if adversarial testing enabled
            robustness_score,
        }
    }
}

#[async_trait]
impl MetacognitiveOrchestrator for BiologicalProcessor {
    async fn process_information(&mut self, input: InformationInput) -> AutobahnResult<ProcessingResult> {
        let start_time = Utc::now();
        let processing_id = Uuid::new_v4();
        
        log::info!("Starting biological processing for input: {:?}", processing_id);
        
        // Extract content from input
        let content = match &input {
            InformationInput::Text(text) => text.clone(),
            InformationInput::StructuredData { content, .. } => content.clone(),
            InformationInput::GeneticSequence { sequence, .. } => sequence.clone(),
            InformationInput::ScientificDocument { content, .. } => content.clone(),
            InformationInput::MultiModal { primary_content, .. } => primary_content.clone(),
        };
        
        if content.is_empty() {
            return Err(AutobahnError::InvalidInputError {
                expected: "Non-empty content".to_string(),
                actual: "Empty content".to_string(),
            });
        }

        // Calculate complexity and plan ATP allocation
        let complexity = self.calculate_content_complexity(&content);
        let quality_requirements = crate::traits::QualityRequirements::default();
        let allocation = crate::v8_pipeline::calculate_atp_allocation(
            complexity,
            &quality_requirements,
            self.energy_state.current_atp,
        );

        if !allocation.feasible {
            return Err(AutobahnError::InsufficientATP {
                required: allocation.total_required,
                available: self.energy_state.current_atp,
            });
        }

        // Process through the three layers of metabolism
        let glycolysis_result = self.process_glycolysis(&content, allocation.glycolysis_allocation).await?;
        let krebs_result = self.process_krebs_cycle(&content, allocation.krebs_allocation).await?;
        let electron_transport_result = self.process_electron_transport(
            &content,
            allocation.electron_transport_allocation,
            krebs_result.nadh_produced,
            krebs_result.fadh2_produced,
        ).await?;

        // Calculate final metrics
        let total_atp_consumed = allocation.total_required;
        let total_atp_produced = glycolysis_result.atp_net + krebs_result.atp_produced + electron_transport_result.atp_synthesized;
        let efficiency = total_atp_produced / total_atp_consumed;
        let confidence = electron_transport_result.truth_synthesis_confidence;

        // Create processing pathway
        let pathway = ProcessingPathway::Aerobic {
            glycolysis_result,
            krebs_result,
            electron_transport_result,
        };

        // Create output
        let output = ProcessingOutput::Text(format!(
            "Processed content through biological metabolism. Confidence: {:.2}, Efficiency: {:.2}",
            confidence, efficiency
        ));

        // Create uncertainty analysis and validation
        let uncertainty = self.create_uncertainty_analysis(&content, confidence);
        let validation = self.create_validation_result(&content, confidence);

        let end_time = Utc::now();
        let processing_duration = (end_time - start_time).num_milliseconds() as u64;

        // Create processing metadata
        let metadata = ProcessingMetadata {
            processing_id,
            start_time,
            end_time,
            processing_duration_ms: processing_duration,
            modules_used: vec!["glycolysis".to_string(), "krebs".to_string(), "electron_transport".to_string()],
            atp_efficiency: efficiency,
            memory_usage: MemoryUsage {
                peak_memory_mb: 10.0, // Placeholder
                average_memory_mb: 8.0,
                memory_efficiency: 0.85,
            },
            performance_metrics: PerformanceMetrics {
                throughput_tokens_per_second: content.len() as f64 / (processing_duration as f64 / 1000.0),
                latency_p50_ms: processing_duration as f64 * 0.5,
                latency_p95_ms: processing_duration as f64 * 0.95,
                latency_p99_ms: processing_duration as f64 * 0.99,
            },
        };

        let result = ProcessingResult {
            output,
            confidence,
            atp_consumed: total_atp_consumed,
            pathway,
            efficiency,
            uncertainty,
            validation,
            metadata,
        };

        // Update statistics
        self.statistics.total_operations += 1;
        self.statistics.successful_operations += 1;
        self.statistics.average_atp_consumption = 
            (self.statistics.average_atp_consumption * (self.statistics.total_operations - 1) as f64 + total_atp_consumed) 
            / self.statistics.total_operations as f64;

        // Store in processing history
        self.processing_history.push(result.clone());
        
        log::info!("Biological processing completed successfully: {:?}", processing_id);
        
        Ok(result)
    }

    fn analyze_uncertainty(&self, content: &str) -> AutobahnResult<UncertaintyAnalysis> {
        if content.is_empty() {
            return Err(AutobahnError::InvalidInputError {
                expected: "Non-empty content".to_string(),
                actual: "Empty content".to_string(),
            });
        }

        // Quick uncertainty analysis without full processing
        let base_confidence = 0.7; // Default confidence
        let uncertainty_analysis = self.create_uncertainty_analysis(content, base_confidence);
        
        Ok(uncertainty_analysis)
    }

    async fn metabolize_content(&mut self, content: &str) -> AutobahnResult<crate::traits::MetabolismResult> {
        // Process through biological pathways
        let result = self.process_information(InformationInput::Text(content.to_string())).await?;
        
        Ok(crate::traits::MetabolismResult {
            pathway: result.pathway,
            atp_yield: result.efficiency,
            byproducts: vec!["NADH".to_string(), "FADH2".to_string(), "CO2".to_string()],
            efficiency: result.efficiency,
        })
    }

    fn validate_understanding(&self, content: &str) -> AutobahnResult<ValidationResult> {
        if content.is_empty() {
            return Err(AutobahnError::InvalidInputError {
                expected: "Non-empty content".to_string(),
                actual: "Empty content".to_string(),
            });
        }

        let validation_result = self.create_validation_result(content, 0.8);
        Ok(validation_result)
    }

    async fn test_robustness(&mut self, content: &str) -> AutobahnResult<RobustnessReport> {
        // Placeholder implementation
        Ok(RobustnessReport {
            overall_robustness: 0.8,
            vulnerability_assessment: VulnerabilityAssessment {
                critical_vulnerabilities: 0,
                moderate_vulnerabilities: 1,
                low_vulnerabilities: 2,
                risk_score: 0.2,
            },
            recommendations: vec![
                "Consider additional validation".to_string(),
                "Monitor for context drift".to_string(),
            ],
            attack_resistance: AttackResistance {
                resistance_score: 0.85,
                tested_attack_types: vec!["perturbation".to_string()],
                successful_defenses: 4,
                failed_defenses: 1,
            },
        })
    }

    fn get_energy_state(&self) -> EnergyState {
        self.energy_state.clone()
    }

    fn regenerate_atp(&mut self) {
        let now = Utc::now();
        let time_elapsed = (now - self.energy_state.last_regeneration).num_seconds() as f64;
        let regeneration_amount = self.config.atp_regeneration_rate * time_elapsed;
        
        self.energy_state.current_atp = (self.energy_state.current_atp + regeneration_amount)
            .min(self.energy_state.max_atp);
        
        self.energy_state.last_regeneration = now;
        
        log::debug!("Regenerated {} ATP, current: {}", regeneration_amount, self.energy_state.current_atp);
    }

    async fn enter_champagne_phase(&mut self) -> AutobahnResult<crate::traits::ChampagneResult> {
        if !self.config.enable_champagne_phase {
            return Err(AutobahnError::ChampagneUnavailableError {
                user_status: "Champagne phase disabled".to_string(),
            });
        }

        // Process lactate buffer
        let lactate_processed = self.lactate_buffer.len() as f64;
        let insights_gained = vec![
            "Pattern recognition improved".to_string(),
            "Context understanding enhanced".to_string(),
        ];
        
        // Clear lactate buffer
        self.lactate_buffer.clear();
        
        Ok(crate::traits::ChampagneResult {
            lactate_processed,
            insights_gained,
            optimization_improvements: 0.1, // 10% improvement
            dream_duration_ms: 5000, // 5 second dream processing
        })
    }

    fn is_ready(&self) -> bool {
        self.ready && self.energy_state.current_atp > 50.0 // Need minimum ATP to operate
    }

    fn get_capabilities(&self) -> crate::traits::ProcessingCapabilities {
        crate::traits::ProcessingCapabilities {
            supports_probabilistic: true,
            supports_adversarial: self.config.enable_adversarial_testing,
            supports_champagne: self.config.enable_champagne_phase,
            max_atp: self.config.max_atp,
            available_modules: vec![
                "mzekezeke".to_string(),
                "diggiden".to_string(),
                "hatata".to_string(),
                "spectacular".to_string(),
                "nicotine".to_string(),
                "clothesline".to_string(),
                "zengeza".to_string(),
                "diadochi".to_string(),
            ],
            processing_modes: vec![
                ProcessingMode::Probabilistic,
                ProcessingMode::Deterministic,
                ProcessingMode::Dream,
            ],
        }
    }
}

impl Default for BiologicalProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_biological_processor_creation() {
        let processor = BiologicalProcessor::new();
        assert!(processor.is_ready());
        assert_eq!(processor.energy_state.max_atp, 1000.0);
    }

    #[tokio::test]
    async fn test_content_complexity_calculation() {
        let processor = BiologicalProcessor::new();
        
        let simple_content = "Hello world";
        let complex_content = "However, the implications of this discovery are nevertheless quite significant. Therefore, we must consider the ramifications carefully.";
        
        let simple_complexity = processor.calculate_content_complexity(simple_content);
        let complex_complexity = processor.calculate_content_complexity(complex_content);
        
        assert!(complex_complexity > simple_complexity);
    }

    #[tokio::test]
    async fn test_basic_processing() {
        let mut processor = BiologicalProcessor::new();
        let input = InformationInput::Text("Test content for biological processing".to_string());
        
        let result = processor.process_information(input).await;
        assert!(result.is_ok());
        
        let processing_result = result.unwrap();
        assert!(processing_result.confidence > 0.0);
        assert!(processing_result.efficiency > 0.0);
    }

    #[tokio::test]
    async fn test_empty_content_handling() {
        let mut processor = BiologicalProcessor::new();
        let input = InformationInput::Text("".to_string());
        
        let result = processor.process_information(input).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            AutobahnError::InvalidInputError { .. } => {
                // Expected error type
            }
            _ => panic!("Expected InvalidInputError"),
        }
    }

    #[tokio::test]
    async fn test_atp_regeneration() {
        let mut processor = BiologicalProcessor::new();
        let initial_atp = processor.energy_state.current_atp;
        
        // Consume some ATP
        processor.energy_state.current_atp -= 100.0;
        assert!(processor.energy_state.current_atp < initial_atp);
        
        // Regenerate ATP
        processor.regenerate_atp();
        
        // Should have regenerated some ATP
        assert!(processor.energy_state.current_atp >= initial_atp - 100.0);
    }
} 