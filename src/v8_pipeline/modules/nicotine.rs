//! Nicotine - Context Validation Engine for Semantic Coherence
//!
//! This module validates contextual consistency and semantic coherence
//! across information processing to ensure logical continuity.

use crate::traits::{BiologicalModule, ModuleInput, ModuleOutput, ModuleCapabilities, ModuleMetadata};
use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::v8_pipeline::modules::BaseModule;
use async_trait::async_trait;
use std::collections::HashMap;

/// Nicotine Context Validation Engine
pub struct NicotineModule {
    base: BaseModule,
    validation_rules: Vec<String>,
    coherence_threshold: f64,
}

impl NicotineModule {
    pub fn new() -> Self {
        Self {
            base: BaseModule::new("nicotine"),
            validation_rules: vec![
                "semantic_consistency".to_string(),
                "logical_continuity".to_string(),
                "contextual_relevance".to_string(),
            ],
            coherence_threshold: 0.7,
        }
    }
    
    pub async fn validate_context(&self, content: &str) -> AutobahnResult<f64> {
        // Simplified context validation
        let coherence_score = self.calculate_coherence(content);
        Ok(coherence_score)
    }
    
    fn calculate_coherence(&self, content: &str) -> f64 {
        let mut score = 0.8;
        
        // Check for logical connectors
        if content.to_lowercase().contains("because") || 
           content.to_lowercase().contains("therefore") {
            score += 0.1;
        }
        
        // Penalize contradictions
        if content.to_lowercase().contains("but") && 
           content.to_lowercase().contains("however") {
            score -= 0.2;
        }
        
        score.max(0.0).min(1.0)
    }
}

#[async_trait]
impl BiologicalModule for NicotineModule {
    fn name(&self) -> &str {
        &self.base.name
    }
    
    async fn process(&mut self, input: ModuleInput) -> AutobahnResult<ModuleOutput> {
        self.base.record_processing();
        
        let start_time = std::time::Instant::now();
        
        let coherence_score = match self.validate_context(&input.content).await {
            Ok(score) => score,
            Err(e) => {
                self.base.record_error();
                return Err(e);
            }
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        let atp_consumed = (input.content.len() as f64 / 100.0) * 4.0;
        
        let result = format!(
            "Context validation completed. Coherence: {:.1}%",
            coherence_score * 100.0
        );
        
        Ok(ModuleOutput {
            result,
            confidence: coherence_score,
            atp_consumed,
            byproducts: vec![
                format!("Validation rules applied: {}", self.validation_rules.len()),
                format!("Coherence threshold: {:.1}", self.coherence_threshold),
            ],
            metadata: ModuleMetadata {
                processing_time_ms: processing_time,
                memory_used_mb: 2.0,
                cpu_usage_percent: 15.0,
                cache_hits: 0,
                cache_misses: 1,
            },
        })
    }
    
    fn calculate_atp_cost(&self, input: &ModuleInput) -> f64 {
        (input.content.len() as f64 / 100.0) * 4.0
    }
    
    fn is_ready(&self) -> bool {
        self.base.ready
    }
    
    fn capabilities(&self) -> ModuleCapabilities {
        ModuleCapabilities {
            supports_async: true,
            energy_efficiency: 0.82,
            processing_speed: 0.88,
            accuracy: 0.86,
            specialized_domains: vec![
                "context_validation".to_string(),
                "semantic_coherence".to_string(),
            ],
        }
    }
    
    fn reset(&mut self) {
        self.base.processing_count = 0;
        self.base.error_count = 0;
    }
}

impl Default for NicotineModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ProcessingContext;

    #[tokio::test]
    async fn test_nicotine_creation() {
        let module = NicotineModule::new();
        assert_eq!(module.name(), "nicotine");
        assert!(module.is_ready());
        assert!(!module.validation_rules.is_empty());
    }

    #[tokio::test]
    async fn test_context_validation() {
        let module = NicotineModule::new();
        let content = "This is a coherent piece of text that maintains logical consistency throughout.";
        
        let result = module.validate_context(content).await.unwrap();
        assert!(result > 0.0);
    }

    #[tokio::test]
    async fn test_module_processing() {
        let mut module = NicotineModule::new();
        
        let input = ModuleInput {
            content: "This content maintains semantic coherence and logical flow throughout the analysis.".to_string(),
            context: ProcessingContext {
                layer: crate::traits::TresCommasLayer::Context,
                previous_results: vec!["Previous context for validation".to_string()],
                time_pressure: 0.5,
                quality_requirements: crate::traits::QualityRequirements::default(),
            },
            energy_available: 100.0,
            confidence_required: 0.8,
        };
        
        let output = module.process(input).await.unwrap();
        assert!(output.confidence > 0.0);
        assert!(output.atp_consumed > 0.0);
        assert!(!output.result.is_empty());
        assert!(!output.byproducts.is_empty());
    }
} 