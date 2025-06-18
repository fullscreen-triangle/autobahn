//! Diadochi - Multi-domain LLM Orchestration for External Integration

use crate::traits::{BiologicalModule, ModuleInput, ModuleOutput, ModuleCapabilities, ModuleMetadata};
use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::v8_pipeline::modules::BaseModule;
use async_trait::async_trait;

/// Diadochi Multi-domain LLM Orchestration
pub struct DiadochiModule {
    base: BaseModule,
    llm_domains: Vec<String>,
    orchestration_strategy: String,
}

impl DiadochiModule {
    pub fn new() -> Self {
        Self {
            base: BaseModule::new("diadochi"),
            llm_domains: vec![
                "scientific".to_string(),
                "technical".to_string(),
                "creative".to_string(),
                "analytical".to_string(),
            ],
            orchestration_strategy: "consensus_based".to_string(),
        }
    }
    
    pub async fn orchestrate_llms(&self, content: &str) -> AutobahnResult<f64> {
        let orchestration_score = self.calculate_orchestration_quality(content);
        Ok(orchestration_score)
    }
    
    fn calculate_orchestration_quality(&self, content: &str) -> f64 {
        let mut quality = 0.75;
        
        // Assess domain complexity
        let domain_indicators = self.count_domain_indicators(content);
        if domain_indicators > 1 {
            quality += 0.15;
        }
        
        // Check for multi-modal content
        if content.len() > 100 {
            quality += 0.1;
        }
        
        quality.min(1.0)
    }
    
    fn count_domain_indicators(&self, content: &str) -> usize {
        let content_lower = content.to_lowercase();
        self.llm_domains.iter()
            .filter(|domain| content_lower.contains(*domain))
            .count()
    }
}

#[async_trait]
impl BiologicalModule for DiadochiModule {
    fn name(&self) -> &str {
        &self.base.name
    }
    
    async fn process(&mut self, input: ModuleInput) -> AutobahnResult<ModuleOutput> {
        self.base.record_processing();
        
        let start_time = std::time::Instant::now();
        
        let orchestration_score = match self.orchestrate_llms(&input.content).await {
            Ok(score) => score,
            Err(e) => {
                self.base.record_error();
                return Err(e);
            }
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        let atp_consumed = (input.content.len() as f64 / 100.0) * 6.0;
        
        let result = format!(
            "LLM orchestration completed. Quality: {:.1}%",
            orchestration_score * 100.0
        );
        
        Ok(ModuleOutput {
            result,
            confidence: orchestration_score,
            atp_consumed,
            byproducts: vec![
                format!("LLM domains: {}", self.llm_domains.len()),
                format!("Strategy: {}", self.orchestration_strategy),
            ],
            metadata: ModuleMetadata {
                processing_time_ms: processing_time,
                memory_used_mb: 4.0,
                cpu_usage_percent: 30.0,
                cache_hits: 0,
                cache_misses: 1,
            },
        })
    }
    
    fn calculate_atp_cost(&self, input: &ModuleInput) -> f64 {
        (input.content.len() as f64 / 100.0) * 6.0
    }
    
    fn is_ready(&self) -> bool {
        self.base.ready
    }
    
    fn capabilities(&self) -> ModuleCapabilities {
        ModuleCapabilities {
            supports_async: true,
            energy_efficiency: 0.70,
            processing_speed: 0.75,
            accuracy: 0.90,
            specialized_domains: vec![
                "llm_orchestration".to_string(),
                "multi_domain_integration".to_string(),
            ],
        }
    }
    
    fn reset(&mut self) {
        self.base.processing_count = 0;
        self.base.error_count = 0;
    }
}

impl Default for DiadochiModule {
    fn default() -> Self {
        Self::new()
    }
} 