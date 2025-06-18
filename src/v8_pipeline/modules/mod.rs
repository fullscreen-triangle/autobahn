//! V8 Biological Intelligence Modules
//!
//! This module contains the eleven specialized intelligence modules that implement
//! the biological metabolism pipeline with quantum-oscillatory enhancements:
//!
//! ## Traditional Modules (1-9):
//! 1. **Mzekezeke** - Bayesian Belief Engine for probabilistic reasoning
//! 2. **Diggiden** - Adversarial Testing System for robustness validation  
//! 3. **Hatata** - Pattern Recognition Engine for insight discovery
//! 4. **Spectacular** - Paradigm Detection System for framework analysis
//! 5. **Nicotine** - Context Validation Engine for semantic coherence
//! 6. **Clothesline** - Comprehension Validator through strategic occlusion
//! 7. **Zengeza** - Noise Reduction Filter for signal clarity
//! 8. **Diadochi** - Multi-domain LLM Orchestration for external integration
//! 9. **Pungwe** - ATP Synthase for metacognitive oversight and self-deception detection
//! 
//! ## NEW: Quantum-Oscillatory Modules (10-11):
//! 10. **Foursidedtriangle** - Non-Euclidean Semantic Processor for paradoxical structures
//! 11. **OscillationEndpointManager** - Entropy Quantification Engine for direct entropy control

pub mod mzekezeke;
pub mod diggiden;
pub mod hatata;
pub mod spectacular;
pub mod nicotine;
pub mod clothesline;
pub mod zengeza;
pub mod diadochi;
pub mod pungwe;
// NEW: Quantum-oscillatory modules
pub mod foursidedtriangle;
pub mod oscillation_endpoint_manager;

// Re-export all modules
pub use mzekezeke::MzekezekerModule;
pub use diggiden::DiggidenModule;
pub use hatata::HatataModule;
pub use spectacular::SpectacularModule;
pub use nicotine::NicotineModule;
pub use clothesline::ClotheslineModule;
pub use zengeza::ZengazerModule;
pub use diadochi::DiadochiModule;
pub use pungwe::PungweModule;
// NEW: Quantum-oscillatory modules
pub use foursidedtriangle::Foursidedtriangle;
pub use oscillation_endpoint_manager::OscillationEndpointManager;

use crate::traits::{BiologicalModule, ModuleInput, ModuleOutput, ModuleCapabilities, ModuleMetadata};
use crate::error::{AutobahnError, AutobahnResult};
use async_trait::async_trait;

/// Module factory for creating V8 modules
pub struct ModuleFactory;

impl ModuleFactory {
    /// Create all V8 modules (now including quantum-oscillatory modules)
    pub fn create_all_modules() -> std::collections::HashMap<String, Box<dyn BiologicalModule + Send + Sync>> {
        let mut modules = std::collections::HashMap::new();
        
        // Traditional V8 modules
        modules.insert("mzekezeke".to_string(), Box::new(MzekezekerModule::new()) as Box<dyn BiologicalModule + Send + Sync>);
        modules.insert("diggiden".to_string(), Box::new(DiggidenModule::new()) as Box<dyn BiologicalModule + Send + Sync>);
        modules.insert("hatata".to_string(), Box::new(HatataModule::new()) as Box<dyn BiologicalModule + Send + Sync>);
        modules.insert("spectacular".to_string(), Box::new(SpectacularModule::new()) as Box<dyn BiologicalModule + Send + Sync>);
        modules.insert("nicotine".to_string(), Box::new(NicotineModule::new()) as Box<dyn BiologicalModule + Send + Sync>);
        modules.insert("clothesline".to_string(), Box::new(ClotheslineModule::new()) as Box<dyn BiologicalModule + Send + Sync>);
        modules.insert("zengeza".to_string(), Box::new(ZengazerModule::new()) as Box<dyn BiologicalModule + Send + Sync>);
        modules.insert("diadochi".to_string(), Box::new(DiadochiModule::new()) as Box<dyn BiologicalModule + Send + Sync>);
        modules.insert("pungwe".to_string(), Box::new(PungweModule::new()) as Box<dyn BiologicalModule + Send + Sync>);
        
        // NEW: Quantum-oscillatory modules
        modules.insert("foursidedtriangle".to_string(), Box::new(Foursidedtriangle::new()) as Box<dyn BiologicalModule + Send + Sync>);
        modules.insert("oscillation_endpoint_manager".to_string(), Box::new(OscillationEndpointManager::new()) as Box<dyn BiologicalModule + Send + Sync>);
        
        modules
    }
    
    /// Create a specific module by name
    pub fn create_module(name: &str) -> AutobahnResult<Box<dyn BiologicalModule + Send + Sync>> {
        match name {
            // Traditional modules
            "mzekezeke" => Ok(Box::new(MzekezekerModule::new())),
            "diggiden" => Ok(Box::new(DiggidenModule::new())),
            "hatata" => Ok(Box::new(HatataModule::new())),
            "spectacular" => Ok(Box::new(SpectacularModule::new())),
            "nicotine" => Ok(Box::new(NicotineModule::new())),
            "clothesline" => Ok(Box::new(ClotheslineModule::new())),
            "zengeza" => Ok(Box::new(ZengazerModule::new())),
            "diadochi" => Ok(Box::new(DiadochiModule::new())),
            "pungwe" => Ok(Box::new(PungweModule::new())),
            // NEW: Quantum-oscillatory modules
            "foursidedtriangle" => Ok(Box::new(Foursidedtriangle::new())),
            "oscillation_endpoint_manager" => Ok(Box::new(OscillationEndpointManager::new())),
            _ => Err(AutobahnError::ConfigurationError(
                format!("Unknown module: {}", name)
            )),
        }
    }
    
    /// Get module names (including quantum-oscillatory modules)
    pub fn module_names() -> Vec<&'static str> {
        vec![
            // Traditional modules
            "mzekezeke",
            "diggiden", 
            "hatata",
            "spectacular",
            "nicotine",
            "clothesline",
            "zengeza",
            "diadochi",
            "pungwe",
            // NEW: Quantum-oscillatory modules
            "foursidedtriangle",
            "oscillation_endpoint_manager",
        ]
    }
}

/// Base implementation for common module functionality
pub struct BaseModule {
    pub name: String,
    pub ready: bool,
    pub processing_count: u64,
    pub error_count: u64,
}

impl BaseModule {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ready: true,
            processing_count: 0,
            error_count: 0,
        }
    }
    
    pub fn record_processing(&mut self) {
        self.processing_count += 1;
    }
    
    pub fn record_error(&mut self) {
        self.error_count += 1;
    }
    
    pub fn success_rate(&self) -> f64 {
        if self.processing_count == 0 {
            return 1.0;
        }
        
        let successful = self.processing_count - self.error_count;
        successful as f64 / self.processing_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_factory_creation() {
        let modules = ModuleFactory::create_all_modules();
        assert_eq!(modules.len(), 11); // Updated to include quantum-oscillatory modules
        
        for module_name in ModuleFactory::module_names() {
            assert!(modules.contains_key(module_name));
        }
    }

    #[test]
    fn test_individual_module_creation() {
        for module_name in ModuleFactory::module_names() {
            let module = ModuleFactory::create_module(module_name);
            assert!(module.is_ok());
        }
    }

    #[test]
    fn test_unknown_module_creation() {
        let result = ModuleFactory::create_module("unknown_module");
        assert!(result.is_err());
    }

    #[test]
    fn test_base_module() {
        let mut base = BaseModule::new("test");
        assert_eq!(base.name, "test");
        assert!(base.ready);
        assert_eq!(base.success_rate(), 1.0);
        
        base.record_processing();
        base.record_processing();
        base.record_error();
        
        assert_eq!(base.processing_count, 2);
        assert_eq!(base.error_count, 1);
        assert_eq!(base.success_rate(), 0.5);
    }
} 