//! Spectacular - Paradigm Detection System for Framework Analysis
//!
//! This module detects and analyzes paradigms, frameworks, and conceptual
//! structures within information to understand underlying methodologies.

use crate::traits::{BiologicalModule, ModuleInput, ModuleOutput, ModuleCapabilities, ModuleMetadata};
use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::v8_pipeline::modules::BaseModule;
use async_trait::async_trait;
use std::collections::HashMap;

/// Spectacular Paradigm Detection System
pub struct SpectacularModule {
    base: BaseModule,
    /// Known paradigm definitions
    paradigms: Vec<ParadigmDefinition>,
    /// Framework detection rules
    framework_rules: Vec<FrameworkRule>,
    /// Conceptual structure patterns
    structure_patterns: Vec<StructurePattern>,
    /// Analysis configuration
    config: ParadigmConfig,
}

/// Paradigm definition
#[derive(Debug, Clone)]
pub struct ParadigmDefinition {
    pub name: String,
    pub paradigm_type: ParadigmType,
    pub core_principles: Vec<String>,
    pub key_concepts: Vec<String>,
    pub methodologies: Vec<String>,
    pub indicators: Vec<String>,
    pub confidence_weight: f64,
}

/// Types of paradigms
#[derive(Debug, Clone)]
pub enum ParadigmType {
    Scientific,
    Philosophical,
    Mathematical,
    Computational,
    Biological,
    Economic,
    Social,
    Linguistic,
    Cognitive,
}

/// Framework detection rule
#[derive(Debug, Clone)]
pub struct FrameworkRule {
    pub name: String,
    pub framework_type: FrameworkType,
    pub detection_patterns: Vec<String>,
    pub required_elements: Vec<String>,
    pub optional_elements: Vec<String>,
    pub exclusion_patterns: Vec<String>,
}

/// Framework types
#[derive(Debug, Clone)]
pub enum FrameworkType {
    Theoretical,
    Methodological,
    Analytical,
    Conceptual,
    Practical,
    Evaluative,
}

/// Structure pattern for conceptual analysis
#[derive(Debug, Clone)]
pub struct StructurePattern {
    pub name: String,
    pub pattern_type: StructureType,
    pub hierarchy_indicators: Vec<String>,
    pub relationship_markers: Vec<String>,
    pub boundary_markers: Vec<String>,
}

/// Structure types
#[derive(Debug, Clone)]
pub enum StructureType {
    Hierarchical,
    Network,
    Sequential,
    Categorical,
    Causal,
    Comparative,
}

/// Paradigm analysis configuration
#[derive(Debug, Clone)]
pub struct ParadigmConfig {
    pub enable_scientific_detection: bool,
    pub enable_philosophical_detection: bool,
    pub enable_framework_analysis: bool,
    pub min_confidence_threshold: f64,
    pub max_paradigms_per_analysis: usize,
}

/// Paradigm detection result
#[derive(Debug, Clone)]
pub struct ParadigmDetection {
    pub paradigm_name: String,
    pub paradigm_type: ParadigmType,
    pub confidence: f64,
    pub evidence: Vec<String>,
    pub core_concepts_found: Vec<String>,
    pub methodological_indicators: Vec<String>,
    pub structural_analysis: StructuralAnalysis,
}

/// Structural analysis result
#[derive(Debug, Clone)]
pub struct StructuralAnalysis {
    pub structure_type: StructureType,
    pub hierarchy_depth: u32,
    pub relationship_density: f64,
    pub conceptual_coherence: f64,
    pub boundary_clarity: f64,
}

impl SpectacularModule {
    /// Create new Spectacular module
    pub fn new() -> Self {
        let mut module = Self {
            base: BaseModule::new("spectacular"),
            paradigms: Vec::new(),
            framework_rules: Vec::new(),
            structure_patterns: Vec::new(),
            config: ParadigmConfig::default(),
        };
        
        module.initialize_paradigms();
        module.initialize_framework_rules();
        module.initialize_structure_patterns();
        module
    }
    
    /// Initialize paradigm definitions
    fn initialize_paradigms(&mut self) {
        // Scientific paradigm
        self.paradigms.push(ParadigmDefinition {
            name: "Scientific Method".to_string(),
            paradigm_type: ParadigmType::Scientific,
            core_principles: vec![
                "Empirical observation".to_string(),
                "Hypothesis testing".to_string(),
                "Reproducibility".to_string(),
                "Peer review".to_string(),
            ],
            key_concepts: vec![
                "experiment".to_string(),
                "hypothesis".to_string(),
                "data".to_string(),
                "evidence".to_string(),
                "theory".to_string(),
            ],
            methodologies: vec![
                "controlled experiment".to_string(),
                "statistical analysis".to_string(),
                "systematic observation".to_string(),
            ],
            indicators: vec![
                "research".to_string(),
                "study".to_string(),
                "analysis".to_string(),
                "findings".to_string(),
            ],
            confidence_weight: 0.9,
        });
        
        // Computational paradigm
        self.paradigms.push(ParadigmDefinition {
            name: "Computational Thinking".to_string(),
            paradigm_type: ParadigmType::Computational,
            core_principles: vec![
                "Algorithmic thinking".to_string(),
                "Decomposition".to_string(),
                "Pattern recognition".to_string(),
                "Abstraction".to_string(),
            ],
            key_concepts: vec![
                "algorithm".to_string(),
                "process".to_string(),
                "system".to_string(),
                "optimization".to_string(),
                "efficiency".to_string(),
            ],
            methodologies: vec![
                "iterative development".to_string(),
                "modular design".to_string(),
                "testing".to_string(),
            ],
            indicators: vec![
                "implementation".to_string(),
                "optimization".to_string(),
                "processing".to_string(),
                "computational".to_string(),
            ],
            confidence_weight: 0.85,
        });
        
        // Biological paradigm
        self.paradigms.push(ParadigmDefinition {
            name: "Systems Biology".to_string(),
            paradigm_type: ParadigmType::Biological,
            core_principles: vec![
                "Emergent properties".to_string(),
                "Evolutionary adaptation".to_string(),
                "Homeostasis".to_string(),
                "Network interactions".to_string(),
            ],
            key_concepts: vec![
                "evolution".to_string(),
                "adaptation".to_string(),
                "metabolism".to_string(),
                "network".to_string(),
                "emergence".to_string(),
            ],
            methodologies: vec![
                "systems analysis".to_string(),
                "network modeling".to_string(),
                "evolutionary analysis".to_string(),
            ],
            indicators: vec![
                "biological".to_string(),
                "organic".to_string(),
                "cellular".to_string(),
                "metabolic".to_string(),
            ],
            confidence_weight: 0.88,
        });
    }
    
    /// Initialize framework detection rules
    fn initialize_framework_rules(&mut self) {
        self.framework_rules.push(FrameworkRule {
            name: "Problem-Solution Framework".to_string(),
            framework_type: FrameworkType::Analytical,
            detection_patterns: vec![
                "problem".to_string(),
                "solution".to_string(),
                "challenge".to_string(),
                "approach".to_string(),
            ],
            required_elements: vec!["problem".to_string(), "solution".to_string()],
            optional_elements: vec!["analysis".to_string(), "evaluation".to_string()],
            exclusion_patterns: vec![],
        });
        
        self.framework_rules.push(FrameworkRule {
            name: "Cause-Effect Framework".to_string(),
            framework_type: FrameworkType::Causal,
            detection_patterns: vec![
                "cause".to_string(),
                "effect".to_string(),
                "because".to_string(),
                "results in".to_string(),
                "leads to".to_string(),
            ],
            required_elements: vec!["cause".to_string(), "effect".to_string()],
            optional_elements: vec!["mechanism".to_string(), "correlation".to_string()],
            exclusion_patterns: vec![],
        });
        
        self.framework_rules.push(FrameworkRule {
            name: "Comparative Framework".to_string(),
            framework_type: FrameworkType::Analytical,
            detection_patterns: vec![
                "compare".to_string(),
                "contrast".to_string(),
                "versus".to_string(),
                "difference".to_string(),
                "similarity".to_string(),
            ],
            required_elements: vec!["comparison".to_string()],
            optional_elements: vec!["criteria".to_string(), "evaluation".to_string()],
            exclusion_patterns: vec![],
        });
    }
    
    /// Initialize structure patterns
    fn initialize_structure_patterns(&mut self) {
        self.structure_patterns.push(StructurePattern {
            name: "Hierarchical Structure".to_string(),
            pattern_type: StructureType::Hierarchical,
            hierarchy_indicators: vec![
                "level".to_string(),
                "tier".to_string(),
                "layer".to_string(),
                "category".to_string(),
                "subcategory".to_string(),
            ],
            relationship_markers: vec![
                "consists of".to_string(),
                "includes".to_string(),
                "contains".to_string(),
                "subdivided into".to_string(),
            ],
            boundary_markers: vec![
                "distinct".to_string(),
                "separate".to_string(),
                "independent".to_string(),
            ],
        });
        
        self.structure_patterns.push(StructurePattern {
            name: "Network Structure".to_string(),
            pattern_type: StructureType::Network,
            hierarchy_indicators: vec![
                "node".to_string(),
                "connection".to_string(),
                "link".to_string(),
                "relationship".to_string(),
            ],
            relationship_markers: vec![
                "connected to".to_string(),
                "linked with".to_string(),
                "related to".to_string(),
                "interacts with".to_string(),
            ],
            boundary_markers: vec![
                "cluster".to_string(),
                "community".to_string(),
                "group".to_string(),
            ],
        });
    }
    
    /// Detect paradigms in content
    pub async fn detect_paradigms(&mut self, content: &str) -> AutobahnResult<Vec<ParadigmDetection>> {
        let mut detections = Vec::new();
        
        for paradigm in &self.paradigms {
            if let Some(detection) = self.analyze_paradigm(content, paradigm).await? {
                if detection.confidence >= self.config.min_confidence_threshold {
                    detections.push(detection);
                }
            }
        }
        
        // Sort by confidence
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit results
        detections.truncate(self.config.max_paradigms_per_analysis);
        
        Ok(detections)
    }
    
    /// Analyze content for a specific paradigm
    async fn analyze_paradigm(&self, content: &str, paradigm: &ParadigmDefinition) -> AutobahnResult<Option<ParadigmDetection>> {
        let content_lower = content.to_lowercase();
        let mut evidence = Vec::new();
        let mut core_concepts_found = Vec::new();
        let mut methodological_indicators = Vec::new();
        let mut confidence_score = 0.0;
        
        // Check for key concepts
        for concept in &paradigm.key_concepts {
            if content_lower.contains(&concept.to_lowercase()) {
                core_concepts_found.push(concept.clone());
                evidence.push(format!("Found key concept: {}", concept));
                confidence_score += 0.1;
            }
        }
        
        // Check for methodologies
        for methodology in &paradigm.methodologies {
            if content_lower.contains(&methodology.to_lowercase()) {
                methodological_indicators.push(methodology.clone());
                evidence.push(format!("Found methodology: {}", methodology));
                confidence_score += 0.15;
            }
        }
        
        // Check for indicators
        for indicator in &paradigm.indicators {
            if content_lower.contains(&indicator.to_lowercase()) {
                evidence.push(format!("Found indicator: {}", indicator));
                confidence_score += 0.05;
            }
        }
        
        // Check for core principles (semantic matching - simplified)
        for principle in &paradigm.core_principles {
            if self.semantic_match(content, principle) {
                evidence.push(format!("Found principle alignment: {}", principle));
                confidence_score += 0.2;
            }
        }
        
        // Apply paradigm-specific confidence weight
        confidence_score *= paradigm.confidence_weight;
        confidence_score = confidence_score.min(1.0);
        
        if confidence_score > 0.1 {
            // Perform structural analysis
            let structural_analysis = self.analyze_structure(content).await?;
            
            Ok(Some(ParadigmDetection {
                paradigm_name: paradigm.name.clone(),
                paradigm_type: paradigm.paradigm_type.clone(),
                confidence: confidence_score,
                evidence,
                core_concepts_found,
                methodological_indicators,
                structural_analysis,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Perform structural analysis of content
    async fn analyze_structure(&self, content: &str) -> AutobahnResult<StructuralAnalysis> {
        let content_lower = content.to_lowercase();
        let mut structure_type = StructureType::Sequential; // Default
        let mut hierarchy_depth = 0;
        let mut relationship_density = 0.0;
        let mut conceptual_coherence = 0.7; // Default
        let mut boundary_clarity = 0.6; // Default
        
        // Analyze for different structure types
        for pattern in &self.structure_patterns {
            let mut pattern_score = 0.0;
            
            // Check hierarchy indicators
            for indicator in &pattern.hierarchy_indicators {
                if content_lower.contains(&indicator.to_lowercase()) {
                    pattern_score += 0.2;
                    hierarchy_depth = hierarchy_depth.max(1);
                }
            }
            
            // Check relationship markers
            let relationship_count = pattern.relationship_markers.iter()
                .map(|marker| content_lower.matches(&marker.to_lowercase()).count())
                .sum::<usize>();
            
            relationship_density = (relationship_count as f64 / content.len() as f64 * 1000.0).min(1.0);
            pattern_score += relationship_density * 0.3;
            
            // Check boundary markers
            for marker in &pattern.boundary_markers {
                if content_lower.contains(&marker.to_lowercase()) {
                    pattern_score += 0.1;
                    boundary_clarity = (boundary_clarity + 0.1).min(1.0);
                }
            }
            
            // Update structure type if this pattern has higher score
            if pattern_score > 0.3 {
                structure_type = pattern.pattern_type.clone();
                conceptual_coherence = (conceptual_coherence + pattern_score * 0.2).min(1.0);
            }
        }
        
        // Estimate hierarchy depth based on content structure
        hierarchy_depth = hierarchy_depth.max(
            content.matches('\n').count().min(5) as u32
        );
        
        Ok(StructuralAnalysis {
            structure_type,
            hierarchy_depth,
            relationship_density,
            conceptual_coherence,
            boundary_clarity,
        })
    }
    
    /// Simple semantic matching (would use more sophisticated NLP in real implementation)
    fn semantic_match(&self, content: &str, principle: &str) -> bool {
        let content_lower = content.to_lowercase();
        let principle_lower = principle.to_lowercase();
        
        // Simple keyword-based semantic matching
        let principle_words: Vec<&str> = principle_lower.split_whitespace().collect();
        let matches = principle_words.iter()
            .filter(|word| content_lower.contains(*word))
            .count();
        
        matches as f64 / principle_words.len() as f64 > 0.5
    }
    
    /// Generate paradigm summary
    pub fn generate_paradigm_summary(&self, detections: &[ParadigmDetection]) -> String {
        if detections.is_empty() {
            return "No clear paradigms detected".to_string();
        }
        
        let primary_paradigm = &detections[0];
        let mut summary = format!(
            "Primary paradigm: {} ({:.1}% confidence)",
            primary_paradigm.paradigm_name,
            primary_paradigm.confidence * 100.0
        );
        
        if detections.len() > 1 {
            let secondary_paradigms: Vec<String> = detections[1..3].iter()
                .map(|d| format!("{} ({:.1}%)", d.paradigm_name, d.confidence * 100.0))
                .collect();
            
            summary.push_str(&format!(
                ". Secondary paradigms: {}",
                secondary_paradigms.join(", ")
            ));
        }
        
        summary
    }
}

impl Default for ParadigmConfig {
    fn default() -> Self {
        Self {
            enable_scientific_detection: true,
            enable_philosophical_detection: true,
            enable_framework_analysis: true,
            min_confidence_threshold: 0.3,
            max_paradigms_per_analysis: 5,
        }
    }
}

#[async_trait]
impl BiologicalModule for SpectacularModule {
    fn name(&self) -> &str {
        &self.base.name
    }
    
    async fn process(&mut self, input: ModuleInput) -> AutobahnResult<ModuleOutput> {
        self.base.record_processing();
        
        let start_time = std::time::Instant::now();
        
        // Detect paradigms in the input
        let paradigm_detections = match self.detect_paradigms(&input.content).await {
            Ok(detections) => detections,
            Err(e) => {
                self.base.record_error();
                return Err(e);
            }
        };
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Calculate ATP consumption based on analysis complexity
        let atp_consumed = (paradigm_detections.len() as f64 * 6.0) + 
                          (self.paradigms.len() as f64 * 2.0);
        
        // Calculate overall confidence
        let confidence = if !paradigm_detections.is_empty() {
            paradigm_detections.iter().map(|d| d.confidence).sum::<f64>() / paradigm_detections.len() as f64
        } else {
            0.4 // Low confidence when no paradigms detected
        };
        
        let paradigm_summary = self.generate_paradigm_summary(&paradigm_detections);
        
        let result = format!(
            "Paradigm analysis completed. {}. Structural coherence: {:.1}%",
            paradigm_summary,
            if !paradigm_detections.is_empty() {
                paradigm_detections[0].structural_analysis.conceptual_coherence * 100.0
            } else {
                50.0
            }
        );
        
        Ok(ModuleOutput {
            result,
            confidence,
            atp_consumed,
            byproducts: vec![
                format!("Paradigms detected: {}", paradigm_detections.len()),
                format!("Framework rules applied: {}", self.framework_rules.len()),
                format!("Structure patterns analyzed: {}", self.structure_patterns.len()),
                format!("Evidence items: {}", 
                    paradigm_detections.iter().map(|d| d.evidence.len()).sum::<usize>()
                ),
            ],
            metadata: ModuleMetadata {
                processing_time_ms: processing_time,
                memory_used_mb: 3.0,
                cpu_usage_percent: 22.0,
                cache_hits: 0,
                cache_misses: paradigm_detections.len() as u32,
            },
        })
    }
    
    fn calculate_atp_cost(&self, input: &ModuleInput) -> f64 {
        // ATP cost based on content complexity and number of paradigms to check
        let base_cost = (input.content.len() as f64 / 100.0) * 4.0;
        let paradigm_cost = self.paradigms.len() as f64 * 2.0;
        let framework_cost = self.framework_rules.len() as f64 * 1.5;
        
        base_cost + paradigm_cost + framework_cost
    }
    
    fn is_ready(&self) -> bool {
        self.base.ready && !self.paradigms.is_empty() && !self.framework_rules.is_empty()
    }
    
    fn capabilities(&self) -> ModuleCapabilities {
        ModuleCapabilities {
            supports_async: true,
            energy_efficiency: 0.75,
            processing_speed: 0.80,
            accuracy: 0.85,
            specialized_domains: vec![
                "paradigm_detection".to_string(),
                "framework_analysis".to_string(),
                "conceptual_structure".to_string(),
                "methodological_analysis".to_string(),
                "theoretical_frameworks".to_string(),
            ],
        }
    }
    
    fn reset(&mut self) {
        self.base.processing_count = 0;
        self.base.error_count = 0;
    }
}

impl Default for SpectacularModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ProcessingContext;

    #[tokio::test]
    async fn test_spectacular_creation() {
        let module = SpectacularModule::new();
        assert_eq!(module.name(), "spectacular");
        assert!(module.is_ready());
        assert!(!module.paradigms.is_empty());
        assert!(!module.framework_rules.is_empty());
    }

    #[tokio::test]
    async fn test_paradigm_detection() {
        let mut module = SpectacularModule::new();
        let content = "This research study uses experimental methodology to test the hypothesis with empirical data and statistical analysis.";
        
        let detections = module.detect_paradigms(content).await.unwrap();
        // Should detect scientific paradigm
        assert!(!detections.is_empty());
    }

    #[tokio::test]
    async fn test_structural_analysis() {
        let module = SpectacularModule::new();
        let content = "The system consists of multiple levels and layers, with hierarchical organization and clear categories.";
        
        let analysis = module.analyze_structure(content).await.unwrap();
        assert!(analysis.hierarchy_depth > 0);
        assert!(analysis.conceptual_coherence > 0.0);
    }

    #[tokio::test]
    async fn test_module_processing() {
        let mut module = SpectacularModule::new();
        
        let input = ModuleInput {
            content: "This computational approach uses algorithmic processing to optimize system efficiency through iterative development.".to_string(),
            context: ProcessingContext {
                layer: crate::traits::TresCommasLayer::Reasoning,
                previous_results: vec![],
                time_pressure: 0.5,
                quality_requirements: crate::traits::QualityRequirements::default(),
            },
            energy_available: 150.0,
            confidence_required: 0.7,
        };
        
        let output = module.process(input).await.unwrap();
        assert!(output.confidence > 0.0);
        assert!(output.atp_consumed > 0.0);
        assert!(!output.result.is_empty());
        assert!(!output.byproducts.is_empty());
    }

    #[test]
    fn test_semantic_matching() {
        let module = SpectacularModule::new();
        
        let content = "This involves empirical observation and systematic testing";
        let principle = "Empirical observation";
        
        assert!(module.semantic_match(content, principle));
        
        let unrelated_content = "This is about cooking recipes";
        assert!(!module.semantic_match(unrelated_content, principle));
    }

    #[test]
    fn test_paradigm_summary_generation() {
        let module = SpectacularModule::new();
        
        let detections = vec![
            ParadigmDetection {
                paradigm_name: "Scientific Method".to_string(),
                paradigm_type: ParadigmType::Scientific,
                confidence: 0.85,
                evidence: vec![],
                core_concepts_found: vec![],
                methodological_indicators: vec![],
                structural_analysis: StructuralAnalysis {
                    structure_type: StructureType::Hierarchical,
                    hierarchy_depth: 2,
                    relationship_density: 0.5,
                    conceptual_coherence: 0.8,
                    boundary_clarity: 0.7,
                },
            },
        ];
        
        let summary = module.generate_paradigm_summary(&detections);
        assert!(summary.contains("Scientific Method"));
        assert!(summary.contains("85"));
    }
} 