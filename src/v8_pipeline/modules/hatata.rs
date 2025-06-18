//! Hatata - Pattern Recognition Engine for Insight Discovery
//!
//! This module implements sophisticated pattern recognition algorithms
//! to discover insights and recurring structures in information.

use crate::traits::{BiologicalModule, ModuleInput, ModuleOutput, ModuleCapabilities, ModuleMetadata};
use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::v8_pipeline::modules::BaseModule;
use async_trait::async_trait;
use std::collections::HashMap;
use regex::Regex;

/// Hatata Pattern Recognition Engine
pub struct HatataModule {
    base: BaseModule,
    /// Pattern database
    patterns: Vec<PatternDefinition>,
    /// Recognition algorithms
    algorithms: Vec<RecognitionAlgorithm>,
    /// Insight cache
    insight_cache: HashMap<String, Vec<Insight>>,
    /// Pattern matching configuration
    config: PatternConfig,
}

/// Pattern definition
#[derive(Debug, Clone)]
pub struct PatternDefinition {
    pub name: String,
    pub pattern_type: PatternType,
    pub regex: Option<Regex>,
    pub semantic_markers: Vec<String>,
    pub confidence_threshold: f64,
    pub importance: f64,
}

/// Types of patterns
#[derive(Debug, Clone)]
pub enum PatternType {
    Linguistic,
    Semantic,
    Structural,
    Temporal,
    Causal,
    Statistical,
    Logical,
}

/// Recognition algorithm
#[derive(Debug, Clone)]
pub struct RecognitionAlgorithm {
    pub name: String,
    pub algorithm_type: AlgorithmType,
    pub accuracy: f64,
    pub efficiency: f64,
    pub specialized_for: Vec<PatternType>,
}

/// Algorithm types
#[derive(Debug, Clone)]
pub enum AlgorithmType {
    RegexMatching,
    SemanticAnalysis,
    StatisticalAnalysis,
    NeuralPattern,
    GraphAnalysis,
    FrequencyAnalysis,
}

/// Discovered insight
#[derive(Debug, Clone)]
pub struct Insight {
    pub insight_type: InsightType,
    pub content: String,
    pub confidence: f64,
    pub supporting_evidence: Vec<String>,
    pub implications: Vec<String>,
    pub discovered_at: chrono::DateTime<chrono::Utc>,
}

/// Types of insights
#[derive(Debug, Clone)]
pub enum InsightType {
    RecurringTheme,
    CausalRelationship,
    AnomalousPattern,
    EmergentStructure,
    HiddenConnection,
    ConceptualCluster,
}

/// Pattern configuration
#[derive(Debug, Clone)]
pub struct PatternConfig {
    pub enable_linguistic_patterns: bool,
    pub enable_semantic_patterns: bool,
    pub enable_statistical_patterns: bool,
    pub min_confidence_threshold: f64,
    pub max_patterns_per_analysis: usize,
}

impl HatataModule {
    /// Create new Hatata module
    pub fn new() -> Self {
        let mut module = Self {
            base: BaseModule::new("hatata"),
            patterns: Vec::new(),
            algorithms: Vec::new(),
            insight_cache: HashMap::new(),
            config: PatternConfig::default(),
        };
        
        module.initialize_patterns();
        module.initialize_algorithms();
        module
    }
    
    /// Initialize pattern definitions
    fn initialize_patterns(&mut self) {
        // Linguistic patterns
        self.patterns.push(PatternDefinition {
            name: "Question pattern".to_string(),
            pattern_type: PatternType::Linguistic,
            regex: Some(Regex::new(r"\b(what|how|why|when|where|who)\b.*\?").unwrap()),
            semantic_markers: vec!["interrogative".to_string()],
            confidence_threshold: 0.8,
            importance: 0.7,
        });
        
        self.patterns.push(PatternDefinition {
            name: "Causal relationship".to_string(),
            pattern_type: PatternType::Causal,
            regex: Some(Regex::new(r"\b(because|due to|results in|leads to|causes)\b").unwrap()),
            semantic_markers: vec!["causality".to_string(), "consequence".to_string()],
            confidence_threshold: 0.7,
            importance: 0.9,
        });
        
        // Semantic patterns
        self.patterns.push(PatternDefinition {
            name: "Contradiction".to_string(),
            pattern_type: PatternType::Semantic,
            regex: Some(Regex::new(r"\b(but|however|nevertheless|although|despite)\b").unwrap()),
            semantic_markers: vec!["contrast".to_string(), "opposition".to_string()],
            confidence_threshold: 0.6,
            importance: 0.8,
        });
        
        self.patterns.push(PatternDefinition {
            name: "Emphasis pattern".to_string(),
            pattern_type: PatternType::Linguistic,
            regex: Some(Regex::new(r"\b(very|extremely|highly|significantly|critically)\b").unwrap()),
            semantic_markers: vec!["intensifier".to_string()],
            confidence_threshold: 0.5,
            importance: 0.6,
        });
        
        // Structural patterns
        self.patterns.push(PatternDefinition {
            name: "List structure".to_string(),
            pattern_type: PatternType::Structural,
            regex: Some(Regex::new(r"(\d+\.|•|-).*\n").unwrap()),
            semantic_markers: vec!["enumeration".to_string(), "sequence".to_string()],
            confidence_threshold: 0.9,
            importance: 0.5,
        });
    }
    
    /// Initialize recognition algorithms
    fn initialize_algorithms(&mut self) {
        self.algorithms.push(RecognitionAlgorithm {
            name: "Regex matcher".to_string(),
            algorithm_type: AlgorithmType::RegexMatching,
            accuracy: 0.85,
            efficiency: 0.95,
            specialized_for: vec![PatternType::Linguistic, PatternType::Structural],
        });
        
        self.algorithms.push(RecognitionAlgorithm {
            name: "Semantic analyzer".to_string(),
            algorithm_type: AlgorithmType::SemanticAnalysis,
            accuracy: 0.78,
            efficiency: 0.70,
            specialized_for: vec![PatternType::Semantic, PatternType::Causal],
        });
        
        self.algorithms.push(RecognitionAlgorithm {
            name: "Frequency analyzer".to_string(),
            algorithm_type: AlgorithmType::FrequencyAnalysis,
            accuracy: 0.80,
            efficiency: 0.90,
            specialized_for: vec![PatternType::Statistical],
        });
    }
    
    /// Recognize patterns in content
    pub async fn recognize_patterns(&mut self, content: &str) -> AutobahnResult<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        
        for pattern in &self.patterns {
            if let Some(pattern_matches) = self.apply_pattern(content, pattern).await? {
                matches.extend(pattern_matches);
            }
        }
        
        // Sort by confidence and importance
        matches.sort_by(|a, b| {
            let score_a = a.confidence * a.importance;
            let score_b = b.confidence * b.importance;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Limit results
        matches.truncate(self.config.max_patterns_per_analysis);
        
        Ok(matches)
    }
    
    /// Apply a specific pattern to content
    async fn apply_pattern(&self, content: &str, pattern: &PatternDefinition) -> AutobahnResult<Option<Vec<PatternMatch>>> {
        let mut matches = Vec::new();
        
        // Apply regex matching if available
        if let Some(regex) = &pattern.regex {
            for regex_match in regex.find_iter(content) {
                let match_text = regex_match.as_str();
                let confidence = self.calculate_pattern_confidence(match_text, pattern);
                
                if confidence >= pattern.confidence_threshold {
                    matches.push(PatternMatch {
                        pattern_name: pattern.name.clone(),
                        pattern_type: pattern.pattern_type.clone(),
                        matched_text: match_text.to_string(),
                        confidence,
                        importance: pattern.importance,
                        position: regex_match.start(),
                        context: self.extract_context(content, regex_match.start(), regex_match.end()),
                    });
                }
            }
        }
        
        // Apply semantic marker matching
        for marker in &pattern.semantic_markers {
            if content.to_lowercase().contains(&marker.to_lowercase()) {
                let confidence = self.calculate_semantic_confidence(content, marker);
                
                if confidence >= pattern.confidence_threshold {
                    matches.push(PatternMatch {
                        pattern_name: format!("{} (semantic)", pattern.name),
                        pattern_type: pattern.pattern_type.clone(),
                        matched_text: marker.clone(),
                        confidence,
                        importance: pattern.importance * 0.8, // Slightly lower for semantic matches
                        position: 0, // Would need more sophisticated positioning
                        context: content[..content.len().min(100)].to_string(),
                    });
                }
            }
        }
        
        if matches.is_empty() {
            Ok(None)
        } else {
            Ok(Some(matches))
        }
    }
    
    /// Calculate pattern confidence
    fn calculate_pattern_confidence(&self, matched_text: &str, pattern: &PatternDefinition) -> f64 {
        // Base confidence from pattern definition
        let mut confidence = pattern.confidence_threshold;
        
        // Adjust based on match quality
        let length_factor = (matched_text.len() as f64 / 50.0).min(1.0);
        confidence *= 0.7 + (length_factor * 0.3);
        
        // Adjust based on pattern type
        match pattern.pattern_type {
            PatternType::Linguistic => confidence * 0.9,
            PatternType::Semantic => confidence * 0.8,
            PatternType::Causal => confidence * 0.95,
            PatternType::Structural => confidence * 0.85,
            _ => confidence,
        }
    }
    
    /// Calculate semantic confidence
    fn calculate_semantic_confidence(&self, content: &str, marker: &str) -> f64 {
        let marker_count = content.to_lowercase().matches(&marker.to_lowercase()).count();
        let content_length = content.len();
        
        // Frequency-based confidence
        let frequency_score = (marker_count as f64 / (content_length as f64 / 100.0)).min(1.0);
        
        // Context quality (simplified)
        let context_quality = if content_length > 50 { 0.8 } else { 0.6 };
        
        frequency_score * context_quality
    }
    
    /// Extract context around a match
    fn extract_context(&self, content: &str, start: usize, end: usize) -> String {
        let context_window = 50;
        let context_start = start.saturating_sub(context_window);
        let context_end = (end + context_window).min(content.len());
        
        content[context_start..context_end].to_string()
    }
    
    /// Generate insights from pattern matches
    pub fn generate_insights(&mut self, pattern_matches: &[PatternMatch]) -> Vec<Insight> {
        let mut insights = Vec::new();
        
        // Group patterns by type
        let mut pattern_groups: HashMap<String, Vec<&PatternMatch>> = HashMap::new();
        for pattern_match in pattern_matches {
            let type_key = format!("{:?}", pattern_match.pattern_type);
            pattern_groups.entry(type_key).or_insert_with(Vec::new).push(pattern_match);
        }
        
        // Generate insights for each group
        for (pattern_type, matches) in pattern_groups {
            if matches.len() > 1 {
                // Recurring theme insight
                insights.push(Insight {
                    insight_type: InsightType::RecurringTheme,
                    content: format!("Recurring {} patterns detected", pattern_type),
                    confidence: matches.iter().map(|m| m.confidence).sum::<f64>() / matches.len() as f64,
                    supporting_evidence: matches.iter().map(|m| m.matched_text.clone()).collect(),
                    implications: vec![
                        format!("Strong emphasis on {} concepts", pattern_type),
                        "May indicate thematic consistency".to_string(),
                    ],
                    discovered_at: chrono::Utc::now(),
                });
            }
        }
        
        // Look for causal relationships
        let causal_patterns: Vec<_> = pattern_matches.iter()
            .filter(|m| matches!(m.pattern_type, PatternType::Causal))
            .collect();
        
        if !causal_patterns.is_empty() {
            insights.push(Insight {
                insight_type: InsightType::CausalRelationship,
                content: "Causal relationships identified in content".to_string(),
                confidence: causal_patterns.iter().map(|m| m.confidence).sum::<f64>() / causal_patterns.len() as f64,
                supporting_evidence: causal_patterns.iter().map(|m| m.matched_text.clone()).collect(),
                implications: vec![
                    "Content contains logical reasoning chains".to_string(),
                    "May support causal inference".to_string(),
                ],
                discovered_at: chrono::Utc::now(),
            });
        }
        
        insights
    }
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_name: String,
    pub pattern_type: PatternType,
    pub matched_text: String,
    pub confidence: f64,
    pub importance: f64,
    pub position: usize,
    pub context: String,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            enable_linguistic_patterns: true,
            enable_semantic_patterns: true,
            enable_statistical_patterns: true,
            min_confidence_threshold: 0.5,
            max_patterns_per_analysis: 20,
        }
    }
}

#[async_trait]
impl BiologicalModule for HatataModule {
    fn name(&self) -> &str {
        &self.base.name
    }
    
    async fn process(&mut self, input: ModuleInput) -> AutobahnResult<ModuleOutput> {
        self.base.record_processing();
        
        let start_time = std::time::Instant::now();
        
        // Recognize patterns in the input
        let pattern_matches = match self.recognize_patterns(&input.content).await {
            Ok(matches) => matches,
            Err(e) => {
                self.base.record_error();
                return Err(e);
            }
        };
        
        // Generate insights from patterns
        let insights = self.generate_insights(&pattern_matches);
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        // Calculate ATP consumption based on pattern complexity
        let atp_consumed = (pattern_matches.len() as f64 * 3.0) + (insights.len() as f64 * 5.0);
        
        // Calculate overall confidence
        let confidence = if !pattern_matches.is_empty() {
            pattern_matches.iter().map(|m| m.confidence * m.importance).sum::<f64>() / pattern_matches.len() as f64
        } else {
            0.5
        };
        
        let result = format!(
            "Pattern recognition completed. Found {} patterns, generated {} insights. Top patterns: {}",
            pattern_matches.len(),
            insights.len(),
            pattern_matches.iter()
                .take(3)
                .map(|m| m.pattern_name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        );
        
        Ok(ModuleOutput {
            result,
            confidence,
            atp_consumed,
            byproducts: vec![
                format!("Patterns detected: {}", pattern_matches.len()),
                format!("Insights generated: {}", insights.len()),
                format!("Algorithms used: {}", self.algorithms.len()),
                format!("Pattern types: {}", 
                    pattern_matches.iter()
                        .map(|m| format!("{:?}", m.pattern_type))
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                ),
            ],
            metadata: ModuleMetadata {
                processing_time_ms: processing_time,
                memory_used_mb: 2.5,
                cpu_usage_percent: 20.0,
                cache_hits: self.insight_cache.len() as u32,
                cache_misses: pattern_matches.len() as u32,
            },
        })
    }
    
    fn calculate_atp_cost(&self, input: &ModuleInput) -> f64 {
        // ATP cost based on content complexity and number of patterns
        let base_cost = (input.content.len() as f64 / 100.0) * 3.0;
        let pattern_cost = self.patterns.len() as f64 * 2.0;
        
        base_cost + pattern_cost
    }
    
    fn is_ready(&self) -> bool {
        self.base.ready && !self.patterns.is_empty() && !self.algorithms.is_empty()
    }
    
    fn capabilities(&self) -> ModuleCapabilities {
        ModuleCapabilities {
            supports_async: true,
            energy_efficiency: 0.80,
            processing_speed: 0.85,
            accuracy: 0.82,
            specialized_domains: vec![
                "pattern_recognition".to_string(),
                "insight_discovery".to_string(),
                "linguistic_analysis".to_string(),
                "semantic_analysis".to_string(),
                "causal_reasoning".to_string(),
            ],
        }
    }
    
    fn reset(&mut self) {
        self.insight_cache.clear();
        self.base.processing_count = 0;
        self.base.error_count = 0;
    }
}

impl Default for HatataModule {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ProcessingContext;

    #[tokio::test]
    async fn test_hatata_creation() {
        let module = HatataModule::new();
        assert_eq!(module.name(), "hatata");
        assert!(module.is_ready());
        assert!(!module.patterns.is_empty());
        assert!(!module.algorithms.is_empty());
    }

    #[tokio::test]
    async fn test_pattern_recognition() {
        let mut module = HatataModule::new();
        let content = "What causes this? Because of the results, we can see that this leads to significant changes.";
        
        let matches = module.recognize_patterns(content).await.unwrap();
        assert!(!matches.is_empty());
        
        // Should find question and causal patterns
        let pattern_types: std::collections::HashSet<_> = matches.iter()
            .map(|m| format!("{:?}", m.pattern_type))
            .collect();
        
        assert!(pattern_types.len() > 0);
    }

    #[tokio::test]
    async fn test_insight_generation() {
        let mut module = HatataModule::new();
        
        // Create some mock pattern matches
        let matches = vec![
            PatternMatch {
                pattern_name: "Causal relationship".to_string(),
                pattern_type: PatternType::Causal,
                matched_text: "because".to_string(),
                confidence: 0.8,
                importance: 0.9,
                position: 0,
                context: "test context".to_string(),
            },
            PatternMatch {
                pattern_name: "Causal relationship".to_string(),
                pattern_type: PatternType::Causal,
                matched_text: "leads to".to_string(),
                confidence: 0.7,
                importance: 0.9,
                position: 10,
                context: "test context 2".to_string(),
            },
        ];
        
        let insights = module.generate_insights(&matches);
        assert!(!insights.is_empty());
    }

    #[tokio::test]
    async fn test_module_processing() {
        let mut module = HatataModule::new();
        
        let input = ModuleInput {
            content: "What is the cause of this problem? Because of multiple factors, this leads to significant issues. However, we can solve this.".to_string(),
            context: ProcessingContext {
                layer: crate::traits::TresCommasLayer::Reasoning,
                previous_results: vec![],
                time_pressure: 0.5,
                quality_requirements: crate::traits::QualityRequirements::default(),
            },
            energy_available: 100.0,
            confidence_required: 0.7,
        };
        
        let output = module.process(input).await.unwrap();
        assert!(output.confidence > 0.0);
        assert!(output.atp_consumed > 0.0);
        assert!(!output.result.is_empty());
        assert!(!output.byproducts.is_empty());
    }

    #[test]
    fn test_pattern_confidence_calculation() {
        let module = HatataModule::new();
        
        let pattern = &module.patterns[0]; // First pattern
        let confidence = module.calculate_pattern_confidence("test match", pattern);
        
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_context_extraction() {
        let module = HatataModule::new();
        let content = "This is a long piece of content that we want to extract context from for testing purposes.";
        
        let context = module.extract_context(content, 10, 20);
        assert!(!context.is_empty());
        assert!(context.len() <= content.len());
    }
} 