//! Complete Oscillatory Bio-Metabolic RAG System
//! 
//! This module implements the full theoretical framework from code.md:
//! - Universal Oscillation Equation integration
//! - Quantum membrane computation with ENAQT
//! - 10-level hierarchy processing
//! - ATP-driven metabolic modes
//! - Three biological processing layers
//! - Oscillatory entropy calculations

use crate::error::{AutobahnError, AutobahnResult};
use crate::oscillatory::{UniversalOscillator, OscillationProfile, OscillationPhase};
use crate::quantum::{QuantumMembraneProcessor, ENAQTProcessor};
use crate::atp::{QuantumATPManager, MetabolicMode, ATPState};
use crate::hierarchy::{NestedHierarchyProcessor, HierarchyLevel, HierarchyResult};
use crate::biological::{BiologicalLayerProcessor, BiologicalLayer, BiologicalProcessingResult};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{timeout, Duration};
use chrono::{DateTime, Utc};

/// Configuration for the complete oscillatory RAG system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryRAGConfig {
    /// Operating temperature in Kelvin (affects quantum coherence)
    pub temperature: f64,
    /// Target entropy for oscillatory processing
    pub target_entropy: f64,
    /// Number of oscillatory dimensions
    pub oscillation_dimensions: usize,
    /// Hierarchy levels to process through
    pub hierarchy_levels: Vec<HierarchyLevel>,
    /// Metabolic mode for ATP management
    pub metabolic_mode: MetabolicMode,
    /// Enable quantum enhancement processing
    pub quantum_enhancement: bool,
    /// Enable ENAQT optimization
    pub enaqt_optimization: bool,
    /// Maximum processing time per query
    pub max_processing_time_ms: u64,
    /// ATP regeneration rate per second
    pub atp_regeneration_rate: f64,
}

impl Default for OscillatoryRAGConfig {
    fn default() -> Self {
        Self {
            temperature: 285.0, // Cold-blooded advantage
            target_entropy: 2.0,
            oscillation_dimensions: 8,
            hierarchy_levels: vec![
                HierarchyLevel::CellularOscillations,
                HierarchyLevel::OrganismalOscillations,
                HierarchyLevel::CognitiveOscillations,
            ],
            metabolic_mode: MetabolicMode::ColdBlooded {
                temperature_advantage: 1.4,
                metabolic_reduction: 0.7,
            },
            quantum_enhancement: true,
            enaqt_optimization: true,
            max_processing_time_ms: 30000,
            atp_regeneration_rate: 100.0,
        }
    }
}

/// Query input for oscillatory processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryQuery {
    /// The input query text
    pub content: String,
    /// Complexity measure (0.0 - 10.0)
    pub complexity: f64,
    /// Target oscillation frequency in Hz
    pub frequency: f64,
    /// Desired hierarchy levels for processing
    pub hierarchy_levels: Vec<HierarchyLevel>,
    /// Required biological layers
    pub biological_layers: Vec<BiologicalLayer>,
    /// Maximum acceptable processing cost
    pub max_cost: Option<f64>,
    /// Minimum required output quality
    pub min_quality: Option<f64>,
}

impl OscillatoryQuery {
    pub fn new(content: String) -> Self {
        let complexity = Self::estimate_complexity(&content);
        let frequency = Self::estimate_frequency(&content);
        
        Self {
            content,
            complexity,
            frequency,
            hierarchy_levels: vec![
                HierarchyLevel::CellularOscillations,
                HierarchyLevel::OrganismalOscillations,
                HierarchyLevel::CognitiveOscillations,
            ],
            biological_layers: vec![
                BiologicalLayer::Context,
                BiologicalLayer::Reasoning,
                BiologicalLayer::Intuition,
            ],
            max_cost: None,
            min_quality: Some(0.7),
        }
    }
    
    fn estimate_complexity(content: &str) -> f64 {
        let base_complexity = (content.len() as f64 / 100.0).min(10.0);
        
        // Adjust for question words and complexity indicators
        let complexity_indicators = [
            "why", "how", "explain", "analyze", "compare", "evaluate",
            "synthesize", "predict", "infer", "deduce", "quantum", "complex"
        ];
        
        let indicator_count = complexity_indicators.iter()
            .map(|&indicator| content.to_lowercase().matches(indicator).count())
            .sum::<usize>() as f64;
        
        (base_complexity + indicator_count * 0.5).min(10.0)
    }
    
    fn estimate_frequency(content: &str) -> f64 {
        let word_count = content.split_whitespace().count();
        match word_count {
            0..=10 => 1.0,      // High frequency for simple queries
            11..=50 => 0.5,     // Medium frequency for moderate queries
            51..=200 => 0.1,    // Low frequency for complex queries
            _ => 0.05,          // Very low frequency for extensive queries
        }
    }
}

/// Complete response from oscillatory processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryResponse {
    /// Generated response content
    pub content: String,
    /// Processing success status
    pub success: bool,
    /// Overall processing quality (0.0 - 1.0)
    pub quality: f64,
    /// Total ATP cost consumed
    pub atp_cost: f64,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Oscillatory processing results
    pub oscillation_results: Vec<OscillationProfile>,
    /// Quantum processing results
    pub quantum_results: HashMap<String, f64>,
    /// Hierarchy processing results
    pub hierarchy_results: Vec<HierarchyResult>,
    /// Biological layer processing results
    pub biological_results: Vec<BiologicalProcessingResult>,
    /// Final oscillation phase
    pub final_phase: OscillationPhase,
    /// Information content achieved
    pub information_content: f64,
    /// System state after processing
    pub system_state: SystemState,
}

/// Current state of the oscillatory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    pub atp_level: f64,
    pub quantum_coherence: f64,
    pub oscillation_synchrony: f64,
    pub hierarchy_coupling: f64,
    pub biological_efficiency: f64,
    pub metabolic_mode: MetabolicMode,
    pub active_oscillators: usize,
    pub processing_timestamp: DateTime<Utc>,
}

/// Complete Oscillatory Bio-Metabolic RAG System
#[derive(Debug)]
pub struct OscillatoryRAGSystem {
    /// System configuration
    config: OscillatoryRAGConfig,
    /// Universal oscillator for core dynamics
    oscillator: UniversalOscillator,
    /// Quantum membrane processor
    quantum_processor: QuantumMembraneProcessor,
    /// ENAQT processor for quantum transport optimization
    enaqt_processor: ENAQTProcessor,
    /// ATP manager for metabolic processing
    atp_manager: QuantumATPManager,
    /// Hierarchy processor for multi-scale processing
    hierarchy_processor: NestedHierarchyProcessor,
    /// Biological layer processor
    biological_processor: BiologicalLayerProcessor,
    /// Current system state
    system_state: SystemState,
    /// Processing statistics
    processing_stats: ProcessingStatistics,
}

#[derive(Debug, Clone, Default)]
struct ProcessingStatistics {
    total_queries: usize,
    successful_queries: usize,
    total_atp_consumed: f64,
    total_processing_time_ms: f64,
    average_quality: f64,
    quantum_enhancements: usize,
    hierarchy_emergences: usize,
}

impl OscillatoryRAGSystem {
    /// Initialize the complete oscillatory RAG system
    pub async fn new(config: OscillatoryRAGConfig) -> AutobahnResult<Self> {
        log::info!("🧬 Initializing Oscillatory Bio-Metabolic RAG System");
        log::info!("   Temperature: {:.1} K", config.temperature);
        log::info!("   Dimensions: {}", config.oscillation_dimensions);
        log::info!("   Hierarchy Levels: {:?}", config.hierarchy_levels);
        log::info!("   Metabolic Mode: {:?}", config.metabolic_mode);
        
        // Initialize core oscillator
        let oscillator = UniversalOscillator::new(
            config.oscillation_dimensions,
            1.0, // Base frequency
            0.1, // Damping coefficient
            );
        
        // Initialize quantum processor
        let quantum_processor = QuantumMembraneProcessor::new(config.temperature)?;
        
        // Initialize ENAQT processor
        let enaqt_processor = ENAQTProcessor::new(config.temperature)?;
        
        // Initialize ATP manager
        let atp_manager = QuantumATPManager::new(
            1000.0, // Initial ATP level
            config.atp_regeneration_rate,
            config.metabolic_mode.clone(),
        );
        
        // Initialize hierarchy processor
        let hierarchy_processor = NestedHierarchyProcessor::new();
        
        // Initialize biological processor
        let biological_processor = BiologicalLayerProcessor::new(config.temperature);
        
        // Initialize system state
        let system_state = SystemState {
            atp_level: 1000.0,
            quantum_coherence: 0.8,
            oscillation_synchrony: 0.9,
            hierarchy_coupling: 0.7,
            biological_efficiency: 0.8,
            metabolic_mode: config.metabolic_mode.clone(),
            active_oscillators: 1,
            processing_timestamp: Utc::now(),
        };
        
        Ok(Self {
            config,
            oscillator,
            quantum_processor,
            enaqt_processor,
            atp_manager,
            hierarchy_processor,
            biological_processor,
            system_state,
            processing_stats: ProcessingStatistics::default(),
        })
    }
    
    /// Process a query through the complete oscillatory system
    pub async fn process_query(&mut self, query: &str) -> AutobahnResult<OscillatoryResponse> {
        let start_time = std::time::Instant::now();
        let oscillatory_query = OscillatoryQuery::new(query.to_string());
        
        log::info!("🔄 Processing query: {} chars, complexity: {:.1}", 
                  query.len(), oscillatory_query.complexity);
        
        // Wrap processing in timeout
        let processing_result = timeout(
            Duration::from_millis(self.config.max_processing_time_ms),
            self.process_query_internal(oscillatory_query)
        ).await;
        
        match processing_result {
            Ok(result) => {
                let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;
                self.update_processing_stats(&result, processing_time);
                Ok(result?)
            },
            Err(_) => {
                log::warn!("⏱️ Query processing timeout after {}ms", self.config.max_processing_time_ms);
                Err(AutobahnError::ModelTimeout {
                    model_id: "oscillatory_rag".to_string(),
                    timeout_ms: self.config.max_processing_time_ms,
                })
            }
        }
    }
    
    /// Internal query processing implementation
    async fn process_query_internal(&mut self, query: OscillatoryQuery) -> AutobahnResult<OscillatoryResponse> {
        // Phase 1: ATP Cost Assessment and Allocation
        let estimated_cost = self.estimate_processing_cost(&query)?;
        self.atp_manager.reserve_atp(estimated_cost).await?;
        
        log::debug!("💰 Estimated ATP cost: {:.2}, available: {:.2}", 
                   estimated_cost, self.atp_manager.get_current_atp_level());
        
        // Phase 2: Oscillatory Dynamics Configuration
        self.configure_oscillator_for_query(&query)?;
        let mut oscillation_results = Vec::new();
        
        // Phase 3: Quantum Membrane Processing (if enabled)
        let mut quantum_results = HashMap::new();
        if self.config.quantum_enhancement {
            quantum_results = self.process_quantum_membrane(&query).await?;
            log::debug!("🔬 Quantum processing completed: {} results", quantum_results.len());
        }
        
        // Phase 4: ENAQT Optimization (if enabled)
        if self.config.enaqt_optimization && !quantum_results.is_empty() {
            let optimization_result = self.enaqt_processor.optimize_transport(&query.content, &quantum_results)?;
            quantum_results.insert("enaqt_efficiency".to_string(), optimization_result);
            log::debug!("⚡ ENAQT optimization: efficiency {:.2}", optimization_result);
        }
        
        // Phase 5: Multi-Scale Hierarchy Processing
        let mut hierarchy_results = Vec::new();
        for level in &query.hierarchy_levels {
            let hierarchy_result = self.hierarchy_processor.process_at_level(
                *level,
                &query.content,
                query.complexity,
            )?;
            hierarchy_results.push(hierarchy_result);
        }
        
        log::debug!("🏗️ Hierarchy processing: {} levels completed", hierarchy_results.len());
        
        // Phase 6: Biological Layer Processing
        let mut biological_results = Vec::new();
        self.biological_processor.update_metabolic_mode(self.config.metabolic_mode.clone());
        
        for layer in &query.biological_layers {
            let biological_result = self.biological_processor.process_at_layer(
                *layer,
                &query.content,
                query.complexity,
            )?;
            biological_results.push(biological_result);
        }
        
        log::debug!("🧠 Biological processing: {} layers completed", biological_results.len());
        
        // Phase 7: Oscillatory Integration and Response Generation
        let integration_result = self.integrate_processing_results(
            &query,
            &quantum_results,
            &hierarchy_results,
            &biological_results,
        )?;
        
        // Phase 8: Final Oscillation State and Response Assembly
        let final_profile = self.oscillator.get_current_profile();
        oscillation_results.push(final_profile.clone());
        
        let response_content = self.generate_response_content(&integration_result)?;
        let overall_quality = self.calculate_overall_quality(&integration_result)?;
        let actual_atp_cost = self.atp_manager.consume_reserved_atp().await?;
        
        // Update system state
        self.update_system_state(&hierarchy_results, &biological_results, actual_atp_cost);
        
        let response = OscillatoryResponse {
            content: response_content,
            success: overall_quality > query.min_quality.unwrap_or(0.0),
            quality: overall_quality,
            atp_cost: actual_atp_cost,
            processing_time_ms: 0.0, // Will be set by caller
            oscillation_results,
            quantum_results,
            hierarchy_results,
            biological_results,
            final_phase: final_profile.phase,
            information_content: final_profile.calculate_information_content(),
            system_state: self.system_state.clone(),
        };
        
        log::info!("✅ Query processed: quality={:.2}, cost={:.2} ATP", 
                  overall_quality, actual_atp_cost);
        
        Ok(response)
    }
    
    /// Estimate ATP cost for processing a query
    fn estimate_processing_cost(&self, query: &OscillatoryQuery) -> AutobahnResult<f64> {
        let base_cost = 10.0 * query.complexity;
        
        // Hierarchy processing cost
        let hierarchy_cost = query.hierarchy_levels.iter()
            .map(|level| match level {
                HierarchyLevel::QuantumOscillations => 100.0,
                HierarchyLevel::AtomicOscillations => 80.0,
                HierarchyLevel::MolecularOscillations => 60.0,
                HierarchyLevel::CellularOscillations => 40.0,
                HierarchyLevel::OrganismalOscillations => 30.0,
                HierarchyLevel::CognitiveOscillations => 50.0,
                HierarchyLevel::SocialOscillations => 35.0,
                HierarchyLevel::TechnologicalOscillations => 45.0,
                HierarchyLevel::CivilizationalOscillations => 70.0,
                HierarchyLevel::CosmicOscillations => 90.0,
            })
            .sum::<f64>();
        
        // Biological layer cost
        let biological_cost = query.biological_layers.iter()
            .map(|layer| layer.metabolic_cost_multiplier() * 15.0)
            .sum::<f64>();
        
        // Quantum enhancement cost
        let quantum_cost = if self.config.quantum_enhancement { 50.0 } else { 0.0 };
        
        // ENAQT optimization cost
        let enaqt_cost = if self.config.enaqt_optimization { 30.0 } else { 0.0 };
        
        // Apply metabolic mode modifier
        let total_base_cost = base_cost + hierarchy_cost + biological_cost + quantum_cost + enaqt_cost;
        let metabolic_modifier = match &self.config.metabolic_mode {
            MetabolicMode::SustainedFlight { efficiency_boost, .. } => 1.0 / efficiency_boost,
            MetabolicMode::ColdBlooded { metabolic_reduction, .. } => *metabolic_reduction,
            MetabolicMode::MammalianBurden { quantum_cost_multiplier, .. } => *quantum_cost_multiplier,
            MetabolicMode::AnaerobicEmergency { efficiency_penalty, .. } => 1.0 + efficiency_penalty,
        };
        
        Ok(total_base_cost * metabolic_modifier)
    }
    
    /// Configure oscillator for specific query characteristics
    fn configure_oscillator_for_query(&mut self, query: &OscillatoryQuery) -> AutobahnResult<()> {
        // Update frequency based on query requirements
        let target_frequency = query.frequency;
        self.oscillator.set_frequency(target_frequency);
        
        // Apply forcing function based on complexity
        let forcing_amplitude = query.complexity / 10.0;
        self.oscillator.apply_forcing_function(forcing_amplitude);
        
        // Set damping based on desired precision
        let damping = if query.complexity > 7.0 { 0.05 } else { 0.1 };
        self.oscillator.set_damping(damping);
        
        log::debug!("🎛️ Oscillator configured: freq={:.2} Hz, damping={:.3}, forcing={:.2}",
                   target_frequency, damping, forcing_amplitude);
        
        Ok(())
    }
    
    /// Process quantum membrane computations
    async fn process_quantum_membrane(&mut self, query: &OscillatoryQuery) -> AutobahnResult<HashMap<String, f64>> {
        let mut results = HashMap::new();
        
        // Calculate quantum tunneling probability
        let tunneling_prob = self.quantum_processor.calculate_tunneling_probability(
            query.complexity * 0.1, // Barrier height in eV
            1.0, // Particle energy
        )?;
        results.insert("tunneling_probability".to_string(), tunneling_prob);
        
        // Calculate transport efficiency
        let transport_efficiency = self.quantum_processor.calculate_transport_efficiency(
            query.complexity,
            self.config.temperature,
        )?;
        results.insert("transport_efficiency".to_string(), transport_efficiency);
        
        // Calculate quantum coherence time
        let coherence_time = self.quantum_processor.calculate_coherence_time(
            self.config.temperature,
            query.complexity,
        )?;
        results.insert("coherence_time_fs".to_string(), coherence_time);
        
        // Apply quantum enhancement to oscillation
        let quantum_enhancement = 1.0 + transport_efficiency * 0.3;
        self.oscillator.apply_quantum_enhancement(quantum_enhancement);
        
        log::debug!("🌌 Quantum membrane results: tunneling={:.4}, transport={:.3}, coherence={:.1}fs",
                   tunneling_prob, transport_efficiency, coherence_time);
        
        Ok(results)
    }
    
    /// Integration structure for processing results
    #[derive(Debug)]
    struct IntegrationResult {
        overall_coherence: f64,
        information_synthesis: f64,
        emergence_detected: bool,
        quantum_advantage: f64,
        biological_efficiency: f64,
        hierarchy_coupling: f64,
        response_components: Vec<String>,
    }
    
    /// Integrate all processing results
    fn integrate_processing_results(
        &self,
        query: &OscillatoryQuery,
        quantum_results: &HashMap<String, f64>,
        hierarchy_results: &[HierarchyResult],
        biological_results: &[BiologicalProcessingResult],
    ) -> AutobahnResult<IntegrationResult> {
        
        // Calculate overall coherence from all subsystems
        let quantum_coherence = quantum_results.get("transport_efficiency").unwrap_or(&0.0);
        let hierarchy_coherence = hierarchy_results.iter()
            .map(|r| r.coupling_strength)
            .sum::<f64>() / hierarchy_results.len() as f64;
        let biological_coherence = biological_results.iter()
            .map(|r| r.oscillation_coherence)
            .sum::<f64>() / biological_results.len() as f64;
        
        let overall_coherence = (quantum_coherence + hierarchy_coherence + biological_coherence) / 3.0;
        
        // Calculate information synthesis
        let hierarchy_info = hierarchy_results.iter()
            .map(|r| r.information_content)
            .sum::<f64>();
        let biological_info = biological_results.iter()
            .map(|r| r.information_content)
            .sum::<f64>();
        
        let information_synthesis = hierarchy_info + biological_info;
        
        // Detect emergence patterns
        let emergence_count = hierarchy_results.iter()
            .filter(|r| r.emergence_detected)
            .count();
        let emergence_detected = emergence_count >= 2;
        
        // Calculate quantum advantage
        let quantum_advantage = quantum_results.values().copied().sum::<f64>() / quantum_results.len() as f64;
        
        // Calculate biological efficiency
        let biological_efficiency = biological_results.iter()
            .map(|r| r.output_quality)
            .sum::<f64>() / biological_results.len() as f64;
        
        // Calculate hierarchy coupling strength
        let hierarchy_coupling = hierarchy_results.iter()
            .map(|r| r.coupling_strength)
            .sum::<f64>() / hierarchy_results.len() as f64;
        
        // Generate response components
        let mut response_components = Vec::new();
        
        // Add quantum insights
        if quantum_advantage > 0.5 {
            response_components.push(format!(
                "Quantum membrane processing indicates {} transport efficiency.",
                if quantum_advantage > 0.8 { "high" } else { "moderate" }
            ));
        }
        
        // Add hierarchy insights
        if emergence_detected {
            response_components.push(format!(
                "Multi-scale analysis reveals emergent patterns across {} hierarchy levels.",
                emergence_count
            ));
        }
        
        // Add biological insights
        if biological_efficiency > 0.7 {
            response_components.push(format!(
                "Biological layer processing achieved {:.0}% efficiency through {}-layer integration.",
                biological_efficiency * 100.0,
                biological_results.len()
            ));
        }
        
        // Add metabolic context
        response_components.push(format!(
            "Processing optimized for {:?} metabolic conditions.",
            self.config.metabolic_mode
        ));
        
        Ok(IntegrationResult {
            overall_coherence,
            information_synthesis,
            emergence_detected,
            quantum_advantage,
            biological_efficiency,
            hierarchy_coupling,
            response_components,
        })
    }
    
    /// Generate final response content
    fn generate_response_content(&self, integration: &IntegrationResult) -> AutobahnResult<String> {
        let mut response = String::new();
        
        // Add main response based on integration results
        if integration.emergence_detected {
            response.push_str("## Emergent Pattern Analysis\n\n");
            response.push_str("The multi-scale oscillatory analysis reveals emergent properties that arise from the complex interplay between quantum membrane dynamics, hierarchical organization, and biological processing layers.\n\n");
        }
        
        if integration.quantum_advantage > 0.6 {
            response.push_str("## Quantum-Enhanced Insights\n\n");
            response.push_str("Quantum membrane computation provides enhanced information transport efficiency, leveraging Environment-Assisted Quantum Transport (ENAQT) principles for optimal processing.\n\n");
        }
        
        response.push_str("## Integrated Analysis\n\n");
        
        // Add component insights
        for component in &integration.response_components {
            response.push_str(&format!("- {}\n", component));
        }
        
        response.push_str("\n## Processing Summary\n\n");
        response.push_str(&format!(
            "- Overall Coherence: {:.1}%\n",
            integration.overall_coherence * 100.0
        ));
        response.push_str(&format!(
            "- Information Synthesis: {:.2} bits\n",
            integration.information_synthesis
        ));
        response.push_str(&format!(
            "- Quantum Advantage: {:.1}%\n",
            integration.quantum_advantage * 100.0
        ));
        response.push_str(&format!(
            "- Biological Efficiency: {:.1}%\n",
            integration.biological_efficiency * 100.0
        ));
        response.push_str(&format!(
            "- Hierarchy Coupling: {:.1}%\n",
            integration.hierarchy_coupling * 100.0
        ));
        
        Ok(response)
    }
    
    /// Calculate overall processing quality
    fn calculate_overall_quality(&self, integration: &IntegrationResult) -> AutobahnResult<f64> {
        let base_quality = 0.5;
        
        let coherence_bonus = integration.overall_coherence * 0.3;
        let quantum_bonus = integration.quantum_advantage * 0.2;
        let biological_bonus = integration.biological_efficiency * 0.2;
        let emergence_bonus = if integration.emergence_detected { 0.1 } else { 0.0 };
        
        let total_quality = base_quality + coherence_bonus + quantum_bonus + biological_bonus + emergence_bonus;
        
        Ok(total_quality.min(1.0))
    }
    
    /// Update system state after processing
    fn update_system_state(
        &mut self,
        hierarchy_results: &[HierarchyResult],
        biological_results: &[BiologicalProcessingResult],
        atp_consumed: f64,
    ) {
        self.system_state.atp_level = self.atp_manager.get_current_atp_level();
        
        self.system_state.hierarchy_coupling = hierarchy_results.iter()
            .map(|r| r.coupling_strength)
            .sum::<f64>() / hierarchy_results.len() as f64;
        
        self.system_state.biological_efficiency = biological_results.iter()
            .map(|r| r.output_quality)
            .sum::<f64>() / biological_results.len() as f64;
        
        self.system_state.processing_timestamp = Utc::now();
        
        log::debug!("📊 System state updated: ATP={:.1}, coupling={:.2}, efficiency={:.2}",
                   self.system_state.atp_level,
                   self.system_state.hierarchy_coupling,
                   self.system_state.biological_efficiency);
    }
    
    /// Update processing statistics
    fn update_processing_stats(&mut self, response: &OscillatoryResponse, processing_time: f64) {
        self.processing_stats.total_queries += 1;
        
        if response.success {
            self.processing_stats.successful_queries += 1;
        }
        
        self.processing_stats.total_atp_consumed += response.atp_cost;
        self.processing_stats.total_processing_time_ms += processing_time;
        
        let current_avg = self.processing_stats.average_quality;
        let n = self.processing_stats.total_queries as f64;
        self.processing_stats.average_quality = (current_avg * (n - 1.0) + response.quality) / n;
        
        if !response.quantum_results.is_empty() {
            self.processing_stats.quantum_enhancements += 1;
        }
        
        let emergence_count = response.hierarchy_results.iter()
            .filter(|r| r.emergence_detected)
            .count();
        if emergence_count > 0 {
            self.processing_stats.hierarchy_emergences += 1;
        }
    }
    
    /// Get current system statistics
    pub fn get_processing_statistics(&self) -> ProcessingStatistics {
        self.processing_stats.clone()
    }
    
    /// Get current system state
    pub fn get_system_state(&self) -> SystemState {
        self.system_state.clone()
    }
    
    /// Update system configuration
    pub fn update_config(&mut self, new_config: OscillatoryRAGConfig) -> AutobahnResult<()> {
        log::info!("🔧 Updating system configuration");
        
        // Update metabolic mode if changed
        if new_config.metabolic_mode != self.config.metabolic_mode {
            self.atp_manager.update_metabolic_mode(new_config.metabolic_mode.clone())?;
            self.biological_processor.update_metabolic_mode(new_config.metabolic_mode.clone());
        }
        
        // Update temperature if changed
        if (new_config.temperature - self.config.temperature).abs() > 1.0 {
            // Note: Would need to reinitialize quantum processors for significant temperature changes
            log::warn!("Temperature change detected: {:.1}K -> {:.1}K", 
                      self.config.temperature, new_config.temperature);
        }
        
        self.config = new_config;
        self.system_state.metabolic_mode = self.config.metabolic_mode.clone();
        
        Ok(())
    }
} 