//! Context Processing Module - Glycolysis Layer
//!
//! This module implements context comprehension through glycolysis pathway simulation.
//! It handles context frames, semantic understanding, and biological energy conversion.

use crate::error::{AutobahnError, AutobahnResult};
use crate::traits::*;
use crate::types::*;
use crate::atp::MetabolicMode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Context processor for glycolysis-based comprehension
#[derive(Debug)]
pub struct ContextProcessor {
    /// Processor configuration
    config: ContextProcessorConfig,
    /// Current processing state
    state: ContextProcessorState,
    /// Glycolysis pathway simulator
    glycolysis_engine: GlycolysisEngine,
    /// Context memory system
    context_memory: ContextMemory,
    /// Membrane interface for ion transport
    membrane_interface: MembraneInterface,
}

/// Configuration for context processor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextProcessorConfig {
    /// Maximum context frames to maintain
    pub max_context_frames: usize,
    /// Context decay rate (per second)
    pub context_decay_rate: f64,
    /// Glucose processing rate (mol/s)
    pub glucose_processing_rate: f64,
    /// ATP efficiency factor
    pub atp_efficiency: f64,
    /// Membrane potential threshold
    pub membrane_threshold: f64,
    /// Quantum coherence time (ms)
    pub coherence_time_ms: u64,
}

impl Default for ContextProcessorConfig {
    fn default() -> Self {
        Self {
            max_context_frames: 50,
            context_decay_rate: 0.1,
            glucose_processing_rate: 2.5,
            atp_efficiency: 0.38, // ~38% efficiency in glycolysis
            membrane_threshold: -70.0,
            coherence_time_ms: 1000,
        }
    }
}

/// Current state of context processor
#[derive(Debug, Clone)]
pub struct ContextProcessorState {
    /// Context understanding level (0.0 to 1.0)
    pub context_understanding: f64,
    /// Context stability (0.0 to 1.0)
    pub context_stability: f64,
    /// Active context frames
    pub active_context_frames: Vec<ContextFrame>,
    /// Available ATP for processing
    pub available_atp: f64,
    /// Current metabolic mode
    pub metabolic_mode: MetabolicMode,
    /// Last context update timestamp
    pub last_context_update: DateTime<Utc>,
    /// Glycolysis pathway state
    pub glycolysis_state: GlycolysisState,
}

/// Context frame for semantic understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFrame {
    /// Unique frame identifier
    pub frame_id: String,
    /// Context content
    pub content: String,
    /// Relevance score (0.0 to 1.0)
    pub relevance: f64,
    /// Temporal validity (decay factor)
    pub temporal_validity: f64,
    /// Associated quantum states
    pub quantum_states: Vec<String>,
    /// Semantic connections
    pub semantic_connections: HashMap<String, f64>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last accessed timestamp
    pub last_accessed: DateTime<Utc>,
}

/// Glycolysis pathway state
#[derive(Debug, Clone)]
pub struct GlycolysisState {
    /// Glucose concentration
    pub glucose_concentration: f64,
    /// Pyruvate production rate
    pub pyruvate_production: f64,
    /// ATP production rate
    pub atp_production_rate: f64,
    /// NADH production rate
    pub nadh_production_rate: f64,
    /// Pathway efficiency
    pub pathway_efficiency: f64,
    /// Active enzymes
    pub active_enzymes: Vec<GlycolyticEnzyme>,
}

/// Glycolytic enzymes
#[derive(Debug, Clone)]
pub struct GlycolyticEnzyme {
    /// Enzyme name
    pub name: String,
    /// Activity level (0.0 to 1.0)
    pub activity: f64,
    /// Substrate affinity
    pub substrate_affinity: f64,
    /// Product inhibition factor
    pub product_inhibition: f64,
}

/// Glycolysis processing engine
#[derive(Debug)]
pub struct GlycolysisEngine {
    /// Engine configuration
    config: ContextProcessorConfig,
    /// Current pathway state
    pathway_state: GlycolysisState,
    /// Enzyme kinetics model
    enzyme_kinetics: EnzymeKineticsModel,
}

/// Enzyme kinetics modeling
#[derive(Debug)]
pub struct EnzymeKineticsModel {
    /// Michaelis-Menten constants
    pub km_values: HashMap<String, f64>,
    /// Maximum reaction velocities
    pub vmax_values: HashMap<String, f64>,
    /// Inhibition constants
    pub ki_values: HashMap<String, f64>,
}

/// Context memory system
#[derive(Debug)]
pub struct ContextMemory {
    /// Stored context frames
    frames: Vec<ContextFrame>,
    /// Frame index for quick lookup
    frame_index: HashMap<String, usize>,
    /// Semantic network connections
    semantic_network: HashMap<String, Vec<String>>,
    /// Memory consolidation state
    consolidation_state: MemoryConsolidationState,
}

/// Memory consolidation state
#[derive(Debug, Clone)]
pub struct MemoryConsolidationState {
    /// Consolidation progress (0.0 to 1.0)
    pub progress: f64,
    /// Active consolidation processes
    pub active_processes: Vec<String>,
    /// Energy allocated to consolidation
    pub energy_allocated: f64,
}

/// Membrane interface for ion transport
#[derive(Debug)]
pub struct MembraneInterface {
    /// Ion channel states
    pub ion_channels: HashMap<String, IonChannelState>,
    /// Membrane potential
    pub membrane_potential: f64,
    /// Quantum tunneling events
    pub tunneling_events: u32,
    /// Transport efficiency
    pub transport_efficiency: f64,
}

/// Ion channel state
#[derive(Debug, Clone)]
pub struct IonChannelState {
    /// Channel type
    pub channel_type: String,
    /// Open probability
    pub open_probability: f64,
    /// Conductance
    pub conductance: f64,
    /// Current flow
    pub current_flow: f64,
}

impl ContextProcessor {
    /// Create new context processor
    pub async fn new(config: ContextProcessorConfig) -> AutobahnResult<Self> {
        let glycolysis_engine = GlycolysisEngine::new(config.clone()).await?;
        let context_memory = ContextMemory::new();
        let membrane_interface = MembraneInterface::new();
        
        let state = ContextProcessorState {
            context_understanding: 0.0,
            context_stability: 0.5,
            active_context_frames: Vec::new(),
            available_atp: 1000.0,
            metabolic_mode: MetabolicMode::Normal,
            last_context_update: Utc::now(),
            glycolysis_state: GlycolysisState::default(),
        };
        
        Ok(Self {
            config,
            state,
            glycolysis_engine,
            context_memory,
            membrane_interface,
        })
    }
    
    /// Process context information through glycolysis pathway
    pub async fn process_context(&mut self, content: &str) -> AutobahnResult<ContextProcessingResult> {
        let start_time = std::time::Instant::now();
        
        // Convert content to glucose equivalent
        let glucose_input = self.convert_content_to_glucose(content)?;
        
        // Process through glycolysis pathway
        let glycolysis_result = self.glycolysis_engine.process_glucose(glucose_input).await?;
        
        // Extract context frames
        let context_frames = self.extract_context_frames(content)?;
        
        // Update context memory
        self.context_memory.update_contexts(context_frames.clone()).await?;
        
        // Generate quantum states
        let quantum_states = self.generate_quantum_states(&context_frames)?;
        
        // Update membrane activity
        let membrane_activity = self.simulate_membrane_activity(content).await?;
        
        // Update processor state
        self.update_state(&glycolysis_result, &context_frames).await?;
        
        let processing_time = start_time.elapsed();
        
        Ok(ContextProcessingResult {
            context_frames,
            glycolysis_result,
            quantum_states,
            membrane_activity,
            context_understanding: self.state.context_understanding,
            atp_consumed: glycolysis_result.atp_consumed,
            processing_time_ms: processing_time.as_millis() as u64,
            semantic_connections: self.extract_semantic_connections(content)?,
        })
    }
    
    /// Convert text content to glucose processing units
    fn convert_content_to_glucose(&self, content: &str) -> AutobahnResult<f64> {
        // Simple heuristic: content complexity -> glucose requirement
        let word_count = content.split_whitespace().count() as f64;
        let complexity_factor = self.calculate_content_complexity(content);
        
        Ok(word_count * complexity_factor * 0.1) // Glucose units
    }
    
    /// Calculate content complexity
    fn calculate_content_complexity(&self, content: &str) -> f64 {
        let unique_words = content.split_whitespace()
            .collect::<std::collections::HashSet<_>>()
            .len() as f64;
        let total_words = content.split_whitespace().count() as f64;
        
        if total_words > 0.0 {
            unique_words / total_words
        } else {
            0.0
        }
    }
    
    /// Extract context frames from content
    fn extract_context_frames(&self, content: &str) -> AutobahnResult<Vec<ContextFrame>> {
        let sentences: Vec<&str> = content.split('.').collect();
        let mut frames = Vec::new();
        
        for (i, sentence) in sentences.iter().enumerate() {
            if sentence.trim().is_empty() {
                continue;
            }
            
            let frame = ContextFrame {
                frame_id: format!("frame_{}", i),
                content: sentence.trim().to_string(),
                relevance: self.calculate_relevance(sentence),
                temporal_validity: 1.0,
                quantum_states: vec![format!("quantum_state_{}", i)],
                semantic_connections: HashMap::new(),
                created_at: Utc::now(),
                last_accessed: Utc::now(),
            };
            
            frames.push(frame);
        }
        
        Ok(frames)
    }
    
    /// Calculate relevance score for content
    fn calculate_relevance(&self, content: &str) -> f64 {
        // Simple relevance calculation based on content length and complexity
        let word_count = content.split_whitespace().count() as f64;
        let complexity = self.calculate_content_complexity(content);
        
        (word_count.ln() * complexity).min(1.0).max(0.0)
    }
    
    /// Generate quantum states for context frames
    fn generate_quantum_states(&self, frames: &[ContextFrame]) -> AutobahnResult<Vec<QuantumState>> {
        let mut quantum_states = Vec::new();
        
        for frame in frames {
            let quantum_state = QuantumState {
                state_id: format!("quantum_{}", frame.frame_id),
                coherence_level: frame.relevance * 0.8,
                entanglement_strength: frame.temporal_validity,
                phase: 0.0,
                energy_level: frame.relevance * 100.0,
            };
            
            quantum_states.push(quantum_state);
        }
        
        Ok(quantum_states)
    }
    
    /// Simulate membrane activity during processing
    async fn simulate_membrane_activity(&mut self, _content: &str) -> AutobahnResult<MembraneActivity> {
        // Simulate ion channel activity
        let mut ion_activities = HashMap::new();
        ion_activities.insert("H+".to_string(), 0.8);
        ion_activities.insert("Na+".to_string(), 0.6);
        ion_activities.insert("K+".to_string(), 0.7);
        
        // Update membrane potential
        self.membrane_interface.membrane_potential += 5.0;
        self.membrane_interface.tunneling_events += 1;
        
        Ok(MembraneActivity {
            ion_channel_activities: ion_activities,
            membrane_potential_change: 5.0,
            tunneling_events: 1,
            transport_efficiency: 0.85,
        })
    }
    
    /// Extract semantic connections
    fn extract_semantic_connections(&self, content: &str) -> AutobahnResult<HashMap<String, f64>> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let mut connections = HashMap::new();
        
        // Simple co-occurrence based connections
        for i in 0..words.len() {
            for j in (i+1)..words.len() {
                if j - i <= 5 { // Within 5-word window
                    let key = format!("{}_{}", words[i], words[j]);
                    let strength = 1.0 / (j - i) as f64;
                    connections.insert(key, strength);
                }
            }
        }
        
        Ok(connections)
    }
    
    /// Update processor state after processing
    async fn update_state(&mut self, glycolysis_result: &GlycolysisResult, context_frames: &[ContextFrame]) -> AutobahnResult<()> {
        // Update context understanding
        let frame_quality = context_frames.iter()
            .map(|f| f.relevance)
            .sum::<f64>() / context_frames.len().max(1) as f64;
        
        self.state.context_understanding = (self.state.context_understanding * 0.7 + frame_quality * 0.3).min(1.0);
        
        // Update context stability
        self.state.context_stability = glycolysis_result.pathway_efficiency;
        
        // Update active frames
        self.state.active_context_frames = context_frames.to_vec();
        
        // Update ATP
        self.state.available_atp -= glycolysis_result.atp_consumed;
        
        // Update timestamp
        self.state.last_context_update = Utc::now();
        
        Ok(())
    }
    
    /// Get current processor state
    pub fn get_state(&self) -> &ContextProcessorState {
        &self.state
    }
    
    /// Get processor metrics
    pub fn get_metrics(&self) -> ContextProcessorMetrics {
        ContextProcessorMetrics {
            context_understanding: self.state.context_understanding,
            context_stability: self.state.context_stability,
            active_frames_count: self.state.active_context_frames.len(),
            available_atp: self.state.available_atp,
            glycolysis_efficiency: self.state.glycolysis_state.pathway_efficiency,
            membrane_potential: self.membrane_interface.membrane_potential,
        }
    }
}

impl GlycolysisEngine {
    /// Create new glycolysis engine
    async fn new(config: ContextProcessorConfig) -> AutobahnResult<Self> {
        let pathway_state = GlycolysisState::default();
        let enzyme_kinetics = EnzymeKineticsModel::default();
        
        Ok(Self {
            config,
            pathway_state,
            enzyme_kinetics,
        })
    }
    
    /// Process glucose through glycolysis pathway
    async fn process_glucose(&mut self, glucose_amount: f64) -> AutobahnResult<GlycolysisResult> {
        // Simulate glycolysis steps
        let atp_produced = glucose_amount * 2.0 * self.config.atp_efficiency; // Net 2 ATP per glucose
        let pyruvate_produced = glucose_amount * 2.0; // 2 pyruvate per glucose
        let nadh_produced = glucose_amount * 2.0; // 2 NADH per glucose
        
        // Calculate pathway efficiency
        let efficiency = self.calculate_pathway_efficiency(glucose_amount);
        
        // Update pathway state
        self.pathway_state.glucose_concentration -= glucose_amount;
        self.pathway_state.pyruvate_production += pyruvate_produced;
        self.pathway_state.atp_production_rate = atp_produced / 0.1; // per 100ms
        self.pathway_state.pathway_efficiency = efficiency;
        
        Ok(GlycolysisResult {
            atp_consumed: atp_produced * 0.1, // Small consumption for processing
            atp_produced,
            pyruvate_produced,
            nadh_produced,
            pathway_efficiency: efficiency,
            glucose_consumed: glucose_amount,
        })
    }
    
    /// Calculate pathway efficiency
    fn calculate_pathway_efficiency(&self, glucose_amount: f64) -> f64 {
        // Simple efficiency model based on substrate concentration
        let optimal_concentration = 5.0;
        let efficiency = 1.0 - ((glucose_amount - optimal_concentration).abs() / optimal_concentration).min(0.5);
        efficiency.max(0.1)
    }
}

impl ContextMemory {
    /// Create new context memory
    fn new() -> Self {
        Self {
            frames: Vec::new(),
            frame_index: HashMap::new(),
            semantic_network: HashMap::new(),
            consolidation_state: MemoryConsolidationState {
                progress: 0.0,
                active_processes: Vec::new(),
                energy_allocated: 0.0,
            },
        }
    }
    
    /// Update context frames
    async fn update_contexts(&mut self, new_frames: Vec<ContextFrame>) -> AutobahnResult<()> {
        for frame in new_frames {
            self.frames.push(frame.clone());
            self.frame_index.insert(frame.frame_id.clone(), self.frames.len() - 1);
            
            // Update semantic network
            self.update_semantic_network(&frame).await?;
        }
        
        // Prune old frames if necessary
        self.prune_old_frames().await?;
        
        Ok(())
    }
    
    /// Update semantic network connections
    async fn update_semantic_network(&mut self, frame: &ContextFrame) -> AutobahnResult<()> {
        let words: Vec<String> = frame.content
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        
        for word in words {
            let connections = self.semantic_network.entry(word.clone()).or_insert_with(Vec::new);
            
            // Add frame ID to word connections
            if !connections.contains(&frame.frame_id) {
                connections.push(frame.frame_id.clone());
            }
        }
        
        Ok(())
    }
    
    /// Prune old or irrelevant frames
    async fn prune_old_frames(&mut self) -> AutobahnResult<()> {
        let current_time = Utc::now();
        let max_age = chrono::Duration::hours(24);
        
        self.frames.retain(|frame| {
            current_time.signed_duration_since(frame.created_at) < max_age
                && frame.relevance > 0.1
        });
        
        // Rebuild index
        self.frame_index.clear();
        for (i, frame) in self.frames.iter().enumerate() {
            self.frame_index.insert(frame.frame_id.clone(), i);
        }
        
        Ok(())
    }
}

impl MembraneInterface {
    /// Create new membrane interface
    fn new() -> Self {
        let mut ion_channels = HashMap::new();
        
        // Initialize ion channels
        ion_channels.insert("H+".to_string(), IonChannelState {
            channel_type: "H+".to_string(),
            open_probability: 0.1,
            conductance: 1.0,
            current_flow: 0.0,
        });
        
        ion_channels.insert("Na+".to_string(), IonChannelState {
            channel_type: "Na+".to_string(),
            open_probability: 0.05,
            conductance: 0.8,
            current_flow: 0.0,
        });
        
        Self {
            ion_channels,
            membrane_potential: -70.0,
            tunneling_events: 0,
            transport_efficiency: 0.8,
        }
    }
}

// Default implementations
impl Default for GlycolysisState {
    fn default() -> Self {
        Self {
            glucose_concentration: 10.0,
            pyruvate_production: 0.0,
            atp_production_rate: 0.0,
            nadh_production_rate: 0.0,
            pathway_efficiency: 0.8,
            active_enzymes: vec![
                GlycolyticEnzyme {
                    name: "Hexokinase".to_string(),
                    activity: 0.8,
                    substrate_affinity: 0.9,
                    product_inhibition: 0.1,
                },
                GlycolyticEnzyme {
                    name: "Phosphofructokinase".to_string(),
                    activity: 0.7,
                    substrate_affinity: 0.8,
                    product_inhibition: 0.2,
                },
                GlycolyticEnzyme {
                    name: "Pyruvate kinase".to_string(),
                    activity: 0.9,
                    substrate_affinity: 0.85,
                    product_inhibition: 0.05,
                },
            ],
        }
    }
}

impl Default for EnzymeKineticsModel {
    fn default() -> Self {
        let mut km_values = HashMap::new();
        let mut vmax_values = HashMap::new();
        let mut ki_values = HashMap::new();
        
        // Typical values for glycolytic enzymes
        km_values.insert("Hexokinase".to_string(), 0.1);
        km_values.insert("Phosphofructokinase".to_string(), 0.5);
        km_values.insert("Pyruvate kinase".to_string(), 0.3);
        
        vmax_values.insert("Hexokinase".to_string(), 10.0);
        vmax_values.insert("Phosphofructokinase".to_string(), 15.0);
        vmax_values.insert("Pyruvate kinase".to_string(), 20.0);
        
        ki_values.insert("Hexokinase".to_string(), 1.0);
        ki_values.insert("Phosphofructokinase".to_string(), 2.0);
        ki_values.insert("Pyruvate kinase".to_string(), 0.5);
        
        Self {
            km_values,
            vmax_values,
            ki_values,
        }
    }
}

// Result structures
#[derive(Debug, Clone)]
pub struct ContextProcessingResult {
    pub context_frames: Vec<ContextFrame>,
    pub glycolysis_result: GlycolysisResult,
    pub quantum_states: Vec<QuantumState>,
    pub membrane_activity: MembraneActivity,
    pub context_understanding: f64,
    pub atp_consumed: f64,
    pub processing_time_ms: u64,
    pub semantic_connections: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct GlycolysisResult {
    pub atp_consumed: f64,
    pub atp_produced: f64,
    pub pyruvate_produced: f64,
    pub nadh_produced: f64,
    pub pathway_efficiency: f64,
    pub glucose_consumed: f64,
}

#[derive(Debug, Clone)]
pub struct QuantumState {
    pub state_id: String,
    pub coherence_level: f64,
    pub entanglement_strength: f64,
    pub phase: f64,
    pub energy_level: f64,
}

#[derive(Debug, Clone)]
pub struct MembraneActivity {
    pub ion_channel_activities: HashMap<String, f64>,
    pub membrane_potential_change: f64,
    pub tunneling_events: u32,
    pub transport_efficiency: f64,
}

#[derive(Debug, Clone)]
pub struct ContextProcessorMetrics {
    pub context_understanding: f64,
    pub context_stability: f64,
    pub active_frames_count: usize,
    pub available_atp: f64,
    pub glycolysis_efficiency: f64,
    pub membrane_potential: f64,
}
