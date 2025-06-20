//! Tres Commas Trinity Engine - Three Consciousness Layers
//!
//! This module implements the three-layer consciousness system:
//! - Context Layer (Glycolysis)
//! - Reasoning Layer (Krebs Cycle) 
//! - Intuition Layer (Electron Transport)

use crate::traits::{TresCommasLayer, MetacognitiveOrchestrator};
use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};

/// Trinity Engine implementing three consciousness layers
pub struct TrinityEngine {
    /// Current active layer
    current_layer: TresCommasLayer,
    /// Layer processing states
    layer_states: std::collections::HashMap<TresCommasLayer, LayerState>,
    /// Inter-layer communication buffer
    communication_buffer: Vec<LayerMessage>,
}

/// State of a processing layer
#[derive(Debug, Clone)]
pub struct LayerState {
    pub active: bool,
    pub processing_load: f64,
    pub efficiency: f64,
    pub last_activity: chrono::DateTime<chrono::Utc>,
}

/// Message between layers
#[derive(Debug, Clone)]
pub struct LayerMessage {
    pub from_layer: TresCommasLayer,
    pub to_layer: TresCommasLayer,
    pub content: String,
    pub priority: MessagePriority,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Message priority levels
#[derive(Debug, Clone)]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

impl TrinityEngine {
    /// Create new Trinity Engine
    pub fn new() -> Self {
        let mut layer_states = std::collections::HashMap::new();
        let now = chrono::Utc::now();
        
        layer_states.insert(TresCommasLayer::Context, LayerState {
            active: true,
            processing_load: 0.0,
            efficiency: 1.0,
            last_activity: now,
        });
        
        layer_states.insert(TresCommasLayer::Reasoning, LayerState {
            active: true,
            processing_load: 0.0,
            efficiency: 1.0,
            last_activity: now,
        });
        
        layer_states.insert(TresCommasLayer::Intuition, LayerState {
            active: true,
            processing_load: 0.0,
            efficiency: 1.0,
            last_activity: now,
        });
        
        Self {
            current_layer: TresCommasLayer::Context,
            layer_states,
            communication_buffer: Vec::new(),
        }
    }
    
    /// Switch to specific layer
    pub fn switch_to_layer(&mut self, layer: TresCommasLayer) -> AutobahnResult<()> {
        if let Some(state) = self.layer_states.get(&layer) {
            if !state.active {
                return Err(AutobahnError::ProcessingError {
                    layer: format!("{:?}", layer),
                    reason: "Layer is not active".to_string(),
                });
            }
        }
        
        self.current_layer = layer;
        Ok(())
    }
    
    /// Get current layer
    pub fn current_layer(&self) -> &TresCommasLayer {
        &self.current_layer
    }
    
    /// Send message between layers
    pub fn send_layer_message(&mut self, from: TresCommasLayer, to: TresCommasLayer, content: String, priority: MessagePriority) {
        let message = LayerMessage {
            from_layer: from,
            to_layer: to,
            content,
            priority,
            timestamp: chrono::Utc::now(),
        };
        
        self.communication_buffer.push(message);
    }
    
    /// Process layer messages
    pub fn process_messages(&mut self) -> Vec<LayerMessage> {
        let messages = self.communication_buffer.clone();
        self.communication_buffer.clear();
        messages
    }
}

impl Default for TrinityEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trinity_engine_creation() {
        let engine = TrinityEngine::new();
        assert_eq!(*engine.current_layer(), TresCommasLayer::Context);
        assert_eq!(engine.layer_states.len(), 3);
    }

    #[test]
    fn test_layer_switching() {
        let mut engine = TrinityEngine::new();
        
        assert!(engine.switch_to_layer(TresCommasLayer::Reasoning).is_ok());
        assert_eq!(*engine.current_layer(), TresCommasLayer::Reasoning);
        
        assert!(engine.switch_to_layer(TresCommasLayer::Intuition).is_ok());
        assert_eq!(*engine.current_layer(), TresCommasLayer::Intuition);
    }

    #[test]
    fn test_layer_messaging() {
        let mut engine = TrinityEngine::new();
        
        engine.send_layer_message(
            TresCommasLayer::Context,
            TresCommasLayer::Reasoning,
            "Test message".to_string(),
            MessagePriority::Normal,
        );
        
        let messages = engine.process_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "Test message");
    }
}

pub mod engine;

// Re-export the main components
pub use engine::{
    ConsciousComputationalEngine,
    CategoricalPredeterminismEngine,
    ConfigurationSpaceExplorer,
    HeatDeathTrajectoryCalculator,
    CategoricalCompletionTracker,
    ConsciousInput,
    ConsciousOutput,
    PredeterminismAnalysis,
    ConfigurationSpacePosition,
    TrajectoryAnalysis,
    CompletionAnalysis,
    Context,
    ContextualizedInput,
}; 