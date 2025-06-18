//! Champagne Phase - Dream Processing and Lactate Recovery
//!
//! This module implements the dreaming mode for background processing,
//! lactate recovery, and system optimization during user inactivity.

use crate::traits::{ChampagneProcessor, UserStatus, DreamMode, DreamInitialization, ChampagneResult};
use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use async_trait::async_trait;

/// Champagne phase processor for dream processing
pub struct ChampagnePhaseProcessor {
    /// Current dream state
    dream_state: Option<DreamState>,
    /// Lactate buffer for processing
    lactate_buffer: Vec<LactateEntry>,
    /// Processing configuration
    config: ChampagneConfig,
    /// Available for champagne processing
    available: bool,
}

/// Configuration for champagne phase
#[derive(Debug, Clone)]
pub struct ChampagneConfig {
    /// Minimum user idle time before champagne phase
    pub min_idle_time_ms: u64,
    /// Maximum dream duration
    pub max_dream_duration_ms: u64,
    /// Lactate processing rate
    pub lactate_processing_rate: f64,
    /// Enable automatic dream scheduling
    pub auto_schedule_dreams: bool,
}

/// Current dream processing state
#[derive(Debug, Clone)]
pub struct DreamState {
    pub mode: DreamMode,
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub estimated_duration: u64,
    pub progress: f64,
    pub insights_generated: Vec<String>,
}

/// Lactate entry for processing
#[derive(Debug, Clone)]
pub struct LactateEntry {
    pub content: String,
    pub accumulation_time: chrono::DateTime<chrono::Utc>,
    pub processing_attempts: u32,
    pub priority: LactatePriority,
}

/// Priority levels for lactate processing
#[derive(Debug, Clone)]
pub enum LactatePriority {
    Low,
    Medium,
    High,
    Critical,
}

impl ChampagnePhaseProcessor {
    /// Create new champagne phase processor
    pub fn new() -> Self {
        Self {
            dream_state: None,
            lactate_buffer: Vec::new(),
            config: ChampagneConfig::default(),
            available: true,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ChampagneConfig) -> Self {
        Self {
            dream_state: None,
            lactate_buffer: Vec::new(),
            config,
            available: true,
        }
    }

    /// Add lactate entry for processing
    pub fn add_lactate_entry(&mut self, content: String, priority: LactatePriority) {
        let entry = LactateEntry {
            content,
            accumulation_time: chrono::Utc::now(),
            processing_attempts: 0,
            priority,
        };
        
        self.lactate_buffer.push(entry);
    }

    /// Get lactate buffer size
    pub fn lactate_buffer_size(&self) -> usize {
        self.lactate_buffer.len()
    }

    /// Check if currently dreaming
    pub fn is_dreaming(&self) -> bool {
        self.dream_state.is_some()
    }

    /// Get current dream progress
    pub fn dream_progress(&self) -> Option<f64> {
        self.dream_state.as_ref().map(|state| state.progress)
    }
}

#[async_trait]
impl ChampagneProcessor for ChampagnePhaseProcessor {
    async fn enter_dream_state(&mut self, user_status: UserStatus) -> AutobahnResult<DreamInitialization> {
        if !self.is_champagne_available(user_status.clone()) {
            return Err(AutobahnError::ChampagneUnavailableError {
                user_status: format!("{:?}", user_status),
            });
        }

        if self.is_dreaming() {
            return Err(AutobahnError::ProcessingError {
                layer: "champagne".to_string(),
                reason: "Already in dream state".to_string(),
            });
        }

        // Determine dream mode based on lactate buffer and user status
        let dream_mode = match user_status {
            UserStatus::Sleeping => DreamMode::Deep,
            UserStatus::Away => DreamMode::REM,
            UserStatus::Idle => DreamMode::Light,
            UserStatus::Active => DreamMode::Recovery,
        };

        let estimated_duration = match dream_mode {
            DreamMode::Deep => self.config.max_dream_duration_ms,
            DreamMode::REM => self.config.max_dream_duration_ms / 2,
            DreamMode::Light => self.config.max_dream_duration_ms / 4,
            DreamMode::Recovery => self.config.max_dream_duration_ms / 8,
        };

        let atp_allocated = match dream_mode {
            DreamMode::Deep => 500.0,
            DreamMode::REM => 300.0,
            DreamMode::Light => 150.0,
            DreamMode::Recovery => 75.0,
        };

        // Initialize dream state
        self.dream_state = Some(DreamState {
            mode: dream_mode.clone(),
            start_time: chrono::Utc::now(),
            estimated_duration,
            progress: 0.0,
            insights_generated: Vec::new(),
        });

        Ok(DreamInitialization {
            success: true,
            dream_mode,
            atp_allocated,
            estimated_duration_ms: estimated_duration,
        })
    }

    async fn process_lactate_buffer(&mut self) -> AutobahnResult<ChampagneResult> {
        if !self.is_dreaming() {
            return Err(AutobahnError::ProcessingError {
                layer: "champagne".to_string(),
                reason: "Not in dream state".to_string(),
            });
        }

        let start_time = std::time::Instant::now();
        let lactate_count = self.lactate_buffer.len();

        // Process lactate entries
        let mut insights_gained = Vec::new();
        let mut processed_count = 0;

        for entry in &mut self.lactate_buffer {
            entry.processing_attempts += 1;
            
            // Simulate processing insight generation
            match entry.priority {
                LactatePriority::Critical => {
                    insights_gained.push(format!("Critical insight from: {}", entry.content));
                }
                LactatePriority::High => {
                    insights_gained.push(format!("High-priority insight from: {}", entry.content));
                }
                _ => {
                    if processed_count % 3 == 0 {
                        insights_gained.push(format!("General insight from processing"));
                    }
                }
            }
            
            processed_count += 1;
        }

        // Clear processed lactate
        self.lactate_buffer.clear();

        let processing_time = start_time.elapsed().as_millis() as u64;
        let lactate_processed = lactate_count as f64;

        // Update dream state progress
        if let Some(ref mut dream_state) = self.dream_state {
            dream_state.progress = 1.0; // Mark as complete
            dream_state.insights_generated.extend(insights_gained.clone());
        }

        Ok(ChampagneResult {
            lactate_processed,
            insights_gained,
            optimization_improvements: lactate_processed * 0.1, // 10% improvement per processed item
            dream_duration_ms: processing_time,
        })
    }

    async fn complete_comprehension_processing(&mut self, partial: crate::traits::PartialComprehension) -> AutobahnResult<crate::traits::CompletedInsight> {
        if !self.is_dreaming() {
            return Err(AutobahnError::ProcessingError {
                layer: "champagne".to_string(),
                reason: "Not in dream state for comprehension completion".to_string(),
            });
        }

        // Simulate completion processing
        let completion_method = if partial.processing_attempts > 3 {
            crate::traits::CompletionMethod::AlternativeStrategy
        } else if partial.comprehension_score < 0.5 {
            crate::traits::CompletionMethod::DeepProcessing
        } else {
            crate::traits::CompletionMethod::CrossDomainAnalysis
        };

        let final_score = (partial.comprehension_score + 0.2).min(1.0);
        
        let insights_gained = vec![
            "Comprehension gaps identified and filled".to_string(),
            "Alternative interpretation pathways explored".to_string(),
            "Context dependencies resolved".to_string(),
        ];

        let patterns_discovered = vec![
            "Recurring semantic patterns".to_string(),
            "Syntactic structures".to_string(),
        ];

        Ok(crate::traits::CompletedInsight {
            original_partial: partial,
            completion_method,
            final_comprehension_score: final_score,
            insights_gained,
            patterns_discovered,
        })
    }

    async fn auto_correct_scripts(&mut self, scripts: Vec<String>) -> AutobahnResult<Vec<String>> {
        if !self.is_dreaming() {
            return Err(AutobahnError::ProcessingError {
                layer: "champagne".to_string(),
                reason: "Not in dream state for script correction".to_string(),
            });
        }

        // Simulate script auto-correction
        let corrected_scripts = scripts
            .into_iter()
            .map(|script| {
                // Simple corrections - in real implementation would be more sophisticated
                script
                    .replace("turbulance", "turbulence")
                    .replace("occurence", "occurrence")
                    .replace("recieve", "receive")
                    .replace("seperate", "separate")
            })
            .collect();

        Ok(corrected_scripts)
    }

    fn is_champagne_available(&self, user_status: UserStatus) -> bool {
        if !self.available {
            return false;
        }

        // Champagne phase is available when user is not actively working
        matches!(user_status, UserStatus::Idle | UserStatus::Away | UserStatus::Sleeping)
    }
}

impl Default for ChampagneConfig {
    fn default() -> Self {
        Self {
            min_idle_time_ms: 300_000, // 5 minutes
            max_dream_duration_ms: 3_600_000, // 1 hour
            lactate_processing_rate: 10.0, // 10 items per second
            auto_schedule_dreams: true,
        }
    }
}

impl Default for ChampagnePhaseProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_champagne_processor_creation() {
        let processor = ChampagnePhaseProcessor::new();
        assert!(!processor.is_dreaming());
        assert_eq!(processor.lactate_buffer_size(), 0);
    }

    #[tokio::test]
    async fn test_lactate_entry_addition() {
        let mut processor = ChampagnePhaseProcessor::new();
        
        processor.add_lactate_entry("Test content".to_string(), LactatePriority::Medium);
        assert_eq!(processor.lactate_buffer_size(), 1);
    }

    #[tokio::test]
    async fn test_dream_state_entry() {
        let mut processor = ChampagnePhaseProcessor::new();
        
        let result = processor.enter_dream_state(UserStatus::Idle).await;
        assert!(result.is_ok());
        assert!(processor.is_dreaming());
        
        let dream_init = result.unwrap();
        assert!(dream_init.success);
        assert!(dream_init.atp_allocated > 0.0);
    }

    #[tokio::test]
    async fn test_champagne_availability() {
        let processor = ChampagnePhaseProcessor::new();
        
        assert!(processor.is_champagne_available(UserStatus::Idle));
        assert!(processor.is_champagne_available(UserStatus::Away));
        assert!(processor.is_champagne_available(UserStatus::Sleeping));
        assert!(!processor.is_champagne_available(UserStatus::Active));
    }

    #[tokio::test]
    async fn test_lactate_processing() {
        let mut processor = ChampagnePhaseProcessor::new();
        
        // Add some lactate entries
        processor.add_lactate_entry("Entry 1".to_string(), LactatePriority::High);
        processor.add_lactate_entry("Entry 2".to_string(), LactatePriority::Medium);
        
        // Enter dream state
        let _ = processor.enter_dream_state(UserStatus::Idle).await.unwrap();
        
        // Process lactate buffer
        let result = processor.process_lactate_buffer().await;
        assert!(result.is_ok());
        
        let champagne_result = result.unwrap();
        assert_eq!(champagne_result.lactate_processed, 2.0);
        assert!(!champagne_result.insights_gained.is_empty());
        assert_eq!(processor.lactate_buffer_size(), 0); // Buffer should be cleared
    }
} 