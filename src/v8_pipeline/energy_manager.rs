//! ATP Energy Manager - Biological Energy Management System
//!
//! This module manages ATP (Adenosine Triphosphate) allocation and regeneration
//! for the biological metabolism computer.

use crate::traits::EnergyManager;
use crate::types::EnergyState;
use crate::error::{AutobahnError, AutobahnResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// ATP Manager for biological energy management
#[derive(Debug, Clone)]
pub struct ATPManager {
    /// Current energy state
    energy_state: EnergyState,
    /// Energy consumption history
    consumption_history: Vec<EnergyConsumption>,
    /// Regeneration configuration
    regeneration_config: RegenerationConfig,
    /// Last regeneration time
    last_regeneration: DateTime<Utc>,
    /// Energy efficiency metrics
    efficiency_metrics: EfficiencyMetrics,
}

/// Configuration for ATP regeneration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegenerationConfig {
    /// Base regeneration rate per second
    pub base_rate: f64,
    /// Maximum regeneration burst
    pub max_burst: f64,
    /// Efficiency factor (0.0 - 1.0)
    pub efficiency_factor: f64,
    /// Recovery time after high consumption
    pub recovery_time_ms: u64,
}

/// Energy consumption record
#[derive(Debug, Clone)]
pub struct EnergyConsumption {
    /// Amount consumed
    pub amount: f64,
    /// Operation that consumed energy
    pub operation: String,
    /// Timestamp of consumption
    pub timestamp: DateTime<Utc>,
    /// Module that consumed energy
    pub module: Option<String>,
}

/// Energy efficiency metrics
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// Total energy consumed
    pub total_consumed: f64,
    /// Total operations performed
    pub total_operations: u64,
    /// Average energy per operation
    pub avg_energy_per_operation: f64,
    /// Peak consumption rate
    pub peak_consumption_rate: f64,
    /// Current efficiency score (0.0 - 1.0)
    pub efficiency_score: f64,
}

impl ATPManager {
    /// Create new ATP manager
    pub fn new(max_atp: f64) -> Self {
        Self {
            energy_state: EnergyState::new(max_atp),
            consumption_history: Vec::new(),
            regeneration_config: RegenerationConfig::default(),
            last_regeneration: Utc::now(),
            efficiency_metrics: EfficiencyMetrics::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(max_atp: f64, config: RegenerationConfig) -> Self {
        Self {
            energy_state: EnergyState::new(max_atp),
            consumption_history: Vec::new(),
            regeneration_config: config,
            last_regeneration: Utc::now(),
            efficiency_metrics: EfficiencyMetrics::default(),
        }
    }

    /// Allocate ATP for specific operation
    pub fn allocate_atp(&mut self, amount: f64, operation: String, module: Option<String>) -> AutobahnResult<()> {
        if !self.can_afford(amount) {
            return Err(AutobahnError::InsufficientATP {
                required: amount,
                available: self.energy_state.current_atp,
            });
        }

        // Consume ATP
        let actual_consumed = self.consume_atp(amount)?;

        // Record consumption
        let consumption = EnergyConsumption {
            amount: actual_consumed,
            operation,
            timestamp: Utc::now(),
            module,
        };
        self.consumption_history.push(consumption);

        // Update efficiency metrics
        self.update_efficiency_metrics();

        Ok(())
    }

    /// Get energy consumption by module
    pub fn get_consumption_by_module(&self) -> HashMap<String, f64> {
        let mut consumption_by_module = HashMap::new();
        
        for consumption in &self.consumption_history {
            if let Some(module) = &consumption.module {
                *consumption_by_module.entry(module.clone()).or_insert(0.0) += consumption.amount;
            }
        }
        
        consumption_by_module
    }

    /// Get recent consumption rate
    pub fn get_recent_consumption_rate(&self, window_ms: u64) -> f64 {
        let now = Utc::now();
        let window_start = now - chrono::Duration::milliseconds(window_ms as i64);
        
        let recent_consumption: f64 = self.consumption_history
            .iter()
            .filter(|c| c.timestamp >= window_start)
            .map(|c| c.amount)
            .sum();
        
        recent_consumption / (window_ms as f64 / 1000.0) // Per second
    }

    /// Predict energy needs
    pub fn predict_energy_needs(&self, operation: &str, complexity: f64) -> f64 {
        // Base energy requirements by operation type
        let base_energy = match operation {
            "glycolysis" => 20.0,
            "krebs_cycle" => 30.0,
            "electron_transport" => 50.0,
            "pattern_recognition" => 15.0,
            "adversarial_testing" => 40.0,
            "bayesian_inference" => 25.0,
            "comprehension_validation" => 35.0,
            "noise_reduction" => 10.0,
            _ => 20.0, // Default
        };
        
        // Scale by complexity
        let complexity_factor = 1.0 + (complexity * 0.5);
        
        // Apply efficiency factor
        let efficiency_factor = self.efficiency_metrics.efficiency_score.max(0.1);
        
        base_energy * complexity_factor / efficiency_factor
    }

    /// Update efficiency metrics
    fn update_efficiency_metrics(&mut self) {
        let total_consumed: f64 = self.consumption_history.iter().map(|c| c.amount).sum();
        let total_operations = self.consumption_history.len() as u64;
        
        self.efficiency_metrics.total_consumed = total_consumed;
        self.efficiency_metrics.total_operations = total_operations;
        
        if total_operations > 0 {
            self.efficiency_metrics.avg_energy_per_operation = total_consumed / total_operations as f64;
        }
        
        // Calculate efficiency score based on consumption patterns
        let recent_rate = self.get_recent_consumption_rate(60000); // Last minute
        let optimal_rate = self.regeneration_config.base_rate * 0.8; // 80% of regeneration rate
        
        self.efficiency_metrics.efficiency_score = if recent_rate > 0.0 {
            (optimal_rate / recent_rate).min(1.0)
        } else {
            1.0
        };
        
        // Update peak consumption rate
        if recent_rate > self.efficiency_metrics.peak_consumption_rate {
            self.efficiency_metrics.peak_consumption_rate = recent_rate;
        }
    }

    /// Get energy state summary
    pub fn get_energy_summary(&self) -> EnergySummary {
        EnergySummary {
            current_atp: self.energy_state.current_atp,
            max_atp: self.energy_state.max_atp,
            utilization_percent: (self.energy_state.current_atp / self.energy_state.max_atp) * 100.0,
            recent_consumption_rate: self.get_recent_consumption_rate(60000),
            efficiency_score: self.efficiency_metrics.efficiency_score,
            total_operations: self.efficiency_metrics.total_operations,
        }
    }
}

/// Energy state summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergySummary {
    pub current_atp: f64,
    pub max_atp: f64,
    pub utilization_percent: f64,
    pub recent_consumption_rate: f64,
    pub efficiency_score: f64,
    pub total_operations: u64,
}

impl EnergyManager for ATPManager {
    fn get_energy_state(&self) -> &EnergyState {
        &self.energy_state
    }

    fn consume_atp(&mut self, amount: f64) -> AutobahnResult<f64> {
        if amount <= 0.0 {
            return Ok(0.0);
        }

        if self.energy_state.current_atp < amount {
            return Err(AutobahnError::InsufficientATP {
                required: amount,
                available: self.energy_state.current_atp,
            });
        }

        self.energy_state.current_atp -= amount;
        self.energy_state.total_consumed += amount;
        Ok(amount)
    }

    fn regenerate_atp(&mut self, time_elapsed_ms: u64) {
        let time_elapsed_s = time_elapsed_ms as f64 / 1000.0;
        let regeneration_amount = self.regeneration_config.base_rate * time_elapsed_s * self.regeneration_config.efficiency_factor;
        
        let new_atp = (self.energy_state.current_atp + regeneration_amount).min(self.energy_state.max_atp);
        self.energy_state.current_atp = new_atp;
        self.last_regeneration = Utc::now();
    }

    fn can_afford(&self, cost: f64) -> bool {
        self.energy_state.current_atp >= cost
    }

    fn optimize_energy_usage(&mut self) {
        // Implement energy optimization strategies
        
        // 1. Clear old consumption history
        let cutoff_time = Utc::now() - chrono::Duration::hours(1);
        self.consumption_history.retain(|c| c.timestamp >= cutoff_time);
        
        // 2. Adjust regeneration efficiency based on usage patterns
        let recent_rate = self.get_recent_consumption_rate(300000); // Last 5 minutes
        if recent_rate > self.regeneration_config.base_rate * 1.2 {
            // High consumption - increase efficiency
            self.regeneration_config.efficiency_factor = (self.regeneration_config.efficiency_factor * 1.1).min(1.0);
        } else if recent_rate < self.regeneration_config.base_rate * 0.3 {
            // Low consumption - can reduce efficiency slightly
            self.regeneration_config.efficiency_factor = (self.regeneration_config.efficiency_factor * 0.95).max(0.5);
        }
        
        // 3. Update efficiency metrics
        self.update_efficiency_metrics();
    }

    fn get_efficiency(&self) -> f64 {
        self.efficiency_metrics.efficiency_score
    }
}

impl Default for RegenerationConfig {
    fn default() -> Self {
        Self {
            base_rate: 10.0,      // 10 ATP per second
            max_burst: 100.0,     // Maximum burst regeneration
            efficiency_factor: 0.8, // 80% efficiency
            recovery_time_ms: 5000, // 5 second recovery
        }
    }
}

impl Default for EfficiencyMetrics {
    fn default() -> Self {
        Self {
            total_consumed: 0.0,
            total_operations: 0,
            avg_energy_per_operation: 0.0,
            peak_consumption_rate: 0.0,
            efficiency_score: 1.0,
        }
    }
} 