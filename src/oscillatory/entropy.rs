//! Oscillatory Entropy Implementation
//! 
//! This module implements the revolutionary Oscillatory Entropy Theorem:
//! Entropy represents the statistical distribution of oscillation termination points,
//! making entropy directly measurable and controllable through oscillatory dynamics.

use crate::error::{AutobahnError, AutobahnResult};
use crate::oscillatory::{OscillationState, OscillationPhase, UniversalOscillator};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Oscillation termination point with associated probability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminationPoint {
    /// Position where oscillation terminated
    pub position: DVector<f64>,
    /// Velocity at termination
    pub velocity: DVector<f64>,
    /// Phase at termination
    pub phase: f64,
    /// Energy at termination
    pub energy: f64,
    /// Probability weight of this termination
    pub probability: f64,
    /// Timestamp of termination
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl TerminationPoint {
    /// Calculate information content of this termination point
    pub fn information_content(&self) -> f64 {
        if self.probability > 0.0 {
            -self.probability.log2()
        } else {
            f64::INFINITY
        }
    }
    
    /// Calculate entropy contribution
    pub fn entropy_contribution(&self) -> f64 {
        if self.probability > 0.0 {
            -self.probability * self.probability.log2()
        } else {
            0.0
        }
    }
    
    /// Get termination signature for classification
    pub fn termination_signature(&self) -> String {
        let pos_norm = self.position.norm();
        let vel_norm = self.velocity.norm();
        let energy_class = if self.energy > 10.0 { "high" } 
                          else if self.energy > 1.0 { "medium" } 
                          else { "low" };
        
        format!("pos:{:.2}_vel:{:.2}_phase:{:.2}_energy:{}", 
                pos_norm, vel_norm, self.phase, energy_class)
    }
}

/// Oscillatory entropy analyzer implementing the theorem
#[derive(Debug, Clone)]
pub struct OscillatoryEntropyAnalyzer {
    /// Collection of observed termination points
    termination_points: Vec<TerminationPoint>,
    /// Statistical distribution of termination endpoints
    endpoint_distribution: HashMap<String, f64>,
    /// Phase space discretization resolution
    phase_space_resolution: f64,
    /// Maximum number of termination points to track
    max_termination_points: usize,
    /// Entropy calculation cache
    entropy_cache: Option<f64>,
    /// Cache validity timestamp
    cache_timestamp: chrono::DateTime<chrono::Utc>,
}

impl OscillatoryEntropyAnalyzer {
    /// Create new entropy analyzer
    pub fn new(phase_space_resolution: f64) -> Self {
        Self {
            termination_points: Vec::new(),
            endpoint_distribution: HashMap::new(),
            phase_space_resolution,
            max_termination_points: 10000,
            entropy_cache: None,
            cache_timestamp: chrono::Utc::now(),
        }
    }
    
    /// Record oscillation termination point
    pub fn record_termination(&mut self, oscillator: &UniversalOscillator, reason: &str) -> AutobahnResult<()> {
        let termination_point = TerminationPoint {
            position: oscillator.state.position.clone(),
            velocity: oscillator.state.velocity.clone(),
            phase: oscillator.state.phase,
            energy: oscillator.total_energy(),
            probability: 0.0, // Will be calculated later
            timestamp: chrono::Utc::now(),
        };
        
        // Add to collection
        self.termination_points.push(termination_point);
        
        // Limit collection size
        if self.termination_points.len() > self.max_termination_points {
            self.termination_points.drain(..1000);
        }
        
        // Update endpoint distribution
        self.update_endpoint_distribution()?;
        
        // Invalidate entropy cache
        self.entropy_cache = None;
        
        log::debug!("Recorded termination: {} points total", self.termination_points.len());
        
        Ok(())
    }
    
    /// Update statistical distribution of termination endpoints
    fn update_endpoint_distribution(&mut self) -> AutobahnResult<()> {
        self.endpoint_distribution.clear();
        
        if self.termination_points.is_empty() {
            return Ok(());
        }
        
        // Count occurrences of each termination signature
        let mut signature_counts: HashMap<String, usize> = HashMap::new();
        for point in &self.termination_points {
            let signature = point.termination_signature();
            *signature_counts.entry(signature).or_insert(0) += 1;
        }
        
        // Convert counts to probabilities
        let total_points = self.termination_points.len() as f64;
        for (signature, count) in signature_counts {
            let probability = count as f64 / total_points;
            self.endpoint_distribution.insert(signature, probability);
        }
        
        // Update probabilities in termination points
        for point in &mut self.termination_points {
            let signature = point.termination_signature();
            if let Some(&prob) = self.endpoint_distribution.get(&signature) {
                point.probability = prob;
            }
        }
        
        Ok(())
    }
    
    /// Calculate oscillatory entropy using Shannon formula: H = -Σ p_i log₂(p_i)
    pub fn calculate_entropy(&mut self) -> AutobahnResult<f64> {
        // Check cache validity (5 seconds)
        if let Some(cached_entropy) = self.entropy_cache {
            let cache_age = chrono::Utc::now()
                .signed_duration_since(self.cache_timestamp)
                .num_seconds();
            if cache_age < 5 {
                return Ok(cached_entropy);
            }
        }
        
        if self.endpoint_distribution.is_empty() {
            return Ok(0.0);
        }
        
        let mut entropy = 0.0;
        let mut total_probability = 0.0;
        
        for probability in self.endpoint_distribution.values() {
            total_probability += probability;
            if *probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }
        
        // Normalize if probabilities don't sum to 1
        if (total_probability - 1.0).abs() > 1e-10 {
            entropy /= total_probability;
            log::warn!("Probability distribution not normalized: sum = {:.6}", total_probability);
        }
        
        // Cache result
        self.entropy_cache = Some(entropy);
        self.cache_timestamp = chrono::Utc::now();
        
        Ok(entropy)
    }
    
    /// Calculate differential entropy in continuous phase space
    pub fn calculate_differential_entropy(&self) -> AutobahnResult<f64> {
        if self.termination_points.len() < 10 {
            return Ok(0.0);
        }
        
        // Estimate probability density using kernel density estimation
        let mut differential_entropy = 0.0;
        let bandwidth = self.estimate_optimal_bandwidth();
        
        for point in &self.termination_points {
            let density = self.estimate_density_at_point(point, bandwidth)?;
            if density > 1e-10 {
                differential_entropy -= density.ln() / (self.termination_points.len() as f64).ln();
            }
        }
        
        Ok(differential_entropy)
    }
    
    /// Estimate probability density at a given point using Gaussian kernels
    fn estimate_density_at_point(&self, target: &TerminationPoint, bandwidth: f64) -> AutobahnResult<f64> {
        let mut density = 0.0;
        let normalization = 1.0 / (self.termination_points.len() as f64);
        let kernel_norm = 1.0 / (bandwidth * (2.0 * PI).sqrt());
        
        for point in &self.termination_points {
            // Calculate distance in phase space
            let pos_dist = (&target.position - &point.position).norm();
            let vel_dist = (&target.velocity - &point.velocity).norm();
            let phase_dist = (target.phase - point.phase).abs();
            let energy_dist = (target.energy - point.energy).abs();
            
            // Combined distance metric
            let distance = (pos_dist.powi(2) + vel_dist.powi(2) + phase_dist.powi(2) + energy_dist.powi(2)).sqrt();
            
            // Gaussian kernel
            let kernel_value = kernel_norm * (-0.5 * (distance / bandwidth).powi(2)).exp();
            density += kernel_value;
        }
        
        Ok(density * normalization)
    }
    
    /// Estimate optimal bandwidth for kernel density estimation
    fn estimate_optimal_bandwidth(&self) -> f64 {
        if self.termination_points.is_empty() {
            return 1.0;
        }
        
        // Silverman's rule of thumb for bandwidth selection
        let n = self.termination_points.len() as f64;
        let dimension = if !self.termination_points.is_empty() {
            self.termination_points[0].position.len() * 2 + 2 // position + velocity + phase + energy
        } else {
            4
        } as f64;
        
        // Calculate standard deviation of data
        let mut sum_squared_distances = 0.0;
        let count = self.termination_points.len();
        
        for i in 0..count {
            for j in i+1..count {
                let point_i = &self.termination_points[i];
                let point_j = &self.termination_points[j];
                
                let pos_dist = (&point_i.position - &point_j.position).norm();
                let vel_dist = (&point_i.velocity - &point_j.velocity).norm();
                let phase_dist = (point_i.phase - point_j.phase).abs();
                let energy_dist = (point_i.energy - point_j.energy).abs();
                
                let distance = (pos_dist.powi(2) + vel_dist.powi(2) + phase_dist.powi(2) + energy_dist.powi(2)).sqrt();
                sum_squared_distances += distance.powi(2);
            }
        }
        
        let variance = if count > 1 {
            sum_squared_distances / ((count * (count - 1) / 2) as f64)
        } else {
            1.0
        };
        
        let std_dev = variance.sqrt();
        
        // Silverman's bandwidth
        let bandwidth = std_dev * (4.0 / ((dimension + 2.0) * n)).powf(1.0 / (dimension + 4.0));
        
        bandwidth.max(0.01) // Minimum bandwidth
    }
    
    /// Get most probable termination endpoints
    pub fn get_most_probable_endpoints(&self, top_n: usize) -> Vec<(String, f64)> {
        let mut endpoints: Vec<_> = self.endpoint_distribution.iter()
            .map(|(sig, &prob)| (sig.clone(), prob))
            .collect();
        
        endpoints.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        endpoints.truncate(top_n);
        endpoints
    }
    
    /// Calculate entropy production rate
    pub fn calculate_entropy_production_rate(&self, time_window_seconds: f64) -> AutobahnResult<f64> {
        let now = chrono::Utc::now();
        let cutoff_time = now - chrono::Duration::seconds(time_window_seconds as i64);
        
        let recent_points: Vec<_> = self.termination_points.iter()
            .filter(|point| point.timestamp > cutoff_time)
            .collect();
        
        if recent_points.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate entropy of recent terminations
        let mut recent_distribution: HashMap<String, f64> = HashMap::new();
        for point in &recent_points {
            let signature = point.termination_signature();
            *recent_distribution.entry(signature).or_insert(0.0) += 1.0;
        }
        
        // Normalize
        let total = recent_points.len() as f64;
        for prob in recent_distribution.values_mut() {
            *prob /= total;
        }
        
        // Calculate entropy
        let mut recent_entropy = 0.0;
        for &prob in recent_distribution.values() {
            if prob > 0.0 {
                recent_entropy -= prob * prob.log2();
            }
        }
        
        // Production rate = entropy / time
        Ok(recent_entropy / time_window_seconds)
    }
    
    /// Predict future entropy based on current trends
    pub fn predict_entropy_evolution(&self, prediction_time_seconds: f64) -> AutobahnResult<f64> {
        if self.termination_points.len() < 5 {
            return Ok(0.0);
        }
        
        // Calculate entropy production rate over last minute
        let production_rate = self.calculate_entropy_production_rate(60.0)?;
        
        // Get current entropy
        let mut analyzer_copy = self.clone();
        let current_entropy = analyzer_copy.calculate_entropy()?;
        
        // Linear prediction (could be made more sophisticated)
        let predicted_entropy = current_entropy + production_rate * prediction_time_seconds;
        
        Ok(predicted_entropy.max(0.0))
    }
    
    /// Control entropy by biasing oscillation parameters
    pub fn suggest_entropy_control(&self, target_entropy: f64) -> AutobahnResult<EntropyControlSuggestion> {
        let mut analyzer_copy = self.clone();
        let current_entropy = analyzer_copy.calculate_entropy()?;
        
        let entropy_difference = target_entropy - current_entropy;
        
        let suggestion = if entropy_difference > 0.1 {
            // Need to increase entropy - more chaotic oscillations
            EntropyControlSuggestion {
                action: EntropyControlAction::IncreaseEntropy,
                damping_adjustment: -0.1, // Reduce damping for more chaos
                forcing_adjustment: 0.2,  // Increase forcing for more variation
                frequency_adjustment: 0.0,
                confidence: 0.8,
                explanation: "Increase oscillation chaos to raise entropy".to_string(),
            }
        } else if entropy_difference < -0.1 {
            // Need to decrease entropy - more ordered oscillations
            EntropyControlSuggestion {
                action: EntropyControlAction::DecreaseEntropy,
                damping_adjustment: 0.1,  // Increase damping for more order
                forcing_adjustment: -0.1, // Reduce forcing for less variation
                frequency_adjustment: 0.0,
                confidence: 0.8,
                explanation: "Increase oscillation order to reduce entropy".to_string(),
            }
        } else {
            // Entropy is close to target
            EntropyControlSuggestion {
                action: EntropyControlAction::MaintainEntropy,
                damping_adjustment: 0.0,
                forcing_adjustment: 0.0,
                frequency_adjustment: 0.0,
                confidence: 0.9,
                explanation: "Current entropy level is optimal".to_string(),
            }
        };
        
        Ok(suggestion)
    }
    
    /// Get entropy analysis summary
    pub fn get_analysis_summary(&mut self) -> AutobahnResult<EntropyAnalysisSummary> {
        let entropy = self.calculate_entropy()?;
        let differential_entropy = self.calculate_differential_entropy()?;
        let production_rate = self.calculate_entropy_production_rate(60.0)?;
        let top_endpoints = self.get_most_probable_endpoints(5);
        
        Ok(EntropyAnalysisSummary {
            total_termination_points: self.termination_points.len(),
            shannon_entropy: entropy,
            differential_entropy,
            entropy_production_rate: production_rate,
            endpoint_distribution_size: self.endpoint_distribution.len(),
            most_probable_endpoints: top_endpoints,
            analysis_timestamp: chrono::Utc::now(),
        })
    }
}

/// Entropy control action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntropyControlAction {
    IncreaseEntropy,
    DecreaseEntropy,
    MaintainEntropy,
}

/// Entropy control suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyControlSuggestion {
    pub action: EntropyControlAction,
    pub damping_adjustment: f64,
    pub forcing_adjustment: f64,
    pub frequency_adjustment: f64,
    pub confidence: f64,
    pub explanation: String,
}

/// Summary of entropy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyAnalysisSummary {
    pub total_termination_points: usize,
    pub shannon_entropy: f64,
    pub differential_entropy: f64,
    pub entropy_production_rate: f64,
    pub endpoint_distribution_size: usize,
    pub most_probable_endpoints: Vec<(String, f64)>,
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
}

// Re-export key types
pub use {
    TerminationPoint,
    OscillatoryEntropyAnalyzer,
    EntropyControlAction,
    EntropyControlSuggestion,
    EntropyAnalysisSummary,
}; 