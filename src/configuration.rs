//! Configuration Management System for Autobahn
//!
//! This module provides comprehensive configuration management including
//! system settings, module configurations, performance tuning, and
//! environment-specific configurations.

use crate::error::{AutobahnError, AutobahnResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Master configuration for the Autobahn system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobahnConfig {
    /// System-wide settings
    pub system: SystemConfig,
    /// V8 pipeline configuration
    pub v8_pipeline: V8PipelineConfig,
    /// Probabilistic engine configuration
    pub probabilistic: ProbabilisticConfig,
    /// Temporal processor configuration
    pub temporal: TemporalConfig,
    /// Research laboratory configuration
    pub research: ResearchConfig,
    /// Plugin system configuration
    pub plugins: PluginConfig,
    /// Benchmarking configuration
    pub benchmarking: BenchmarkConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Performance configuration
    pub performance: PerformanceConfig,
}

/// System-wide configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// System name
    pub name: String,
    /// System version
    pub version: String,
    /// Environment (development, staging, production)
    pub environment: Environment,
    /// Data directory
    pub data_directory: String,
    /// Configuration directory
    pub config_directory: String,
    /// Log directory
    pub log_directory: String,
    /// Maximum ATP capacity
    pub max_atp_capacity: f64,
    /// ATP regeneration rate
    pub atp_regeneration_rate: f64,
    /// Enable debug mode
    pub debug_mode: bool,
    /// Thread pool size
    pub thread_pool_size: usize,
}

/// Environment types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Testing,
    Staging,
    Production,
}

/// V8 Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V8PipelineConfig {
    /// Enable V8 pipeline
    pub enabled: bool,
    /// Module configurations
    pub modules: HashMap<String, ModuleConfig>,
    /// ATP allocation strategy
    pub atp_allocation_strategy: ATPAllocationStrategy,
    /// Processing timeout (ms)
    pub processing_timeout_ms: u64,
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Maximum concurrent processes
    pub max_concurrent_processes: usize,
}

/// Individual module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleConfig {
    /// Module enabled
    pub enabled: bool,
    /// Module priority
    pub priority: u32,
    /// ATP budget
    pub atp_budget: f64,
    /// Processing timeout (ms)
    pub timeout_ms: u64,
    /// Module-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// ATP allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ATPAllocationStrategy {
    /// Equal allocation to all modules
    Equal,
    /// Priority-based allocation
    PriorityBased,
    /// Dynamic allocation based on load
    Dynamic,
    /// Custom allocation
    Custom(HashMap<String, f64>),
}

/// Probabilistic engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticConfig {
    /// Enable probabilistic processing
    pub enabled: bool,
    /// Default inference algorithm
    pub default_inference_algorithm: String,
    /// Monte Carlo settings
    pub monte_carlo: MonteCarloConfig,
    /// Bayesian network settings
    pub bayesian_networks: BayesianNetworkConfig,
    /// Uncertainty quantification settings
    pub uncertainty_quantification: UncertaintyConfig,
}

/// Monte Carlo configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    /// Default number of samples
    pub default_samples: u32,
    /// Burn-in period
    pub burn_in: u32,
    /// Random seed
    pub random_seed: Option<u64>,
}

/// Bayesian network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianNetworkConfig {
    /// Maximum network size
    pub max_network_size: usize,
    /// Learning algorithm
    pub learning_algorithm: String,
    /// Validation threshold
    pub validation_threshold: f64,
}

/// Uncertainty quantification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyConfig {
    /// Enable uncertainty propagation
    pub enable_propagation: bool,
    /// Confidence level
    pub confidence_level: f64,
    /// Sensitivity analysis enabled
    pub enable_sensitivity_analysis: bool,
}

/// Temporal processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Enable temporal processing
    pub enabled: bool,
    /// Evidence decay function
    pub decay_function: String,
    /// Decay rate
    pub decay_rate: f64,
    /// Pattern detection threshold
    pub pattern_detection_threshold: f64,
    /// Historical data retention (days)
    pub historical_retention_days: u32,
}

/// Research laboratory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchConfig {
    /// Enable research features
    pub enabled: bool,
    /// Enable quantum-inspired processing
    pub enable_quantum: bool,
    /// Enable machine learning integration
    pub enable_ml: bool,
    /// Enable experimental pathways
    pub enable_experimental: bool,
    /// Safety constraints
    pub safety_constraints: SafetyConstraints,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

/// Safety constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConstraints {
    /// Maximum experimental runtime (ms)
    pub max_runtime_ms: u64,
    /// Memory usage limits (MB)
    pub max_memory_mb: usize,
    /// CPU usage limits (%)
    pub max_cpu_percent: f64,
}

/// Resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum concurrent experiments
    pub max_concurrent_experiments: usize,
    /// Maximum data storage (GB)
    pub max_storage_gb: f64,
    /// Network bandwidth limits (Mbps)
    pub max_bandwidth_mbps: f64,
}

/// Plugin system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Enable plugin system
    pub enabled: bool,
    /// Plugin directory
    pub plugin_directory: String,
    /// Maximum number of plugins
    pub max_plugins: usize,
    /// Default security level
    pub default_security_level: String,
    /// Plugin loading timeout (ms)
    pub loading_timeout_ms: u64,
}

/// Benchmarking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Enable benchmarking
    pub enabled: bool,
    /// Benchmark timeout (ms)
    pub timeout_ms: u64,
    /// Iterations per benchmark
    pub iterations: u32,
    /// Warmup iterations
    pub warmup_iterations: u32,
    /// Enable performance benchmarks
    pub enable_performance: bool,
    /// Enable ATP efficiency tests
    pub enable_atp_tests: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    /// Log format
    pub format: String,
    /// Log to file
    pub log_to_file: bool,
    /// Log file path
    pub log_file_path: String,
    /// Maximum log file size (MB)
    pub max_file_size_mb: f64,
    /// Log rotation count
    pub rotation_count: u32,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Monitoring interval (ms)
    pub monitoring_interval_ms: u64,
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
    /// Enable auto-optimization
    pub enable_auto_optimization: bool,
}

/// Performance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum processing time (ms)
    pub max_processing_time_ms: u64,
    /// Maximum memory usage (MB)
    pub max_memory_mb: f64,
    /// Maximum ATP consumption per operation
    pub max_atp_per_operation: f64,
    /// Minimum confidence score
    pub min_confidence: f64,
}

/// Configuration manager
#[derive(Debug)]
pub struct ConfigurationManager {
    /// Current configuration
    config: AutobahnConfig,
    /// Configuration file path
    config_file_path: String,
    /// Configuration watchers
    watchers: Vec<Box<dyn ConfigurationWatcher>>,
}

/// Configuration watcher trait
pub trait ConfigurationWatcher: Send + Sync {
    /// Called when configuration changes
    fn on_configuration_changed(&mut self, old_config: &AutobahnConfig, new_config: &AutobahnConfig);
}

impl ConfigurationManager {
    /// Create new configuration manager
    pub fn new() -> Self {
        Self {
            config: AutobahnConfig::default(),
            config_file_path: "autobahn.toml".to_string(),
            watchers: Vec::new(),
        }
    }

    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> AutobahnResult<()> {
        let path = path.as_ref();
        self.config_file_path = path.to_string_lossy().to_string();
        
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .map_err(|e| AutobahnError::InitializationError(format!("Failed to read config file: {}", e)))?;
            
            self.config = toml::from_str(&content)
                .map_err(|e| AutobahnError::InitializationError(format!("Failed to parse config: {}", e)))?;
        }
        
        Ok(())
    }

    /// Save configuration to file
    pub fn save_to_file(&self) -> AutobahnResult<()> {
        let content = toml::to_string_pretty(&self.config)
            .map_err(|e| AutobahnError::InitializationError(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(&self.config_file_path, content)
            .map_err(|e| AutobahnError::InitializationError(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }

    /// Get current configuration
    pub fn get_config(&self) -> &AutobahnConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: AutobahnConfig) -> AutobahnResult<()> {
        let old_config = self.config.clone();
        
        // Validate new configuration
        self.validate_config(&new_config)?;
        
        self.config = new_config;
        
        // Notify watchers
        for watcher in &mut self.watchers {
            watcher.on_configuration_changed(&old_config, &self.config);
        }
        
        Ok(())
    }

    /// Add configuration watcher
    pub fn add_watcher(&mut self, watcher: Box<dyn ConfigurationWatcher>) {
        self.watchers.push(watcher);
    }

    /// Validate configuration
    fn validate_config(&self, config: &AutobahnConfig) -> AutobahnResult<()> {
        // Validate system config
        if config.system.max_atp_capacity <= 0.0 {
            return Err(AutobahnError::InitializationError(
                "ATP capacity must be positive".to_string()
            ));
        }

        if config.system.thread_pool_size == 0 {
            return Err(AutobahnError::InitializationError(
                "Thread pool size must be positive".to_string()
            ));
        }

        // Validate performance thresholds
        if config.performance.thresholds.min_confidence < 0.0 || config.performance.thresholds.min_confidence > 1.0 {
            return Err(AutobahnError::InitializationError(
                "Confidence threshold must be between 0.0 and 1.0".to_string()
            ));
        }

        Ok(())
    }

    /// Get module configuration
    pub fn get_module_config(&self, module_name: &str) -> Option<&ModuleConfig> {
        self.config.v8_pipeline.modules.get(module_name)
    }

    /// Update module configuration
    pub fn update_module_config(&mut self, module_name: String, config: ModuleConfig) -> AutobahnResult<()> {
        let old_config = self.config.clone();
        
        self.config.v8_pipeline.modules.insert(module_name, config);
        
        // Notify watchers
        for watcher in &mut self.watchers {
            watcher.on_configuration_changed(&old_config, &self.config);
        }
        
        Ok(())
    }

    /// Create configuration from environment
    pub fn from_environment() -> AutobahnResult<AutobahnConfig> {
        let mut config = AutobahnConfig::default();
        
        // Override with environment variables
        if let Ok(env) = std::env::var("AUTOBAHN_ENVIRONMENT") {
            config.system.environment = match env.as_str() {
                "development" => Environment::Development,
                "testing" => Environment::Testing,
                "staging" => Environment::Staging,
                "production" => Environment::Production,
                _ => Environment::Development,
            };
        }

        if let Ok(atp_capacity) = std::env::var("AUTOBAHN_ATP_CAPACITY") {
            if let Ok(capacity) = atp_capacity.parse::<f64>() {
                config.system.max_atp_capacity = capacity;
            }
        }

        if let Ok(debug) = std::env::var("AUTOBAHN_DEBUG") {
            config.system.debug_mode = debug.to_lowercase() == "true";
        }

        Ok(config)
    }

    /// Generate default configuration file
    pub fn generate_default_config_file<P: AsRef<Path>>(path: P) -> AutobahnResult<()> {
        let config = AutobahnConfig::default();
        let content = toml::to_string_pretty(&config)
            .map_err(|e| AutobahnError::InitializationError(format!("Failed to serialize default config: {}", e)))?;
        
        std::fs::write(path, content)
            .map_err(|e| AutobahnError::InitializationError(format!("Failed to write default config: {}", e)))?;
        
        Ok(())
    }
}

impl Default for AutobahnConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default(),
            v8_pipeline: V8PipelineConfig::default(),
            probabilistic: ProbabilisticConfig::default(),
            temporal: TemporalConfig::default(),
            research: ResearchConfig::default(),
            plugins: PluginConfig::default(),
            benchmarking: BenchmarkConfig::default(),
            logging: LoggingConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            name: "Autobahn".to_string(),
            version: "0.1.0".to_string(),
            environment: Environment::Development,
            data_directory: "./data".to_string(),
            config_directory: "./config".to_string(),
            log_directory: "./logs".to_string(),
            max_atp_capacity: 1000.0,
            atp_regeneration_rate: 10.0,
            debug_mode: false,
            thread_pool_size: num_cpus::get(),
        }
    }
}

impl Default for V8PipelineConfig {
    fn default() -> Self {
        let mut modules = HashMap::new();
        
        // Default module configurations
        for module_name in ["Mzekezeke", "Diggiden", "Hatata", "Spectacular", "Nicotine", "Clothesline", "Zengeza", "Diadochi"] {
            modules.insert(module_name.to_string(), ModuleConfig::default());
        }

        Self {
            enabled: true,
            modules,
            atp_allocation_strategy: ATPAllocationStrategy::PriorityBased,
            processing_timeout_ms: 30000,
            enable_parallel_processing: true,
            max_concurrent_processes: 4,
        }
    }
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            priority: 50,
            atp_budget: 100.0,
            timeout_ms: 10000,
            parameters: HashMap::new(),
        }
    }
}

impl Default for ProbabilisticConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_inference_algorithm: "variable_elimination".to_string(),
            monte_carlo: MonteCarloConfig::default(),
            bayesian_networks: BayesianNetworkConfig::default(),
            uncertainty_quantification: UncertaintyConfig::default(),
        }
    }
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            default_samples: 1000,
            burn_in: 100,
            random_seed: None,
        }
    }
}

impl Default for BayesianNetworkConfig {
    fn default() -> Self {
        Self {
            max_network_size: 1000,
            learning_algorithm: "hill_climbing".to_string(),
            validation_threshold: 0.95,
        }
    }
}

impl Default for UncertaintyConfig {
    fn default() -> Self {
        Self {
            enable_propagation: true,
            confidence_level: 0.95,
            enable_sensitivity_analysis: true,
        }
    }
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            decay_function: "exponential".to_string(),
            decay_rate: 0.1,
            pattern_detection_threshold: 0.8,
            historical_retention_days: 30,
        }
    }
}

impl Default for ResearchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_quantum: false,
            enable_ml: true,
            enable_experimental: false,
            safety_constraints: SafetyConstraints::default(),
            resource_limits: ResourceLimits::default(),
        }
    }
}

impl Default for SafetyConstraints {
    fn default() -> Self {
        Self {
            max_runtime_ms: 300000,
            max_memory_mb: 1024,
            max_cpu_percent: 80.0,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_concurrent_experiments: 4,
            max_storage_gb: 10.0,
            max_bandwidth_mbps: 100.0,
        }
    }
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            plugin_directory: "./plugins".to_string(),
            max_plugins: 100,
            default_security_level: "standard".to_string(),
            loading_timeout_ms: 30000,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_ms: 30000,
            iterations: 10,
            warmup_iterations: 3,
            enable_performance: true,
            enable_atp_tests: true,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            log_to_file: true,
            log_file_path: "./logs/autobahn.log".to_string(),
            max_file_size_mb: 100.0,
            rotation_count: 5,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            monitoring_interval_ms: 1000,
            thresholds: PerformanceThresholds::default(),
            enable_auto_optimization: false,
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_processing_time_ms: 5000,
            max_memory_mb: 512.0,
            max_atp_per_operation: 50.0,
            min_confidence: 0.7,
        }
    }
}

impl Default for ConfigurationManager {
    fn default() -> Self {
        Self::new()
    }
} 