//! Plugin System for Autobahn Extensions
//!
//! This module provides a comprehensive plugin architecture that allows external
//! developers to extend Autobahn with custom biological modules, processing algorithms,
//! and specialized capabilities while maintaining system integrity and ATP economics.

use crate::types::*;
use crate::error::{AutobahnError, AutobahnResult};
use crate::traits::{BiologicalModule, MetacognitiveOrchestrator};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use async_trait::async_trait;
use std::any::Any;

/// Plugin manager for the Autobahn system
#[derive(Debug)]
pub struct PluginManager {
    /// Registered plugins
    plugins: HashMap<String, Box<dyn AutobahnPlugin>>,
    /// Plugin metadata
    plugin_metadata: HashMap<String, PluginMetadata>,
    /// Plugin dependencies
    dependencies: HashMap<String, Vec<String>>,
    /// Plugin lifecycle state
    lifecycle_states: HashMap<String, PluginLifecycleState>,
    /// Plugin configuration
    config: PluginManagerConfig,
    /// Plugin registry
    registry: PluginRegistry,
    /// Security manager
    security_manager: PluginSecurityManager,
}

/// Core plugin trait that all Autobahn plugins must implement
#[async_trait]
pub trait AutobahnPlugin: Send + Sync + std::fmt::Debug {
    /// Plugin unique identifier
    fn id(&self) -> &str;
    
    /// Plugin name
    fn name(&self) -> &str;
    
    /// Plugin version
    fn version(&self) -> &str;
    
    /// Plugin description
    fn description(&self) -> &str;
    
    /// Plugin author
    fn author(&self) -> &str;
    
    /// Plugin capabilities
    fn capabilities(&self) -> Vec<PluginCapability>;
    
    /// Plugin dependencies
    fn dependencies(&self) -> Vec<String>;
    
    /// Initialize the plugin
    async fn initialize(&mut self, context: &PluginContext) -> AutobahnResult<()>;
    
    /// Start the plugin
    async fn start(&mut self) -> AutobahnResult<()>;
    
    /// Stop the plugin
    async fn stop(&mut self) -> AutobahnResult<()>;
    
    /// Process information through the plugin
    async fn process(&mut self, input: PluginInput) -> AutobahnResult<PluginOutput>;
    
    /// Get plugin configuration schema
    fn config_schema(&self) -> PluginConfigSchema;
    
    /// Configure the plugin
    async fn configure(&mut self, config: PluginConfig) -> AutobahnResult<()>;
    
    /// Get plugin health status
    async fn health_check(&self) -> PluginHealthStatus;
    
    /// Handle plugin events
    async fn handle_event(&mut self, event: PluginEvent) -> AutobahnResult<()>;
    
    /// Get plugin as Any for downcasting
    fn as_any(&self) -> &dyn Any;
    
    /// Get mutable plugin as Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Plugin capabilities enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginCapability {
    /// Text processing capability
    TextProcessing,
    /// Numerical analysis capability
    NumericalAnalysis,
    /// Biological pathway simulation
    BiologicalPathwaySimulation,
    /// Machine learning integration
    MachineLearningIntegration,
    /// Data visualization
    DataVisualization,
    /// External API integration
    ExternalAPIIntegration,
    /// Custom ATP generation
    CustomATPGeneration,
    /// Adversarial testing
    AdversarialTesting,
    /// Temporal analysis
    TemporalAnalysis,
    /// Probabilistic reasoning
    ProbabilisticReasoning,
    /// Custom capability
    Custom(String),
}

/// Plugin metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin ID
    pub id: String,
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Plugin author
    pub author: String,
    /// Plugin homepage URL
    pub homepage: Option<String>,
    /// Plugin repository URL
    pub repository: Option<String>,
    /// Plugin license
    pub license: String,
    /// Plugin keywords
    pub keywords: Vec<String>,
    /// Plugin categories
    pub categories: Vec<PluginCategory>,
    /// Minimum Autobahn version required
    pub min_autobahn_version: String,
    /// Maximum Autobahn version supported
    pub max_autobahn_version: Option<String>,
    /// Plugin capabilities
    pub capabilities: Vec<PluginCapability>,
    /// Plugin dependencies
    pub dependencies: Vec<String>,
    /// Plugin installation time
    pub installed_at: chrono::DateTime<chrono::Utc>,
    /// Plugin last updated
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Plugin categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginCategory {
    /// Biological processing
    BiologicalProcessing,
    /// Data analysis
    DataAnalysis,
    /// Machine learning
    MachineLearning,
    /// Visualization
    Visualization,
    /// Integration
    Integration,
    /// Security
    Security,
    /// Performance
    Performance,
    /// Research
    Research,
    /// Custom category
    Custom(String),
}

/// Plugin lifecycle states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginLifecycleState {
    /// Plugin is installed but not initialized
    Installed,
    /// Plugin is initializing
    Initializing,
    /// Plugin is initialized but not started
    Initialized,
    /// Plugin is starting
    Starting,
    /// Plugin is running
    Running,
    /// Plugin is stopping
    Stopping,
    /// Plugin is stopped
    Stopped,
    /// Plugin has failed
    Failed(String),
    /// Plugin is being uninstalled
    Uninstalling,
}

/// Plugin context provided during initialization
#[derive(Debug, Clone)]
pub struct PluginContext {
    /// Autobahn system version
    pub autobahn_version: String,
    /// Available system capabilities
    pub system_capabilities: Vec<String>,
    /// Plugin data directory
    pub data_directory: String,
    /// Plugin configuration directory
    pub config_directory: String,
    /// System ATP manager reference
    pub atp_manager: Arc<Mutex<dyn crate::traits::EnergyManager>>,
    /// Plugin communication channel
    pub communication_channel: PluginCommunicationChannel,
    /// Security context
    pub security_context: SecurityContext,
}

/// Plugin input data
#[derive(Debug, Clone)]
pub struct PluginInput {
    /// Input data
    pub data: PluginData,
    /// Processing context
    pub context: ProcessingContext,
    /// ATP budget for processing
    pub atp_budget: f64,
    /// Processing priority
    pub priority: ProcessingPriority,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Plugin data types
#[derive(Debug, Clone)]
pub enum PluginData {
    /// Text data
    Text(String),
    /// Numerical data
    Numerical(Vec<f64>),
    /// Structured data
    Structured(serde_json::Value),
    /// Binary data
    Binary(Vec<u8>),
    /// Custom data
    Custom(Box<dyn PluginCustomData>),
}

/// Trait for custom plugin data
pub trait PluginCustomData: Send + Sync + std::fmt::Debug {
    fn as_any(&self) -> &dyn Any;
    fn clone_box(&self) -> Box<dyn PluginCustomData>;
}

impl Clone for Box<dyn PluginCustomData> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Plugin output data
#[derive(Debug, Clone)]
pub struct PluginOutput {
    /// Output data
    pub data: PluginData,
    /// Processing result metadata
    pub metadata: ProcessingResultMetadata,
    /// ATP consumed
    pub atp_consumed: f64,
    /// Confidence score
    pub confidence: f64,
    /// Processing time
    pub processing_time_ms: u64,
}

/// Processing result metadata
#[derive(Debug, Clone)]
pub struct ProcessingResultMetadata {
    /// Processing success
    pub success: bool,
    /// Result type
    pub result_type: String,
    /// Quality metrics
    pub quality_metrics: HashMap<String, f64>,
    /// Warnings
    pub warnings: Vec<String>,
    /// Additional information
    pub additional_info: HashMap<String, String>,
}

/// Processing priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Plugin configuration schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfigSchema {
    /// Configuration parameters
    pub parameters: Vec<ConfigParameter>,
    /// Configuration groups
    pub groups: Vec<ConfigGroup>,
    /// Validation rules
    pub validation_rules: Vec<ValidationRule>,
}

/// Configuration parameter definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub parameter_type: ConfigParameterType,
    /// Parameter description
    pub description: String,
    /// Default value
    pub default_value: Option<serde_json::Value>,
    /// Required parameter
    pub required: bool,
    /// Parameter constraints
    pub constraints: Vec<ParameterConstraint>,
}

/// Configuration parameter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
    Enum(Vec<String>),
}

/// Parameter constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterConstraint {
    MinValue(f64),
    MaxValue(f64),
    MinLength(usize),
    MaxLength(usize),
    Pattern(String),
    Custom(String),
}

/// Configuration group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigGroup {
    /// Group name
    pub name: String,
    /// Group description
    pub description: String,
    /// Parameters in this group
    pub parameters: Vec<String>,
}

/// Validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,
    /// Rule expression
    pub expression: String,
    /// Error message
    pub error_message: String,
}

/// Plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Configuration values
    pub values: HashMap<String, serde_json::Value>,
    /// Configuration metadata
    pub metadata: ConfigMetadata,
}

/// Configuration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMetadata {
    /// Configuration version
    pub version: String,
    /// Last modified timestamp
    pub last_modified: chrono::DateTime<chrono::Utc>,
    /// Modified by
    pub modified_by: String,
    /// Configuration source
    pub source: String,
}

/// Plugin health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginHealthStatus {
    /// Overall health
    pub healthy: bool,
    /// Health score (0.0 to 1.0)
    pub health_score: f64,
    /// Health checks
    pub checks: Vec<HealthCheck>,
    /// Last health check time
    pub last_check: chrono::DateTime<chrono::Utc>,
    /// Health trends
    pub trends: Vec<HealthTrend>,
}

/// Individual health check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    /// Check name
    pub name: String,
    /// Check result
    pub passed: bool,
    /// Check message
    pub message: String,
    /// Check duration
    pub duration_ms: u64,
}

/// Health trend information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthTrend {
    /// Metric name
    pub metric: String,
    /// Trend direction
    pub trend: TrendDirection,
    /// Trend confidence
    pub confidence: f64,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Plugin events
#[derive(Debug, Clone)]
pub enum PluginEvent {
    /// System startup event
    SystemStartup,
    /// System shutdown event
    SystemShutdown,
    /// Configuration changed event
    ConfigurationChanged(PluginConfig),
    /// ATP levels changed event
    ATPLevelsChanged(f64),
    /// Processing request event
    ProcessingRequest(PluginInput),
    /// Plugin dependency updated event
    DependencyUpdated(String, String),
    /// Custom event
    Custom(String, serde_json::Value),
}

/// Plugin communication channel
#[derive(Debug, Clone)]
pub struct PluginCommunicationChannel {
    /// Channel ID
    pub channel_id: String,
    /// Send message function
    pub sender: Arc<Mutex<dyn PluginMessageSender>>,
    /// Message handler
    pub message_handler: Arc<Mutex<dyn PluginMessageHandler>>,
}

/// Plugin message sender trait
pub trait PluginMessageSender: Send + Sync {
    fn send_message(&self, message: PluginMessage) -> AutobahnResult<()>;
    fn broadcast_message(&self, message: PluginMessage) -> AutobahnResult<()>;
}

/// Plugin message handler trait
pub trait PluginMessageHandler: Send + Sync {
    fn handle_message(&mut self, message: PluginMessage) -> AutobahnResult<()>;
}

/// Plugin message
#[derive(Debug, Clone)]
pub struct PluginMessage {
    /// Message ID
    pub id: String,
    /// Source plugin ID
    pub source: String,
    /// Target plugin ID (None for broadcast)
    pub target: Option<String>,
    /// Message type
    pub message_type: PluginMessageType,
    /// Message payload
    pub payload: serde_json::Value,
    /// Message timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Plugin message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginMessageType {
    /// Request message
    Request,
    /// Response message
    Response,
    /// Notification message
    Notification,
    /// Event message
    Event,
    /// Custom message type
    Custom(String),
}

/// Security context for plugins
#[derive(Debug, Clone)]
pub struct SecurityContext {
    /// Plugin permissions
    pub permissions: Vec<PluginPermission>,
    /// Security level
    pub security_level: SecurityLevel,
    /// Sandbox configuration
    pub sandbox_config: SandboxConfig,
    /// Trust level
    pub trust_level: TrustLevel,
}

/// Plugin permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginPermission {
    /// Read file system
    ReadFileSystem,
    /// Write file system
    WriteFileSystem,
    /// Network access
    NetworkAccess,
    /// System process access
    ProcessAccess,
    /// ATP management
    ATPManagement,
    /// Plugin management
    PluginManagement,
    /// Configuration access
    ConfigurationAccess,
    /// Custom permission
    Custom(String),
}

/// Security levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Minimal security (trusted plugins)
    Minimal,
    /// Standard security
    Standard,
    /// High security
    High,
    /// Maximum security (untrusted plugins)
    Maximum,
}

/// Sandbox configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Enable filesystem sandbox
    pub filesystem_sandbox: bool,
    /// Enable network sandbox
    pub network_sandbox: bool,
    /// Enable process sandbox
    pub process_sandbox: bool,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

/// Resource limits for plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage (MB)
    pub max_memory_mb: f64,
    /// Maximum CPU usage (%)
    pub max_cpu_percent: f64,
    /// Maximum ATP consumption per operation
    pub max_atp_per_operation: f64,
    /// Maximum processing time per operation (ms)
    pub max_processing_time_ms: u64,
}

/// Trust levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrustLevel {
    /// Untrusted (third-party plugins)
    Untrusted,
    /// Limited trust
    LimitedTrust,
    /// Trusted (verified plugins)
    Trusted,
    /// Fully trusted (core plugins)
    FullyTrusted,
}

/// Plugin manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManagerConfig {
    /// Enable plugin system
    pub enabled: bool,
    /// Plugin directory
    pub plugin_directory: String,
    /// Maximum number of plugins
    pub max_plugins: usize,
    /// Default security level
    pub default_security_level: SecurityLevel,
    /// Default trust level
    pub default_trust_level: TrustLevel,
    /// Plugin loading timeout (ms)
    pub loading_timeout_ms: u64,
    /// Enable plugin hot-reloading
    pub enable_hot_reload: bool,
}

/// Plugin registry for discovering and managing plugins
#[derive(Debug, Clone)]
pub struct PluginRegistry {
    /// Available plugins
    available_plugins: HashMap<String, PluginRegistryEntry>,
    /// Plugin sources
    plugin_sources: Vec<PluginSource>,
    /// Registry configuration
    config: RegistryConfig,
}

/// Plugin registry entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRegistryEntry {
    /// Plugin metadata
    pub metadata: PluginMetadata,
    /// Plugin location
    pub location: PluginLocation,
    /// Plugin signature (for verification)
    pub signature: Option<String>,
    /// Plugin checksum
    pub checksum: String,
    /// Registry entry timestamp
    pub registry_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Plugin location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginLocation {
    /// Local file path
    LocalPath(String),
    /// Remote URL
    RemoteURL(String),
    /// Package registry
    PackageRegistry { registry: String, package: String, version: String },
    /// Git repository
    GitRepository { url: String, branch: Option<String>, tag: Option<String> },
}

/// Plugin source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginSource {
    /// Source name
    pub name: String,
    /// Source URL
    pub url: String,
    /// Source type
    pub source_type: PluginSourceType,
    /// Authentication configuration
    pub auth_config: Option<AuthConfig>,
    /// Source enabled
    pub enabled: bool,
}

/// Plugin source types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginSourceType {
    /// Official Autobahn plugin registry
    Official,
    /// Community plugin registry
    Community,
    /// Private registry
    Private,
    /// Local directory
    Local,
    /// Git repository
    Git,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication type
    pub auth_type: AuthType,
    /// Credentials
    pub credentials: HashMap<String, String>,
}

/// Authentication types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    None,
    Basic,
    Token,
    OAuth,
    Certificate,
}

/// Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Enable automatic updates
    pub auto_update: bool,
    /// Update check interval (hours)
    pub update_interval_hours: u64,
    /// Enable plugin verification
    pub verify_signatures: bool,
    /// Trusted signers
    pub trusted_signers: Vec<String>,
}

/// Plugin security manager
#[derive(Debug)]
pub struct PluginSecurityManager {
    /// Security policies
    policies: HashMap<String, SecurityPolicy>,
    /// Permission manager
    permission_manager: PermissionManager,
    /// Sandbox manager
    sandbox_manager: SandboxManager,
    /// Audit logger
    audit_logger: AuditLogger,
}

/// Security policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    /// Policy name
    pub name: String,
    /// Policy rules
    pub rules: Vec<SecurityRule>,
    /// Default action
    pub default_action: SecurityAction,
}

/// Security rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityRule {
    /// Rule condition
    pub condition: SecurityCondition,
    /// Rule action
    pub action: SecurityAction,
    /// Rule priority
    pub priority: u32,
}

/// Security condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityCondition {
    /// Plugin ID matches pattern
    PluginIdMatches(String),
    /// Permission requested
    PermissionRequested(PluginPermission),
    /// Trust level condition
    TrustLevel(TrustLevel),
    /// Resource usage condition
    ResourceUsage { resource: String, threshold: f64 },
    /// Custom condition
    Custom(String),
}

/// Security action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityAction {
    /// Allow the action
    Allow,
    /// Deny the action
    Deny,
    /// Allow with restrictions
    AllowWithRestrictions(Vec<String>),
    /// Request user approval
    RequestApproval,
    /// Log and allow
    LogAndAllow,
    /// Log and deny
    LogAndDeny,
}

/// Permission manager
#[derive(Debug)]
pub struct PermissionManager {
    /// Plugin permissions
    plugin_permissions: HashMap<String, Vec<PluginPermission>>,
    /// Permission grants
    permission_grants: HashMap<String, Vec<PermissionGrant>>,
}

/// Permission grant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionGrant {
    /// Permission
    pub permission: PluginPermission,
    /// Grant timestamp
    pub granted_at: chrono::DateTime<chrono::Utc>,
    /// Grant expiration
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Granted by
    pub granted_by: String,
    /// Grant conditions
    pub conditions: Vec<String>,
}

/// Sandbox manager
#[derive(Debug)]
pub struct SandboxManager {
    /// Active sandboxes
    sandboxes: HashMap<String, Sandbox>,
    /// Sandbox configurations
    configurations: HashMap<String, SandboxConfig>,
}

/// Sandbox instance
#[derive(Debug)]
pub struct Sandbox {
    /// Sandbox ID
    pub id: String,
    /// Plugin ID
    pub plugin_id: String,
    /// Sandbox configuration
    pub config: SandboxConfig,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Sandbox status
    pub status: SandboxStatus,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// Current memory usage (MB)
    pub memory_mb: f64,
    /// Current CPU usage (%)
    pub cpu_percent: f64,
    /// ATP consumed
    pub atp_consumed: f64,
    /// Processing time (ms)
    pub processing_time_ms: u64,
}

/// Sandbox status
#[derive(Debug, Clone)]
pub enum SandboxStatus {
    /// Sandbox is initializing
    Initializing,
    /// Sandbox is active
    Active,
    /// Sandbox is suspended
    Suspended,
    /// Sandbox is terminated
    Terminated,
    /// Sandbox has failed
    Failed(String),
}

/// Audit logger for security events
#[derive(Debug)]
pub struct AuditLogger {
    /// Audit log entries
    log_entries: Vec<AuditLogEntry>,
    /// Log configuration
    config: AuditLogConfig,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    /// Entry ID
    pub id: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Plugin ID
    pub plugin_id: String,
    /// Event type
    pub event_type: AuditEventType,
    /// Event description
    pub description: String,
    /// Event severity
    pub severity: AuditSeverity,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Audit event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Plugin installation
    PluginInstalled,
    /// Plugin uninstallation
    PluginUninstalled,
    /// Plugin started
    PluginStarted,
    /// Plugin stopped
    PluginStopped,
    /// Permission granted
    PermissionGranted,
    /// Permission denied
    PermissionDenied,
    /// Security violation
    SecurityViolation,
    /// Resource limit exceeded
    ResourceLimitExceeded,
    /// Custom event
    Custom(String),
}

/// Audit severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Audit log configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Log file path
    pub log_file_path: String,
    /// Maximum log file size (MB)
    pub max_file_size_mb: f64,
    /// Log retention days
    pub retention_days: u32,
    /// Log level
    pub log_level: AuditSeverity,
}

impl PluginManager {
    /// Create new plugin manager
    pub fn new(config: PluginManagerConfig) -> Self {
        Self {
            plugins: HashMap::new(),
            plugin_metadata: HashMap::new(),
            dependencies: HashMap::new(),
            lifecycle_states: HashMap::new(),
            config,
            registry: PluginRegistry::new(),
            security_manager: PluginSecurityManager::new(),
        }
    }

    /// Install a plugin
    pub async fn install_plugin(
        &mut self,
        plugin_location: PluginLocation,
        install_config: PluginInstallConfig,
    ) -> AutobahnResult<String> {
        // Verify plugin security
        self.security_manager.verify_plugin(&plugin_location).await?;
        
        // Load plugin metadata
        let metadata = self.load_plugin_metadata(&plugin_location).await?;
        
        // Check dependencies
        self.check_dependencies(&metadata.dependencies).await?;
        
        // Install plugin
        let plugin_id = metadata.id.clone();
        
        // Update registry
        self.registry.register_plugin(metadata.clone(), plugin_location).await?;
        
        // Set initial state
        self.lifecycle_states.insert(plugin_id.clone(), PluginLifecycleState::Installed);
        self.plugin_metadata.insert(plugin_id.clone(), metadata);
        
        // Log installation
        self.security_manager.log_audit_event(
            AuditLogEntry {
                id: uuid::Uuid::new_v4().to_string(),
                timestamp: chrono::Utc::now(),
                plugin_id: plugin_id.clone(),
                event_type: AuditEventType::PluginInstalled,
                description: "Plugin installed successfully".to_string(),
                severity: AuditSeverity::Info,
                metadata: HashMap::new(),
            }
        );
        
        Ok(plugin_id)
    }

    /// Load a plugin
    pub async fn load_plugin(&mut self, plugin_id: &str) -> AutobahnResult<()> {
        if !self.plugin_metadata.contains_key(plugin_id) {
            return Err(AutobahnError::ProcessingError {
                layer: "plugin_manager".to_string(),
                reason: format!("Plugin {} not found", plugin_id),
            });
        }

        // Set state to initializing
        self.lifecycle_states.insert(plugin_id.to_string(), PluginLifecycleState::Initializing);
        
        // Load plugin implementation (this would be dynamic loading in real implementation)
        // For now, we'll use a placeholder
        
        // Set state to initialized
        self.lifecycle_states.insert(plugin_id.to_string(), PluginLifecycleState::Initialized);
        
        Ok(())
    }

    /// Start a plugin
    pub async fn start_plugin(&mut self, plugin_id: &str) -> AutobahnResult<()> {
        if let Some(plugin) = self.plugins.get_mut(plugin_id) {
            self.lifecycle_states.insert(plugin_id.to_string(), PluginLifecycleState::Starting);
            
            plugin.start().await?;
            
            self.lifecycle_states.insert(plugin_id.to_string(), PluginLifecycleState::Running);
            
            // Log start event
            self.security_manager.log_audit_event(
                AuditLogEntry {
                    id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now(),
                    plugin_id: plugin_id.to_string(),
                    event_type: AuditEventType::PluginStarted,
                    description: "Plugin started successfully".to_string(),
                    severity: AuditSeverity::Info,
                    metadata: HashMap::new(),
                }
            );
            
            Ok(())
        } else {
            Err(AutobahnError::ProcessingError {
                layer: "plugin_manager".to_string(),
                reason: format!("Plugin {} not loaded", plugin_id),
            })
        }
    }

    /// Stop a plugin
    pub async fn stop_plugin(&mut self, plugin_id: &str) -> AutobahnResult<()> {
        if let Some(plugin) = self.plugins.get_mut(plugin_id) {
            self.lifecycle_states.insert(plugin_id.to_string(), PluginLifecycleState::Stopping);
            
            plugin.stop().await?;
            
            self.lifecycle_states.insert(plugin_id.to_string(), PluginLifecycleState::Stopped);
            
            // Log stop event
            self.security_manager.log_audit_event(
                AuditLogEntry {
                    id: uuid::Uuid::new_v4().to_string(),
                    timestamp: chrono::Utc::now(),
                    plugin_id: plugin_id.to_string(),
                    event_type: AuditEventType::PluginStopped,
                    description: "Plugin stopped successfully".to_string(),
                    severity: AuditSeverity::Info,
                    metadata: HashMap::new(),
                }
            );
            
            Ok(())
        } else {
            Err(AutobahnError::ProcessingError {
                layer: "plugin_manager".to_string(),
                reason: format!("Plugin {} not found", plugin_id),
            })
        }
    }

    /// Process data through a plugin
    pub async fn process_with_plugin(
        &mut self,
        plugin_id: &str,
        input: PluginInput,
    ) -> AutobahnResult<PluginOutput> {
        if let Some(plugin) = self.plugins.get_mut(plugin_id) {
            // Check security permissions
            self.security_manager.check_permissions(plugin_id, &input).await?;
            
            // Process through plugin
            let output = plugin.process(input).await?;
            
            Ok(output)
        } else {
            Err(AutobahnError::ProcessingError {
                layer: "plugin_manager".to_string(),
                reason: format!("Plugin {} not available", plugin_id),
            })
        }
    }

    /// List installed plugins
    pub fn list_plugins(&self) -> Vec<&PluginMetadata> {
        self.plugin_metadata.values().collect()
    }

    /// Get plugin status
    pub fn get_plugin_status(&self, plugin_id: &str) -> Option<&PluginLifecycleState> {
        self.lifecycle_states.get(plugin_id)
    }

    /// Update plugin configuration
    pub async fn configure_plugin(
        &mut self,
        plugin_id: &str,
        config: PluginConfig,
    ) -> AutobahnResult<()> {
        if let Some(plugin) = self.plugins.get_mut(plugin_id) {
            plugin.configure(config).await?;
            Ok(())
        } else {
            Err(AutobahnError::ProcessingError {
                layer: "plugin_manager".to_string(),
                reason: format!("Plugin {} not found", plugin_id),
            })
        }
    }

    /// Get plugin health status
    pub async fn get_plugin_health(&self, plugin_id: &str) -> AutobahnResult<PluginHealthStatus> {
        if let Some(plugin) = self.plugins.get(plugin_id) {
            Ok(plugin.health_check().await)
        } else {
            Err(AutobahnError::ProcessingError {
                layer: "plugin_manager".to_string(),
                reason: format!("Plugin {} not found", plugin_id),
            })
        }
    }

    // Helper methods
    async fn load_plugin_metadata(&self, _location: &PluginLocation) -> AutobahnResult<PluginMetadata> {
        // Placeholder implementation
        Ok(PluginMetadata {
            id: "example_plugin".to_string(),
            name: "Example Plugin".to_string(),
            version: "1.0.0".to_string(),
            description: "Example plugin for demonstration".to_string(),
            author: "Autobahn Team".to_string(),
            homepage: None,
            repository: None,
            license: "MIT".to_string(),
            keywords: vec!["example".to_string()],
            categories: vec![PluginCategory::DataAnalysis],
            min_autobahn_version: "0.1.0".to_string(),
            max_autobahn_version: None,
            capabilities: vec![PluginCapability::TextProcessing],
            dependencies: Vec::new(),
            installed_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        })
    }

    async fn check_dependencies(&self, _dependencies: &[String]) -> AutobahnResult<()> {
        // Placeholder implementation
        Ok(())
    }
}

/// Plugin installation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInstallConfig {
    /// Force installation (override conflicts)
    pub force: bool,
    /// Skip dependency check
    pub skip_dependencies: bool,
    /// Installation timeout (ms)
    pub timeout_ms: u64,
    /// Custom installation parameters
    pub parameters: HashMap<String, String>,
}

// Implement placeholder types
impl PluginRegistry {
    fn new() -> Self {
        Self {
            available_plugins: HashMap::new(),
            plugin_sources: Vec::new(),
            config: RegistryConfig {
                auto_update: true,
                update_interval_hours: 24,
                verify_signatures: true,
                trusted_signers: Vec::new(),
            },
        }
    }

    async fn register_plugin(
        &mut self,
        metadata: PluginMetadata,
        location: PluginLocation,
    ) -> AutobahnResult<()> {
        let entry = PluginRegistryEntry {
            metadata: metadata.clone(),
            location,
            signature: None,
            checksum: "placeholder_checksum".to_string(),
            registry_timestamp: chrono::Utc::now(),
        };
        
        self.available_plugins.insert(metadata.id, entry);
        Ok(())
    }
}

impl PluginSecurityManager {
    fn new() -> Self {
        Self {
            policies: HashMap::new(),
            permission_manager: PermissionManager::new(),
            sandbox_manager: SandboxManager::new(),
            audit_logger: AuditLogger::new(),
        }
    }

    async fn verify_plugin(&self, _location: &PluginLocation) -> AutobahnResult<()> {
        // Placeholder implementation
        Ok(())
    }

    async fn check_permissions(&self, _plugin_id: &str, _input: &PluginInput) -> AutobahnResult<()> {
        // Placeholder implementation
        Ok(())
    }

    fn log_audit_event(&mut self, entry: AuditLogEntry) {
        self.audit_logger.log_entries.push(entry);
    }
}

impl PermissionManager {
    fn new() -> Self {
        Self {
            plugin_permissions: HashMap::new(),
            permission_grants: HashMap::new(),
        }
    }
}

impl SandboxManager {
    fn new() -> Self {
        Self {
            sandboxes: HashMap::new(),
            configurations: HashMap::new(),
        }
    }
}

impl AuditLogger {
    fn new() -> Self {
        Self {
            log_entries: Vec::new(),
            config: AuditLogConfig {
                enabled: true,
                log_file_path: "autobahn_audit.log".to_string(),
                max_file_size_mb: 100.0,
                retention_days: 30,
                log_level: AuditSeverity::Info,
            },
        }
    }
}

impl Default for PluginManagerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            plugin_directory: "./plugins".to_string(),
            max_plugins: 100,
            default_security_level: SecurityLevel::Standard,
            default_trust_level: TrustLevel::LimitedTrust,
            loading_timeout_ms: 30000,
            enable_hot_reload: false,
        }
    }
}

impl Default for PluginInstallConfig {
    fn default() -> Self {
        Self {
            force: false,
            skip_dependencies: false,
            timeout_ms: 60000,
            parameters: HashMap::new(),
        }
    }
}

/// Example plugin implementation
#[derive(Debug)]
pub struct ExamplePlugin {
    id: String,
    name: String,
    version: String,
    config: Option<PluginConfig>,
    running: bool,
}

#[async_trait]
impl AutobahnPlugin for ExamplePlugin {
    fn id(&self) -> &str { &self.id }
    fn name(&self) -> &str { &self.name }
    fn version(&self) -> &str { &self.version }
    fn description(&self) -> &str { "Example plugin for demonstration" }
    fn author(&self) -> &str { "Autobahn Team" }
    
    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![PluginCapability::TextProcessing]
    }
    
    fn dependencies(&self) -> Vec<String> { Vec::new() }
    
    async fn initialize(&mut self, _context: &PluginContext) -> AutobahnResult<()> {
        log::info!("Initializing example plugin");
        Ok(())
    }
    
    async fn start(&mut self) -> AutobahnResult<()> {
        self.running = true;
        log::info!("Started example plugin");
        Ok(())
    }
    
    async fn stop(&mut self) -> AutobahnResult<()> {
        self.running = false;
        log::info!("Stopped example plugin");
        Ok(())
    }
    
    async fn process(&mut self, input: PluginInput) -> AutobahnResult<PluginOutput> {
        if !self.running {
            return Err(AutobahnError::ProcessingError {
                layer: "example_plugin".to_string(),
                reason: "Plugin not running".to_string(),
            });
        }

        // Simple text processing example
        let output_data = match input.data {
            PluginData::Text(text) => {
                PluginData::Text(format!("Processed: {}", text))
            }
            other => other,
        };

        Ok(PluginOutput {
            data: output_data,
            metadata: ProcessingResultMetadata {
                success: true,
                result_type: "text_processing".to_string(),
                quality_metrics: HashMap::new(),
                warnings: Vec::new(),
                additional_info: HashMap::new(),
            },
            atp_consumed: 5.0,
            confidence: 0.9,
            processing_time_ms: 10,
        })
    }
    
    fn config_schema(&self) -> PluginConfigSchema {
        PluginConfigSchema {
            parameters: vec![
                ConfigParameter {
                    name: "processing_mode".to_string(),
                    parameter_type: ConfigParameterType::Enum(vec![
                        "fast".to_string(),
                        "accurate".to_string(),
                    ]),
                    description: "Processing mode selection".to_string(),
                    default_value: Some(serde_json::Value::String("fast".to_string())),
                    required: false,
                    constraints: Vec::new(),
                }
            ],
            groups: Vec::new(),
            validation_rules: Vec::new(),
        }
    }
    
    async fn configure(&mut self, config: PluginConfig) -> AutobahnResult<()> {
        self.config = Some(config);
        Ok(())
    }
    
    async fn health_check(&self) -> PluginHealthStatus {
        PluginHealthStatus {
            healthy: self.running,
            health_score: if self.running { 1.0 } else { 0.0 },
            checks: vec![
                HealthCheck {
                    name: "running_status".to_string(),
                    passed: self.running,
                    message: if self.running { "Plugin is running" } else { "Plugin is stopped" }.to_string(),
                    duration_ms: 1,
                }
            ],
            last_check: chrono::Utc::now(),
            trends: Vec::new(),
        }
    }
    
    async fn handle_event(&mut self, _event: PluginEvent) -> AutobahnResult<()> {
        Ok(())
    }
    
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

impl ExamplePlugin {
    pub fn new() -> Self {
        Self {
            id: "example_plugin".to_string(),
            name: "Example Plugin".to_string(),
            version: "1.0.0".to_string(),
            config: None,
            running: false,
        }
    }
} 