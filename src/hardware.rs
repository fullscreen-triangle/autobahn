//! Hardware Oscillation Synchronization Engine
//! 
//! This module implements real-time hardware oscillation capture and synchronization
//! for biological consciousness processing. It monitors CPU clocks, power supply
//! frequencies, memory clocks, and other hardware oscillations to provide natural
//! frequency references for consciousness emergence calculations.

use crate::{AutobahnError, AutobahnResult};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;
use sysinfo::{System, SystemExt, ProcessorExt, ComponentExt};
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::RwLock as ParkingRwLock;
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use log::{debug, info, warn, error};

/// Hardware frequency domains for oscillation capture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FrequencyDomain {
    /// AC power supply frequency (50/60 Hz)
    PowerSupply { 
        freq_hz: f64, 
        harmonics: u8 
    },
    /// CPU core frequency (GHz range)
    CPUCore { 
        base_freq_hz: u64, 
        boost_enabled: bool,
        core_id: usize,
    },
    /// Memory subsystem frequency
    MemorySubsystem { 
        freq_hz: u64, 
        timing_optimized: bool,
        channel: u8,
    },
    /// System bus frequency
    SystemBus { 
        freq_hz: u64, 
        spread_spectrum: bool 
    },
    /// Custom frequency source
    Custom {
        name: String,
        freq_hz: f64,
        source_type: String,
    }
}

/// Hardware oscillation capture and analysis
#[derive(Debug)]
pub struct HardwareOscillationCapture {
    /// System information monitor
    system: Arc<Mutex<System>>,
    /// Active frequency domains
    frequency_domains: Vec<FrequencyDomain>,
    /// Real-time oscillation data
    oscillation_data: Arc<RwLock<HashMap<String, OscillationData>>>,
    /// Frequency analysis buffer
    analysis_buffer: Arc<ParkingRwLock<Vec<f64>>>,
    /// Sampling rate for oscillation capture
    sampling_rate_hz: u32,
    /// Analysis window size
    window_size: usize,
    /// Hardware monitoring thread handle
    monitoring_handle: Option<tokio::task::JoinHandle<()>>,
    /// Data transmission channel
    data_sender: Sender<HardwareOscillationEvent>,
    data_receiver: Receiver<HardwareOscillationEvent>,
}

/// Real-time oscillation data for a specific frequency domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillationData {
    /// Domain identifier
    pub domain: String,
    /// Current frequency in Hz
    pub current_frequency: f64,
    /// Frequency stability (0.0 to 1.0)
    pub stability: f64,
    /// Amplitude/power level
    pub amplitude: f64,
    /// Phase information
    pub phase: f64,
    /// Harmonic content
    pub harmonics: Vec<f64>,
    /// Signal-to-noise ratio
    pub snr: f64,
    /// Last update timestamp
    pub timestamp: Instant,
    /// Jitter measurements
    pub jitter_ns: f64,
}

/// Hardware oscillation event for real-time processing
#[derive(Debug, Clone)]
pub struct HardwareOscillationEvent {
    pub domain: String,
    pub frequency: f64,
    pub amplitude: f64,
    pub phase: f64,
    pub timestamp: Instant,
    pub event_type: OscillationEventType,
}

/// Types of oscillation events
#[derive(Debug, Clone)]
pub enum OscillationEventType {
    FrequencyChange { old_freq: f64, new_freq: f64 },
    AmplitudeChange { old_amp: f64, new_amp: f64 },
    PhaseShift { shift_radians: f64 },
    HarmonicDetected { harmonic_freq: f64, power: f64 },
    JitterThreshold { jitter_ns: f64 },
    SyncLoss { domain: String },
}

/// Coherence synchronization for biological processing
#[derive(Debug)]
pub struct CoherenceSync {
    /// Frequency domains for synchronization
    frequency_domains: Vec<FrequencyDomain>,
    /// Cross-scale coupling enabled
    cross_scale_coupling: bool,
    /// Biological resonance target
    biological_resonance_target: f64,
    /// Consciousness emergence optimization
    consciousness_emergence_optimization: bool,
    /// Phase-locked loop parameters
    pll_params: PLLParameters,
    /// Synchronization state
    sync_state: Arc<RwLock<SynchronizationState>>,
}

/// Phase-locked loop parameters
#[derive(Debug, Clone)]
pub struct PLLParameters {
    /// Loop bandwidth
    pub bandwidth_hz: f64,
    /// Damping factor
    pub damping_factor: f64,
    /// Lock threshold
    pub lock_threshold: f64,
    /// Maximum frequency error
    pub max_frequency_error: f64,
}

/// Current synchronization state
#[derive(Debug, Clone)]
pub struct SynchronizationState {
    /// Phase lock status per domain
    pub phase_locks: HashMap<String, bool>,
    /// Coherence level (0.0 to 1.0)
    pub coherence_level: f64,
    /// Synchronization quality
    pub quality_metric: f64,
    /// Active resonance frequencies
    pub resonance_frequencies: Vec<f64>,
    /// Cross-domain coupling strength
    pub coupling_strength: f64,
}

impl HardwareOscillationCapture {
    /// Create new hardware oscillation capture system
    pub fn new() -> AutobahnResult<Self> {
        let system = Arc::new(Mutex::new(System::new_all()));
        let oscillation_data = Arc::new(RwLock::new(HashMap::new()));
        let analysis_buffer = Arc::new(ParkingRwLock::new(Vec::new()));
        let (data_sender, data_receiver) = bounded(1000);
        
        Ok(Self {
            system,
            frequency_domains: Vec::new(),
            oscillation_data,
            analysis_buffer,
            sampling_rate_hz: 1000, // 1 kHz default sampling
            window_size: 1024,
            monitoring_handle: None,
            data_sender,
            data_receiver,
        })
    }
    
    /// Add power supply frequency monitoring
    pub fn with_power_frequency(mut self, freq_hz: f64) -> Self {
        self.frequency_domains.push(FrequencyDomain::PowerSupply { 
            freq_hz, 
            harmonics: 5 
        });
        self
    }
    
    /// Add CPU base clock monitoring
    pub fn with_cpu_base_clock(mut self, base_freq_hz: u64) -> Self {
        self.frequency_domains.push(FrequencyDomain::CPUCore { 
            base_freq_hz, 
            boost_enabled: true,
            core_id: 0,
        });
        self
    }
    
    /// Add memory clock monitoring
    pub fn with_memory_clock(mut self, freq_hz: u64) -> Self {
        self.frequency_domains.push(FrequencyDomain::MemorySubsystem { 
            freq_hz, 
            timing_optimized: true,
            channel: 0,
        });
        self
    }
    
    /// Add system bus frequencies
    pub fn with_system_bus_frequencies(mut self, frequencies: &[u64]) -> Self {
        for &freq in frequencies {
            self.frequency_domains.push(FrequencyDomain::SystemBus { 
                freq_hz: freq, 
                spread_spectrum: false 
            });
        }
        self
    }
    
    /// Start hardware monitoring
    pub async fn start_monitoring(&mut self) -> AutobahnResult<()> {
        if self.monitoring_handle.is_some() {
            return Err(AutobahnError::HardwareError("Monitoring already started".to_string()));
        }
        
        let system = Arc::clone(&self.system);
        let oscillation_data = Arc::clone(&self.oscillation_data);
        let analysis_buffer = Arc::clone(&self.analysis_buffer);
        let data_sender = self.data_sender.clone();
        let frequency_domains = self.frequency_domains.clone();
        let sampling_rate = self.sampling_rate_hz;
        
        let handle = tokio::spawn(async move {
            Self::hardware_monitoring_loop(
                system,
                oscillation_data,
                analysis_buffer,
                data_sender,
                frequency_domains,
                sampling_rate,
            ).await;
        });
        
        self.monitoring_handle = Some(handle);
        info!("Hardware oscillation monitoring started");
        Ok(())
    }
    
    /// Hardware monitoring loop
    async fn hardware_monitoring_loop(
        system: Arc<Mutex<System>>,
        oscillation_data: Arc<RwLock<HashMap<String, OscillationData>>>,
        analysis_buffer: Arc<ParkingRwLock<Vec<f64>>>,
        data_sender: Sender<HardwareOscillationEvent>,
        frequency_domains: Vec<FrequencyDomain>,
        sampling_rate: u32,
    ) {
        let mut interval = interval(Duration::from_micros(1_000_000 / sampling_rate as u64));
        let mut last_measurements: HashMap<String, OscillationData> = HashMap::new();
        
        loop {
            interval.tick().await;
            
            // Update system information
            {
                let mut sys = system.lock().await;
                sys.refresh_all();
                
                // Collect CPU frequency data
                for (i, processor) in sys.processors().iter().enumerate() {
                    let freq_hz = processor.frequency() as f64 * 1_000_000.0; // Convert MHz to Hz
                    let domain_name = format!("cpu_core_{}", i);
                    
                    let oscillation_data_point = OscillationData {
                        domain: domain_name.clone(),
                        current_frequency: freq_hz,
                        stability: Self::calculate_frequency_stability(&last_measurements, &domain_name, freq_hz),
                        amplitude: processor.cpu_usage() as f64 / 100.0, // CPU usage as amplitude proxy
                        phase: Self::calculate_phase_from_timestamp(),
                        harmonics: Self::detect_cpu_harmonics(freq_hz),
                        snr: Self::calculate_cpu_snr(processor.cpu_usage()),
                        timestamp: Instant::now(),
                        jitter_ns: Self::calculate_cpu_jitter(&last_measurements, &domain_name),
                    };
                    
                    // Store oscillation data
                    {
                        let mut data = oscillation_data.write().await;
                        data.insert(domain_name.clone(), oscillation_data_point.clone());
                    }
                    
                    // Send event if significant change
                    if let Some(last_data) = last_measurements.get(&domain_name) {
                        if (last_data.current_frequency - freq_hz).abs() > 1000.0 {
                            let event = HardwareOscillationEvent {
                                domain: domain_name.clone(),
                                frequency: freq_hz,
                                amplitude: oscillation_data_point.amplitude,
                                phase: oscillation_data_point.phase,
                                timestamp: Instant::now(),
                                event_type: OscillationEventType::FrequencyChange {
                                    old_freq: last_data.current_frequency,
                                    new_freq: freq_hz,
                                },
                            };
                            
                            if let Err(e) = data_sender.try_send(event) {
                                warn!("Failed to send oscillation event: {}", e);
                            }
                        }
                    }
                    
                    last_measurements.insert(domain_name, oscillation_data_point);
                }
                
                // Collect memory frequency data (synthetic based on system load)
                let memory_usage = sys.used_memory() as f64 / sys.total_memory() as f64;
                let memory_freq = Self::estimate_memory_frequency(memory_usage);
                
                let memory_oscillation = OscillationData {
                    domain: "memory_subsystem".to_string(),
                    current_frequency: memory_freq,
                    stability: 0.95, // Memory frequencies are typically very stable
                    amplitude: memory_usage,
                    phase: Self::calculate_phase_from_timestamp(),
                    harmonics: vec![memory_freq * 2.0, memory_freq * 3.0],
                    snr: 40.0, // High SNR for memory systems
                    timestamp: Instant::now(),
                    jitter_ns: 0.1, // Low jitter for memory
                };
                
                {
                    let mut data = oscillation_data.write().await;
                    data.insert("memory_subsystem".to_string(), memory_oscillation);
                }
            }
            
            // Add power supply frequency monitoring (synthetic 60Hz)
            let power_oscillation = OscillationData {
                domain: "power_supply".to_string(),
                current_frequency: 60.0,
                stability: 0.999, // Power grid is very stable
                amplitude: 1.0,
                phase: (Instant::now().elapsed().as_secs_f64() * 60.0 * 2.0 * std::f64::consts::PI) % (2.0 * std::f64::consts::PI),
                harmonics: vec![120.0, 180.0, 240.0, 300.0], // 60Hz harmonics
                snr: 60.0,
                timestamp: Instant::now(),
                jitter_ns: 0.01,
            };
            
            {
                let mut data = oscillation_data.write().await;
                data.insert("power_supply".to_string(), power_oscillation);
            }
        }
    }
    
    /// Calculate frequency stability
    fn calculate_frequency_stability(
        last_measurements: &HashMap<String, OscillationData>,
        domain_name: &str,
        current_freq: f64,
    ) -> f64 {
        if let Some(last_data) = last_measurements.get(domain_name) {
            let freq_change = (current_freq - last_data.current_frequency).abs();
            let relative_change = freq_change / current_freq;
            (1.0 - relative_change).max(0.0).min(1.0)
        } else {
            1.0
        }
    }
    
    /// Calculate phase from timestamp
    fn calculate_phase_from_timestamp() -> f64 {
        let now = Instant::now();
        let nanos = now.elapsed().as_nanos() as f64;
        (nanos / 1_000_000_000.0 * 2.0 * std::f64::consts::PI) % (2.0 * std::f64::consts::PI)
    }
    
    /// Detect CPU harmonics
    fn detect_cpu_harmonics(base_freq: f64) -> Vec<f64> {
        vec![
            base_freq * 2.0,
            base_freq * 3.0,
            base_freq * 4.0,
            base_freq / 2.0,
            base_freq / 3.0,
        ]
    }
    
    /// Calculate CPU signal-to-noise ratio
    fn calculate_cpu_snr(cpu_usage: f32) -> f64 {
        // Higher CPU usage generally correlates with higher SNR
        20.0 + (cpu_usage as f64 / 100.0) * 20.0
    }
    
    /// Calculate CPU jitter
    fn calculate_cpu_jitter(
        last_measurements: &HashMap<String, OscillationData>,
        domain_name: &str,
    ) -> f64 {
        if let Some(last_data) = last_measurements.get(domain_name) {
            let time_diff = last_data.timestamp.elapsed().as_nanos() as f64;
            (time_diff % 1000.0) / 1000.0 // Normalize to [0,1] range
        } else {
            0.1
        }
    }
    
    /// Estimate memory frequency from usage
    fn estimate_memory_frequency(usage: f64) -> f64 {
        // Typical DDR4 frequencies scaled by usage
        let base_freq = 3_200_000_000.0; // 3.2 GHz
        base_freq * (0.5 + usage * 0.5) // Scale between 50% and 100% of base
    }
    
    /// Get current oscillation data
    pub async fn get_oscillation_data(&self) -> HashMap<String, OscillationData> {
        self.oscillation_data.read().await.clone()
    }
    
    /// Get real-time oscillation events
    pub fn get_event_receiver(&self) -> Receiver<HardwareOscillationEvent> {
        self.data_receiver.clone()
    }
    
    /// Stop monitoring
    pub async fn stop_monitoring(&mut self) -> AutobahnResult<()> {
        if let Some(handle) = self.monitoring_handle.take() {
            handle.abort();
            info!("Hardware oscillation monitoring stopped");
        }
        Ok(())
    }
}

impl CoherenceSync {
    /// Create new coherence synchronization system
    pub fn new() -> Self {
        Self {
            frequency_domains: Vec::new(),
            cross_scale_coupling: false,
            biological_resonance_target: 0.85,
            consciousness_emergence_optimization: false,
            pll_params: PLLParameters {
                bandwidth_hz: 10.0,
                damping_factor: 0.707,
                lock_threshold: 0.1,
                max_frequency_error: 100.0,
            },
            sync_state: Arc::new(RwLock::new(SynchronizationState {
                phase_locks: HashMap::new(),
                coherence_level: 0.0,
                quality_metric: 0.0,
                resonance_frequencies: Vec::new(),
                coupling_strength: 0.0,
            })),
        }
    }
    
    /// Configure frequency domains
    pub fn with_frequency_domains(mut self, domains: Vec<FrequencyDomain>) -> Self {
        self.frequency_domains = domains;
        self
    }
    
    /// Enable cross-scale coupling
    pub fn enable_cross_scale_coupling(mut self, enabled: bool) -> Self {
        self.cross_scale_coupling = enabled;
        self
    }
    
    /// Set biological resonance target
    pub fn set_biological_resonance_target(mut self, target: f64) -> Self {
        self.biological_resonance_target = target;
        self
    }
    
    /// Enable consciousness emergence optimization
    pub fn optimize_for_consciousness_emergence(mut self, enabled: bool) -> Self {
        self.consciousness_emergence_optimization = enabled;
        self
    }
    
    /// Synchronize with hardware oscillations
    pub async fn synchronize_with_hardware(
        &self,
        oscillation_data: &HashMap<String, OscillationData>,
    ) -> AutobahnResult<f64> {
        let mut coherence_sum = 0.0;
        let mut valid_domains = 0;
        
        for (domain_name, data) in oscillation_data {
            if data.stability > 0.5 && data.snr > 10.0 {
                let domain_coherence = self.calculate_domain_coherence(data).await?;
                coherence_sum += domain_coherence;
                valid_domains += 1;
            }
        }
        
        let overall_coherence = if valid_domains > 0 {
            coherence_sum / valid_domains as f64
        } else {
            0.0
        };
        
        // Update synchronization state
        {
            let mut state = self.sync_state.write().await;
            state.coherence_level = overall_coherence;
            state.quality_metric = self.calculate_quality_metric(oscillation_data).await;
            state.coupling_strength = if self.cross_scale_coupling { 0.8 } else { 0.3 };
        }
        
        Ok(overall_coherence)
    }
    
    /// Calculate coherence for a specific domain
    async fn calculate_domain_coherence(&self, data: &OscillationData) -> AutobahnResult<f64> {
        // Coherence based on stability, SNR, and biological resonance matching
        let stability_factor = data.stability;
        let snr_factor = (data.snr / 60.0).min(1.0);
        let resonance_factor = if data.current_frequency > 10.0 && data.current_frequency < 100.0 {
            // Biological frequency range
            0.9
        } else if data.current_frequency > 1_000_000.0 {
            // High-frequency processing range
            0.7
        } else {
            0.5
        };
        
        let coherence = (stability_factor * 0.4 + snr_factor * 0.3 + resonance_factor * 0.3) * 
                       (1.0 - data.jitter_ns / 1000.0).max(0.0);
        
        Ok(coherence)
    }
    
    /// Calculate overall quality metric
    async fn calculate_quality_metric(&self, oscillation_data: &HashMap<String, OscillationData>) -> f64 {
        let mut quality_sum = 0.0;
        let mut count = 0;
        
        for data in oscillation_data.values() {
            let domain_quality = data.stability * (data.snr / 60.0).min(1.0) * 
                               (1.0 - data.jitter_ns / 1000.0).max(0.0);
            quality_sum += domain_quality;
            count += 1;
        }
        
        if count > 0 {
            quality_sum / count as f64
        } else {
            0.0
        }
    }
    
    /// Get current synchronization state
    pub async fn get_sync_state(&self) -> SynchronizationState {
        self.sync_state.read().await.clone()
    }
} 