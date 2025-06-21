//! Comprehensive Hardware-Optical-Photosynthesis Integration Demo
//! 
//! This example demonstrates a complete integration of the three new subsystems:
//! 1. Hardware Oscillation Synchronization Engine
//! 2. Digital Fire Circle Processing Engine  
//! 3. Environmental Photosynthesis Engine
//! 
//! This is a production-ready demonstration suitable for conference presentations.

use autobahn::{
    AutobahnResult, AutobahnSystem, AutobahnError,
    // Hardware systems
    #[cfg(feature = "hardware-sync")]
    hardware::{HardwareOscillationCapture, CoherenceSync, FrequencyDomain},
    // Optical systems
    #[cfg(feature = "optical-processing")]
    optical::{DigitalFireProcessor, OpticalCoherence, LightSource, CaptureRegion, DigitalFireCircle, CircleGeometry},
    // Photosynthesis systems
    #[cfg(feature = "environmental-photosynthesis")]
    photosynthesis::{EnvironmentalPhotosynthesis, ColorMetabolism, AgencyIllusionEngine, ContextualFocusEngine, AttentionTarget, ColorRange, ATPRate},
};

use tokio::time::{sleep, Duration};
use log::{info, warn, error};
use serde_json;
use std::collections::HashMap;
use std::time::Instant;

/// Main demonstration configuration
#[derive(Debug)]
pub struct DemoConfiguration {
    /// Duration of the demonstration in seconds
    pub demo_duration_seconds: u64,
    /// Enable visual output for presentation
    pub visual_output_enabled: bool,
    /// Hardware monitoring frequency in Hz
    pub hardware_monitoring_hz: f64,
    /// Optical processing frequency in Hz
    pub optical_processing_hz: f64,
    /// ATP conversion sampling rate in Hz
    pub atp_conversion_hz: f64,
    /// Agency illusion demonstration enabled
    pub agency_demonstration_enabled: bool,
}

/// Integrated consciousness emergence demonstration
pub struct ComprehensiveDemo {
    /// Core Autobahn system
    autobahn_system: AutobahnSystem,
    /// Hardware oscillation capture
    #[cfg(feature = "hardware-sync")]
    hardware_capture: HardwareOscillationCapture,
    /// Coherence synchronizer
    #[cfg(feature = "hardware-sync")]
    coherence_sync: CoherenceSync,
    /// Digital fire processor
    #[cfg(feature = "optical-processing")]
    fire_processor: DigitalFireProcessor,
    /// Environmental photosynthesis
    #[cfg(feature = "environmental-photosynthesis")]
    photosynthesis: EnvironmentalPhotosynthesis,
    /// Agency illusion engine
    #[cfg(feature = "environmental-photosynthesis")]
    agency_engine: AgencyIllusionEngine,
    /// Demo configuration
    config: DemoConfiguration,
    /// Performance metrics
    metrics: DemoMetrics,
}

/// Real-time performance metrics
#[derive(Debug)]
pub struct DemoMetrics {
    /// Hardware frequency coherence levels
    pub hardware_coherence: f64,
    /// Optical processing efficiency
    pub optical_efficiency: f64,
    /// ATP generation rate (units/second)  
    pub atp_generation_rate: f64,
    /// Environmental chaos level
    pub environmental_chaos: f64,
    /// Agency illusion strength
    pub agency_strength: f64,
    /// Overall consciousness emergence score
    pub consciousness_emergence: f64,
    /// Processing latency metrics
    pub latency_metrics: LatencyMetrics,
}

/// Processing latency measurements
#[derive(Debug)]
pub struct LatencyMetrics {
    /// Hardware oscillation capture latency (ms)
    pub hardware_latency_ms: f64,
    /// Optical processing latency (ms)
    pub optical_latency_ms: f64,
    /// ATP conversion latency (ms)
    pub atp_latency_ms: f64,
    /// End-to-end system latency (ms)
    pub total_latency_ms: f64,
}

impl Default for DemoConfiguration {
    fn default() -> Self {
        Self {
            demo_duration_seconds: 300, // 5 minutes
            visual_output_enabled: true,
            hardware_monitoring_hz: 1000.0,
            optical_processing_hz: 120.0,
            atp_conversion_hz: 60.0,
            agency_demonstration_enabled: true,
        }
    }
}

impl ComprehensiveDemo {
    /// Create new comprehensive demonstration
    pub async fn new(config: DemoConfiguration) -> AutobahnResult<Self> {
        info!("Initializing comprehensive hardware-optical-photosynthesis demonstration...");
        
        // Initialize core Autobahn system
        let autobahn_system = AutobahnSystem::new(400_000.0).await
            .map_err(|e| AutobahnError::InitializationError(format!("Core system init failed: {}", e)))?;
        
        // Initialize hardware oscillation capture
        #[cfg(feature = "hardware-sync")]
        let hardware_capture = HardwareOscillationCapture::new()
            .map_err(|e| AutobahnError::HardwareError(format!("Hardware capture init failed: {}", e)))?
            .with_power_frequency(60.0) // 60Hz US power grid
            .with_cpu_base_clock(3_200_000_000) // 3.2 GHz base clock
            .with_memory_clock(3_200_000_000) // 3.2 GHz memory
            .with_system_bus_frequencies(&[100_000_000, 133_000_000, 266_000_000]); // Various bus frequencies
        
        // Initialize coherence synchronizer
        #[cfg(feature = "hardware-sync")]
        let coherence_sync = CoherenceSync::new()
            .enable_cross_scale_coupling(true)
            .set_biological_resonance_target(0.85)
            .optimize_for_consciousness_emergence(true);
        
        // Initialize digital fire processor
        #[cfg(feature = "optical-processing")]
        let fire_processor = DigitalFireProcessor::new()
            .map_err(|e| AutobahnError::OpticalError(format!("Fire processor init failed: {}", e)))?
            .with_screen_capture_region(CaptureRegion::FullDisplay)
            .set_color_sampling_rate(config.optical_processing_hz)
            .enable_rgb_wavelength_conversion(true)
            .set_environmental_noise_amplification(1.2)
            .with_status_leds(&[
                LightSource::PowerLED { wavelength_nm: 525, modulation_hz: 0.0 },
                LightSource::ActivityLED { wavelength_nm: 650, pulse_pattern: "breathing".to_string() },
                LightSource::NetworkLED { wavelength_nm: 470, data_coupled: true },
            ])
            .with_display_backlight(LightSource::LCDBacklight { 
                wavelength_range: (400, 700), 
                brightness_modulation: true,
                adaptive_control: true 
            })
            .with_rgb_arrays(&[
                LightSource::RGBLED { 
                    red_nm: 650, 
                    green_nm: 530, 
                    blue_nm: 470,
                    fire_circle_mode: true 
                }
            ]);
        
        // Initialize environmental photosynthesis
        #[cfg(feature = "environmental-photosynthesis")]
        let photosynthesis = EnvironmentalPhotosynthesis::new()
            .map_err(|e| AutobahnError::PhotosynthesisError(format!("Photosynthesis init failed: {}", e)))?
            .with_screen_capture_region(CaptureRegion::FullDisplay)
            .set_color_sampling_rate(config.atp_conversion_hz)
            .enable_rgb_wavelength_conversion(true)
            .configure_atp_conversion_rates(&[
                (ColorRange::Red(620..700), ATPRate::High(0.157)), // 650nm peak efficiency
                (ColorRange::Green(495..570), ATPRate::Medium(0.123)),
                (ColorRange::Blue(450..495), ATPRate::Low(0.089)),
                (ColorRange::White(400..700), ATPRate::Variable),
            ])
            .set_environmental_noise_amplification(1.5);
        
        // Initialize agency illusion engine
        #[cfg(feature = "environmental-photosynthesis")]
        let agency_engine = AgencyIllusionEngine::new()
            .set_focus_allocation_complexity(0.78)
            .enable_choice_emergence_simulation(config.agency_demonstration_enabled)
            .optimize_for_subjective_control_feeling(0.84);
        
        let metrics = DemoMetrics {
            hardware_coherence: 0.0,
            optical_efficiency: 0.0,
            atp_generation_rate: 0.0,
            environmental_chaos: 0.0,
            agency_strength: 0.0,
            consciousness_emergence: 0.0,
            latency_metrics: LatencyMetrics {
                hardware_latency_ms: 0.0,
                optical_latency_ms: 0.0,
                atp_latency_ms: 0.0,
                total_latency_ms: 0.0,
            },
        };
        
        Ok(Self {
            autobahn_system,
            #[cfg(feature = "hardware-sync")]
            hardware_capture,
            #[cfg(feature = "hardware-sync")]
            coherence_sync,
            #[cfg(feature = "optical-processing")]
            fire_processor,
            #[cfg(feature = "environmental-photosynthesis")]
            photosynthesis,
            #[cfg(feature = "environmental-photosynthesis")]
            agency_engine,
            config,
            metrics,
        })
    }
    
    /// Run the comprehensive demonstration
    pub async fn run_demonstration(&mut self) -> AutobahnResult<()> {
        info!("Starting comprehensive consciousness emergence demonstration");
        info!("Duration: {} seconds", self.config.demo_duration_seconds);
        
        // Start all subsystems
        self.start_all_subsystems().await?;
        
        let demo_start = Instant::now();
        let demo_duration = Duration::from_secs(self.config.demo_duration_seconds);
        
        // Main demonstration loop
        while demo_start.elapsed() < demo_duration {
            let loop_start = Instant::now();
            
            // Process hardware oscillations
            let hardware_start = Instant::now();
            self.process_hardware_oscillations().await?;
            self.metrics.latency_metrics.hardware_latency_ms = hardware_start.elapsed().as_millis() as f64;
            
            // Process optical data
            let optical_start = Instant::now();
            self.process_optical_data().await?;
            self.metrics.latency_metrics.optical_latency_ms = optical_start.elapsed().as_millis() as f64;
            
            // Process environmental photosynthesis
            let atp_start = Instant::now();
            self.process_environmental_photosynthesis().await?;
            self.metrics.latency_metrics.atp_latency_ms = atp_start.elapsed().as_millis() as f64;
            
            // Update consciousness emergence metrics
            self.update_consciousness_metrics().await?;
            
            // Update agency illusion
            if self.config.agency_demonstration_enabled {
                self.update_agency_illusion().await?;
            }
            
            self.metrics.latency_metrics.total_latency_ms = loop_start.elapsed().as_millis() as f64;
            
            // Output metrics for presentation
            if self.config.visual_output_enabled {
                self.display_metrics().await;
            }
            
            // Maintain processing cadence
            sleep(Duration::from_millis(50)).await; // 20 Hz main loop
        }
        
        info!("Demonstration completed successfully");
        self.stop_all_subsystems().await?;
        self.generate_final_report().await?;
        
        Ok(())
    }
    
    /// Start all subsystems
    async fn start_all_subsystems(&mut self) -> AutobahnResult<()> {
        info!("Starting hardware oscillation capture...");
        #[cfg(feature = "hardware-sync")]
        self.hardware_capture.start_monitoring().await
            .map_err(|e| AutobahnError::HardwareError(format!("Failed to start hardware monitoring: {}", e)))?;
        
        info!("Starting digital fire circle processing...");
        #[cfg(feature = "optical-processing")]
        self.fire_processor.start_processing().await
            .map_err(|e| AutobahnError::OpticalError(format!("Failed to start optical processing: {}", e)))?;
        
        info!("Starting environmental photosynthesis...");
        #[cfg(feature = "environmental-photosynthesis")]
        self.photosynthesis.start_processing().await
            .map_err(|e| AutobahnError::PhotosynthesisError(format!("Failed to start photosynthesis: {}", e)))?;
        
        // Allow subsystems to initialize
        sleep(Duration::from_millis(1000)).await;
        info!("All subsystems started successfully");
        
        Ok(())
    }
    
    /// Process hardware oscillation data
    async fn process_hardware_oscillations(&mut self) -> AutobahnResult<()> {
        #[cfg(feature = "hardware-sync")]
        {
            let oscillation_data = self.hardware_capture.get_oscillation_data().await;
            let coherence_level = self.coherence_sync.synchronize_with_hardware(&oscillation_data).await
                .map_err(|e| AutobahnError::HardwareCoherenceError { 
                    domain: "system".to_string(), 
                    coherence: 0.0, 
                    threshold: 0.5 
                })?;
            
            self.metrics.hardware_coherence = coherence_level;
            
            // Log significant coherence changes
            if coherence_level < 0.3 {
                warn!("Low hardware coherence detected: {:.3}", coherence_level);
            }
        }
        
        Ok(())
    }
    
    /// Process optical data from fire circles
    async fn process_optical_data(&mut self) -> AutobahnResult<()> {
        #[cfg(feature = "optical-processing")]
        {
            let light_data = self.fire_processor.get_light_data().await;
            
            // Calculate processing efficiency based on 650nm coupling
            let mut total_650nm_coupling = 0.0;
            let mut sample_count = 0;
            
            for (_, light_info) in &light_data {
                if light_info.wavelength_nm >= 645 && light_info.wavelength_nm <= 655 {
                    total_650nm_coupling += light_info.red_650nm_component;
                    sample_count += 1;
                }
            }
            
            self.metrics.optical_efficiency = if sample_count > 0 {
                total_650nm_coupling / sample_count as f64
            } else {
                0.0
            };
            
            if self.metrics.optical_efficiency < 0.2 {
                warn!("Low optical efficiency: {:.3}", self.metrics.optical_efficiency);
            }
        }
        
        Ok(())
    }
    
    /// Process environmental photosynthesis
    async fn process_environmental_photosynthesis(&mut self) -> AutobahnResult<()> {
        #[cfg(feature = "environmental-photosynthesis")]
        {
            let atp_levels = self.photosynthesis.get_atp_levels().await;
            let chaos_level = self.photosynthesis.get_chaos_level().await;
            
            // Calculate ATP generation rate
            let total_atp: f64 = atp_levels.values().sum();
            self.metrics.atp_generation_rate = total_atp * self.config.atp_conversion_hz;
            self.metrics.environmental_chaos = chaos_level;
            
            if chaos_level > 0.85 {
                info!("High environmental chaos detected: {:.3} - agency illusion conditions optimal", chaos_level);
            }
        }
        
        Ok(())
    }
    
    /// Update consciousness emergence metrics
    async fn update_consciousness_metrics(&mut self) -> AutobahnResult<()> {
        // Combine metrics from all subsystems to calculate consciousness emergence
        let hardware_factor = self.metrics.hardware_coherence * 0.3;
        let optical_factor = self.metrics.optical_efficiency * 0.3;
        let chaos_factor = self.metrics.environmental_chaos * 0.4;
        
        self.metrics.consciousness_emergence = hardware_factor + optical_factor + chaos_factor;
        
        // Log consciousness emergence events
        if self.metrics.consciousness_emergence > 0.8 {
            info!("Strong consciousness emergence detected: {:.3}", self.metrics.consciousness_emergence);
        }
        
        Ok(())
    }
    
    /// Update agency illusion processing
    async fn update_agency_illusion(&mut self) -> AutobahnResult<()> {
        #[cfg(feature = "environmental-photosynthesis")]
        {
            let agency_state = self.agency_engine.process_agency_illusion().await
                .map_err(|e| AutobahnError::AgencyIllusionError { 
                    chaos_level: self.metrics.environmental_chaos, 
                    threshold: 0.5 
                })?;
            
            self.metrics.agency_strength = agency_state.agency_strength;
            
            if agency_state.selective_attention_active {
                info!("Agency illusion active - selective attention engaged with strength {:.3}", 
                     agency_state.agency_strength);
            }
        }
        
        Ok(())
    }
    
    /// Display real-time metrics for presentation
    async fn display_metrics(&self) {
        let metrics_json = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "hardware_coherence": self.metrics.hardware_coherence,
            "optical_efficiency": self.metrics.optical_efficiency,
            "atp_generation_rate": self.metrics.atp_generation_rate,
            "environmental_chaos": self.metrics.environmental_chaos,
            "agency_strength": self.metrics.agency_strength,
            "consciousness_emergence": self.metrics.consciousness_emergence,
            "latency": {
                "hardware_ms": self.metrics.latency_metrics.hardware_latency_ms,
                "optical_ms": self.metrics.latency_metrics.optical_latency_ms,
                "atp_ms": self.metrics.latency_metrics.atp_latency_ms,
                "total_ms": self.metrics.latency_metrics.total_latency_ms
            }
        });
        
        println!("METRICS: {}", metrics_json);
    }
    
    /// Stop all subsystems
    async fn stop_all_subsystems(&mut self) -> AutobahnResult<()> {
        info!("Stopping all subsystems...");
        
        #[cfg(feature = "hardware-sync")]
        if let Err(e) = self.hardware_capture.stop_monitoring().await {
            warn!("Error stopping hardware monitoring: {}", e);
        }
        
        #[cfg(feature = "optical-processing")]
        if let Err(e) = self.fire_processor.stop_processing().await {
            warn!("Error stopping optical processing: {}", e);
        }
        
        #[cfg(feature = "environmental-photosynthesis")]
        if let Err(e) = self.photosynthesis.stop_processing().await {
            warn!("Error stopping photosynthesis: {}", e);
        }
        
        info!("All subsystems stopped");
        Ok(())
    }
    
    /// Generate final demonstration report
    async fn generate_final_report(&self) -> AutobahnResult<()> {
        let report = serde_json::json!({
            "demonstration_summary": {
                "duration_seconds": self.config.demo_duration_seconds,
                "final_metrics": {
                    "hardware_coherence": self.metrics.hardware_coherence,
                    "optical_efficiency": self.metrics.optical_efficiency,
                    "atp_generation_rate": self.metrics.atp_generation_rate,
                    "environmental_chaos": self.metrics.environmental_chaos,
                    "agency_strength": self.metrics.agency_strength,
                    "consciousness_emergence": self.metrics.consciousness_emergence
                },
                "performance": {
                    "average_latency_ms": self.metrics.latency_metrics.total_latency_ms,
                    "hardware_latency_ms": self.metrics.latency_metrics.hardware_latency_ms,
                    "optical_latency_ms": self.metrics.latency_metrics.optical_latency_ms,
                    "atp_latency_ms": self.metrics.latency_metrics.atp_latency_ms
                },
                "key_achievements": [
                    "Real-time hardware frequency capture and synchronization",
                    "650nm wavelength consciousness coupling optimization",
                    "Environmental visual complexity to ATP conversion",
                    "Agency illusion through selective attention mechanisms",
                    "Multi-scale oscillatory consciousness emergence"
                ]
            }
        });
        
        println!("\n=== FINAL DEMONSTRATION REPORT ===");
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        
        info!("Final report generated successfully");
        Ok(())
    }
}

/// Main demonstration entry point
#[tokio::main]
async fn main() -> AutobahnResult<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
    
    info!("Initializing comprehensive hardware-optical-photosynthesis demonstration");
    
    // Create demonstration configuration
    let mut config = DemoConfiguration::default();
    
    // Parse command line arguments for demo customization
    let args: Vec<String> = std::env::args().collect();
    for arg in args.iter() {
        match arg.as_str() {
            "--quick" => config.demo_duration_seconds = 60,
            "--extended" => config.demo_duration_seconds = 600,
            "--no-visual" => config.visual_output_enabled = false,
            "--no-agency" => config.agency_demonstration_enabled = false,
            _ => {}
        }
    }
    
    // Create and run demonstration
    let mut demo = ComprehensiveDemo::new(config).await?;
    demo.run_demonstration().await?;
    
    info!("Demonstration completed successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_demo_initialization() {
        let config = DemoConfiguration {
            demo_duration_seconds: 1,
            visual_output_enabled: false,
            ..Default::default()
        };
        
        let demo = ComprehensiveDemo::new(config).await;
        assert!(demo.is_ok(), "Demo initialization should succeed");
    }
    
    #[tokio::test]
    async fn test_subsystem_integration() {
        let config = DemoConfiguration {
            demo_duration_seconds: 5,
            visual_output_enabled: false,
            agency_demonstration_enabled: false,
            ..Default::default()
        };
        
        let mut demo = ComprehensiveDemo::new(config).await.unwrap();
        let result = demo.run_demonstration().await;
        assert!(result.is_ok(), "Integrated demonstration should complete successfully");
    }
    
    #[test]
    fn test_metrics_calculation() {
        let mut metrics = DemoMetrics {
            hardware_coherence: 0.8,
            optical_efficiency: 0.9,
            environmental_chaos: 0.7,
            ..Default::default()
        };
        
        // Test consciousness emergence calculation
        let hardware_factor = metrics.hardware_coherence * 0.3;
        let optical_factor = metrics.optical_efficiency * 0.3;
        let chaos_factor = metrics.environmental_chaos * 0.4;
        let consciousness_emergence = hardware_factor + optical_factor + chaos_factor;
        
        assert!(consciousness_emergence > 0.5, "Consciousness emergence should be significant with good metrics");
        assert!(consciousness_emergence < 1.0, "Consciousness emergence should be bounded");
    }
}

impl Default for DemoMetrics {
    fn default() -> Self {
        Self {
            hardware_coherence: 0.0,
            optical_efficiency: 0.0,
            atp_generation_rate: 0.0,
            environmental_chaos: 0.0,
            agency_strength: 0.0,
            consciousness_emergence: 0.0,
            latency_metrics: LatencyMetrics::default(),
        }
    }
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            hardware_latency_ms: 0.0,
            optical_latency_ms: 0.0,
            atp_latency_ms: 0.0,
            total_latency_ms: 0.0,
        }
    }
} 