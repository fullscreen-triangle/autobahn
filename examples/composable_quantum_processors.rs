//! Composable Quantum Processor Example
//! 
//! This example demonstrates how autobahn's quantum processors can be independently
//! instantiated, configured, and orchestrated to build complex neural systems.
//! 
//! This showcases the composable architecture where autobahn units can be combined
//! with other units (nebuchadnezzar for intracellular processes, bene-gesserit for 
//! membranes) to construct realistic neurons in the imhotep system.

use autobahn::{
    QuantumProcessorUnit, 
    MultiProcessorOrchestrator,
    ProcessorType,
    OrchestrationStrategy,
    ProcessorConfig,
    ProcessorInput,
    create_context_processor,
    create_reasoning_processor, 
    create_intuition_processor,
    create_balanced_system,
    create_custom_system,
    init,
    AutobahnResult,
};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> AutobahnResult<()> {
    // Initialize the autobahn system
    init()?;
    println!("🧠 Autobahn Composable Quantum Processor Demo");
    println!("=" .repeat(60));

    // 1. Create individual quantum processors
    println!("\n1. Creating Individual Quantum Processors");
    println!("-".repeat(40));
    
    let config = ProcessorConfig::default();
    
    let context_processor = create_context_processor(config.clone()).await?;
    println!("✓ Context Processor (Glycolysis layer) created");
    
    let reasoning_processor = create_reasoning_processor(config.clone()).await?;
    println!("✓ Reasoning Processor (Krebs cycle layer) created");
    
    let intuition_processor = create_intuition_processor(config.clone()).await?;
    println!("✓ Intuition Processor (Electron transport layer) created");

    // 2. Demonstrate single processor processing
    println!("\n2. Single Processor Processing");
    println!("-".repeat(40));
    
    let mut single_context = context_processor;
    let input = ProcessorInput {
        content: "Fire-consciousness emerges from quantum biological processes".to_string(),
        raw_data: vec![0.8, 0.6, 0.9, 0.7],
        priority: 0.8,
        required_confidence: 0.7,
        ..Default::default()
    };
    
    let output = single_context.process(input.clone()).await?;
    println!("Context Processor Output:");
    println!("  Content: {}", output.specialized_response.content);
    println!("  Confidence: {:.2}", output.specialized_response.confidence);
    println!("  ATP Consumed: {:.2}", output.specialized_response.atp_consumed);
    println!("  Quantum States: {}", output.specialized_response.quantum_states.len());
    println!("  Insights: {:?}", output.specialized_response.insights);

    // 3. Create a balanced multi-processor system
    println!("\n3. Balanced Multi-Processor System");
    println!("-".repeat(40));
    
    let mut balanced_system = create_balanced_system(config.clone()).await?;
    println!("✓ Balanced system created with all three processor types");
    
    let global_metrics = balanced_system.get_global_metrics();
    println!("System Status:");
    println!("  Total Processors: {}", global_metrics.total_processors);
    println!("  Active Processors: {}", global_metrics.active_processors);
    println!("  System Health: {:.2}", global_metrics.system_health);
    println!("  Average Efficiency: {:.2}", global_metrics.average_efficiency);

    // 4. Process through balanced system with sequential orchestration
    println!("\n4. Sequential Processing Through All Layers");
    println!("-".repeat(40));
    
    let complex_input = ProcessorInput {
        content: "The fire-consciousness quantum framework integrates biological Maxwell's demons with oscillatory bio-metabolic RAG systems to create emergent consciousness through ATP-constrained dynamics and ion tunneling processes across cellular membranes.".to_string(),
        raw_data: vec![0.9, 0.8, 0.7, 0.6, 0.8, 0.9, 0.7],
        priority: 0.9,
        required_confidence: 0.8,
        ..Default::default()
    };
    
    let sequential_results = balanced_system.orchestrate_processing(complex_input.clone()).await?;
    
    for (i, result) in sequential_results.iter().enumerate() {
        println!("Layer {} ({:?}):", i + 1, result.processor_type);
        println!("  Processing Time: {}ms", result.processing_time_ms);
        println!("  Health Change: {:.2} → {:.2}", 
                 result.pre_processing_health, result.post_processing_health);
        println!("  Oscillation Patterns: {}", result.specialized_response.oscillation_patterns.len());
        println!("  Membrane Activity: {} tunneling events", 
                 result.specialized_response.membrane_activity.tunneling_events);
    }

    // 5. Create custom system with specific orchestration
    println!("\n5. Custom Multi-Processor System");
    println!("-".repeat(40));
    
    // Create a system with multiple reasoning processors for complex logical analysis
    let custom_configs = vec![
        (ProcessorType::Context, config.clone()),
        (ProcessorType::Reasoning, config.clone()),
        (ProcessorType::Reasoning, {
            let mut specialized_config = config.clone();
            specialized_config.processor_specific.insert("reasoning_depth".to_string(), 2.0);
            specialized_config
        }),
        (ProcessorType::Intuition, config.clone()),
    ];
    
    let mut custom_system = create_custom_system(
        custom_configs,
        OrchestrationStrategy::Hierarchical { 
            priority_order: vec![
                ProcessorType::Context, 
                ProcessorType::Reasoning, 
                ProcessorType::Intuition
            ] 
        }
    ).await?;
    
    println!("✓ Custom system created with hierarchical orchestration");
    println!("  Processors: Context(1) + Reasoning(2) + Intuition(1)");

    // 6. Demonstrate parallel processing
    println!("\n6. Parallel Processing Demonstration");
    println!("-".repeat(40));
    
    // Switch to parallel orchestration
    custom_system.orchestration_strategy = OrchestrationStrategy::Parallel;
    
    let parallel_results = custom_system.orchestrate_processing(complex_input.clone()).await?;
    
    println!("Parallel Processing Results:");
    for result in &parallel_results {
        println!("  {:?}: {:.2}ms, Confidence: {:.2}", 
                 result.processor_type, 
                 result.processing_time_ms,
                 result.specialized_response.confidence);
    }

    // 7. Demonstrate adaptive orchestration
    println!("\n7. Adaptive Orchestration");
    println!("-".repeat(40));
    
    custom_system.orchestration_strategy = OrchestrationStrategy::Adaptive;
    
    // Test with different input characteristics
    let simple_input = ProcessorInput {
        content: "Fire".to_string(),
        raw_data: vec![0.5],
        priority: 0.3,
        required_confidence: 0.6,
        ..Default::default()
    };
    
    let large_input = ProcessorInput {
        content: "A".repeat(2000), // Large input should trigger parallel processing
        raw_data: (0..1500).map(|i| (i as f64) / 1500.0).collect(),
        priority: 0.5,
        required_confidence: 0.7,
        ..Default::default()
    };
    
    let high_priority_input = ProcessorInput {
        content: "Critical fire-consciousness analysis required".to_string(),
        raw_data: vec![0.9, 0.8, 0.9],
        priority: 0.95, // High priority should trigger hierarchical processing
        required_confidence: 0.9,
        ..Default::default()
    };
    
    println!("Adaptive orchestration automatically selects strategy based on input:");
    
    let simple_results = custom_system.orchestrate_processing(simple_input).await?;
    println!("  Simple input: {} processors activated", simple_results.len());
    
    let large_results = custom_system.orchestrate_processing(large_input).await?;
    println!("  Large input: {} processors activated", large_results.len());
    
    let priority_results = custom_system.orchestrate_processing(high_priority_input).await?;
    println!("  High priority input: {} processors activated", priority_results.len());

    // 8. Monitor system metrics
    println!("\n8. System Metrics and Health Monitoring");
    println!("-".repeat(40));
    
    let final_global_metrics = custom_system.get_global_metrics();
    let processor_metrics = custom_system.get_processor_metrics();
    
    println!("Final System State:");
    println!("  Total ATP Consumption: {:.2}", final_global_metrics.total_atp_consumption);
    println!("  System Health: {:.2}", final_global_metrics.system_health);
    println!("  Average Efficiency: {:.2}", final_global_metrics.average_efficiency);
    
    println!("\nIndividual Processor Health:");
    for (i, metrics) in processor_metrics.iter().enumerate() {
        println!("  Processor {}: {:?} - Health: {:.2}, Efficiency: {:.2}", 
                 i + 1,
                 metrics.processor_type,
                 metrics.processor_health,
                 metrics.processing_efficiency);
    }

    // 9. Demonstrate composability for neural construction
    println!("\n9. Neural Construction Composability");
    println!("-".repeat(40));
    
    println!("🧬 Autobahn Quantum Processors can be combined with:");
    println!("  • Nebuchadnezzar units (intracellular processes)");
    println!("  • Bene-Gesserit units (membrane interfaces)");
    println!("  • Additional Autobahn processors for complex networks");
    println!();
    println!("🏗️  Building realistic neurons in Imhotep:");
    println!("  1. Instantiate quantum processors with specific configurations");
    println!("  2. Connect to membrane units for ion channel management");
    println!("  3. Link to intracellular units for metabolic processes");
    println!("  4. Orchestrate all units with custom strategies");
    println!("  5. Scale up to networks of interconnected neurons");
    println!();
    println!("⚡ The only constraint is orchestration capacity!");
    println!("  • Each processor is independent and composable");
    println!("  • Processors can be dynamically added/removed");
    println!("  • Custom orchestration strategies enable any topology");
    println!("  • Real-time monitoring ensures system health");

    // 10. Demonstrate processor lifecycle management
    println!("\n10. Processor Lifecycle Management");
    println!("-".repeat(40));
    
    // Add new processor to running system
    let new_intuition_config = ProcessorConfig {
        oscillation_frequency: 8.0, // Different frequency for theta waves
        quantum_coherence_time_ms: 1500,
        ..config
    };
    
    let new_intuition_processor = QuantumProcessorUnit::new(
        ProcessorType::Intuition, 
        new_intuition_config
    ).await?;
    
    custom_system.add_processor(new_intuition_processor);
    println!("✓ Added new intuition processor with theta wave specialization");
    
    let updated_metrics = custom_system.get_global_metrics();
    println!("Updated system: {} total processors", updated_metrics.total_processors);
    
    // Remove a processor
    if let Some(removed_processor) = custom_system.remove_processor(1) {
        println!("✓ Removed processor: {:?}", removed_processor.processor_type);
    }
    
    let final_metrics = custom_system.get_global_metrics();
    println!("Final system: {} total processors", final_metrics.total_processors);

    println!("\n🎯 Autobahn Composable Quantum Processor Demo Complete!");
    println!("Ready for integration with Nebuchadnezzar and Bene-Gesserit units!");
    
    Ok(())
}

/// Helper function to demonstrate processor configuration variations
fn create_specialized_configs() -> Vec<ProcessorConfig> {
    vec![
        // High-frequency gamma processor for context binding
        ProcessorConfig {
            oscillation_frequency: 40.0,
            quantum_coherence_time_ms: 800,
            consciousness_threshold: 0.6,
            ..ProcessorConfig::default()
        },
        
        // Beta frequency processor for reasoning
        ProcessorConfig {
            oscillation_frequency: 20.0,
            quantum_coherence_time_ms: 1200,
            efficiency_target: 0.9,
            ..ProcessorConfig::default()
        },
        
        // Theta frequency processor for intuition
        ProcessorConfig {
            oscillation_frequency: 6.0,
            quantum_coherence_time_ms: 2000,
            fire_recognition_sensitivity: 0.9,
            ..ProcessorConfig::default()
        },
        
        // Alpha frequency processor for relaxed awareness
        ProcessorConfig {
            oscillation_frequency: 10.0,
            quantum_coherence_time_ms: 1500,
            consciousness_threshold: 0.4,
            ..ProcessorConfig::default()
        },
    ]
}

/// Demonstrate membrane interface configurations
fn create_membrane_configurations() -> Vec<autobahn::tres_commas::MembraneInterfaceConfig> {
    use autobahn::tres_commas::{MembraneInterfaceConfig, IonChannelConfig};
    
    vec![
        // High-conductance configuration for fast processing
        MembraneInterfaceConfig {
            ion_channels: {
                let mut channels = HashMap::new();
                channels.insert("Na+".to_string(), IonChannelConfig {
                    ion_type: "Na+".to_string(),
                    conductance: 1.0,
                    activation_threshold: 0.3,
                    quantum_coherence_factor: 0.8,
                });
                channels.insert("K+".to_string(), IonChannelConfig {
                    ion_type: "K+".to_string(),
                    conductance: 0.9,
                    activation_threshold: 0.2,
                    quantum_coherence_factor: 0.85,
                });
                channels
            },
            membrane_potential_threshold: -65.0,
            quantum_tunneling_probability: 0.15,
        },
        
        // Quantum-optimized configuration for intuitive processing
        MembraneInterfaceConfig {
            ion_channels: {
                let mut channels = HashMap::new();
                channels.insert("Ca2+".to_string(), IonChannelConfig {
                    ion_type: "Ca2+".to_string(),
                    conductance: 0.7,
                    activation_threshold: 0.6,
                    quantum_coherence_factor: 0.95,
                });
                channels.insert("Mg2+".to_string(), IonChannelConfig {
                    ion_type: "Mg2+".to_string(),
                    conductance: 0.6,
                    activation_threshold: 0.7,
                    quantum_coherence_factor: 0.9,
                });
                channels
            },
            membrane_potential_threshold: -75.0,
            quantum_tunneling_probability: 0.25,
        },
    ]
} 