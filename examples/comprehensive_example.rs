//! Comprehensive example demonstrating the Oscillatory Bio-Metabolic RAG System
//! 
//! This example shows how to use the quantum-enhanced, ATP-driven RAG system
//! with biological metabolism simulation.

use autobahn::{
    oscillatory_rag::{OscillatoryRAGSystem, OscillatoryRAGConfig, OscillatoryQuery},
    AutobahnError,
};
use tokio;

#[tokio::main]
async fn main() -> Result<(), AutobahnError> {
    // Initialize logging
    env_logger::init();
    
    println!("🧬 Autobahn Oscillatory Bio-Metabolic RAG System Demo");
    println!("=" * 60);
    
    // Create system configuration
    let config = OscillatoryRAGConfig {
        temperature: 285.0, // Lower temperature for cold-blooded advantage
        target_entropy: 2.0,
        oscillation_dimensions: 64,
        base_frequency: 40.0, // Gamma brainwave frequency
        atp_regeneration_rate: 50.0,
        max_hierarchy_levels: 6,
        quantum_coupling_strength: 0.4, // Optimal coupling
        enable_adversarial_detection: true,
        champagne_threshold: 0.7,
    };
    
    // Initialize the oscillatory RAG system
    let mut rag_system = OscillatoryRAGSystem::new(config)?;
    
    println!("✅ System initialized with quantum membrane computation");
    println!("🌡️  Operating temperature: {:.1}K (cold-blooded advantage active)", 285.0);
    println!();
    
    // Test queries of varying complexity
    let test_queries = vec![
        ("What is consciousness?", 1.5),
        ("Explain quantum tunneling in biological systems", 2.5),
        ("Simple question", 0.8),
        ("How do biological membranes function as quantum computers?", 3.2),
    ];
    
    for (i, (query_text, complexity)) in test_queries.iter().enumerate() {
        println!("🔬 Query {}: \"{}\"", i + 1, query_text);
        println!("📊 Complexity: {:.1}", complexity);
        
        // Create query with custom complexity and temperature
        let mut query = OscillatoryQuery::new(query_text.to_string());
        query.complexity = *complexity;
        query.temperature = 285.0;
        query.frequency = 40.0 + complexity * 10.0; // Frequency scales with complexity
        
        // Process the query
        match rag_system.process_query(query).await {
            Ok(response) => {
                println!("✨ Response: {}", response.content);
                println!("💪 Confidence: {:.1}%", response.confidence * 100.0);
                println!("⚡ ATP Consumed: {:.1}", response.atp_consumed);
                println!("🔋 ATP Produced: {:.1}", response.atp_produced);
                println!("🌊 Oscillations: {}", response.total_oscillations);
                println!("🔄 Final Frequency: {:.1} Hz", response.final_frequency);
                println!("🧪 Quantum Enhancement: {:.2}x", response.quantum_enhancement);
                println!("🌡️  Temperature Advantage: {:.2}x", response.temperature_advantage);
                
                if response.champagne_achieved {
                    println!("🍾 CHAMPAGNE PHASE ACHIEVED!");
                }
                
                println!("📈 Entropy: {:.2} → {:.2}", response.initial_entropy, response.final_entropy);
                println!("⏱️  Processing Time: {}ms", response.metadata.processing_time_ms);
                
                if response.metadata.resonance_achieved {
                    println!("🎵 Resonance achieved during processing");
                }
                
                if response.metadata.quantum_coherence_maintained {
                    println!("⚛️  Quantum coherence maintained");
                }
                
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
        
        println!();
        
        // Show system status after each query
        let status = rag_system.get_system_status();
        println!("📊 System Status:");
        println!("   ATP Level: {:.1}", status.current_atp);
        println!("   Entropy: {:.2}", status.current_entropy);
        println!("   Frequency: {:.1} Hz", status.current_frequency);
        println!("   Queries Processed: {}", status.total_queries_processed);
        println!("   Average Confidence: {:.1}%", status.average_confidence * 100.0);
        println!("   System Ready: {}", status.system_ready);
        
        println!("─" * 60);
        
        // Small delay between queries
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
    
    // Demonstrate system limits - very high complexity query
    println!("🚀 Testing system limits with ultra-high complexity query...");
    let extreme_query = OscillatoryQuery {
        content: "Explain the complete unified theory of quantum gravity, consciousness, and biological information processing across all 10 hierarchy levels".to_string(),
        complexity: 5.0, // Very high complexity
        frequency: 100.0, // High frequency
        temperature: 310.0, // Mammalian temperature (burden)
        query_id: uuid::Uuid::new_v4(),
        timestamp: chrono::Utc::now(),
    };
    
    match rag_system.process_query(extreme_query).await {
        Ok(response) => {
            println!("✅ Successfully processed extreme query!");
            println!("💪 Confidence: {:.1}%", response.confidence * 100.0);
            println!("⚡ ATP Consumed: {:.1}", response.atp_consumed);
            println!("🧪 Quantum Enhancement: {:.2}x", response.quantum_enhancement);
            
            if response.champagne_achieved {
                println!("🍾 Incredible! Champagne phase achieved even with extreme complexity!");
            }
        }
        Err(e) => {
            println!("⚠️  System limit reached: {}", e);
        }
    }
    
    println!();
    println!("🎯 Demo completed successfully!");
    println!("📝 Key observations:");
    println!("   • Lower temperature (285K) provides quantum advantage");
    println!("   • ATP consumption scales with query complexity");
    println!("   • Resonance enhances processing efficiency");
    println!("   • Quantum enhancement improves with optimal coupling");
    println!("   • Champagne phase indicates high-quality processing");
    
    Ok(())
} 