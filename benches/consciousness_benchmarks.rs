use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;

use autobahn::{
    consciousness::{FireConsciousnessEngine, ConsciousnessLevel},
    rag::{OscillatoryBioMetabolicRAG, SystemConfiguration},
    tres_commas::{CategoricalPredeterminismEngine, ThermodynamicNecessityAnalysis},
    biological::{BiologicalProcessor, BiologicalLayer},
    AutobahnSystem,
};

/// Benchmark fire consciousness processing
fn bench_fire_consciousness(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_group("fire_consciousness")
        .bench_function("single_fire_pattern", |b| {
            let mut engine = rt.block_on(async {
                FireConsciousnessEngine::new(0.5).unwrap()
            });
            
            let fire_pattern = "A warm orange glow dances in the darkness, casting flickering shadows on the cave walls.";
            
            b.to_async(&rt).iter(|| async {
                let result = engine.process_input(black_box(fire_pattern)).await.unwrap();
                black_box(result)
            });
        })
        .bench_function("complex_fire_scenario", |b| {
            let mut engine = rt.block_on(async {
                FireConsciousnessEngine::new(0.5).unwrap()
            });
            
            let complex_scenario = "The campfire crackles as sparks rise into the star-filled night. \
                                   Around the fire, faces emerge from shadow as the flames dance higher. \
                                   The warmth spreads outward, creating a circle of light and consciousness \
                                   in the vast darkness of the prehistoric world.";
            
            b.to_async(&rt).iter(|| async {
                let result = engine.process_input(black_box(complex_scenario)).await.unwrap();
                black_box(result)
            });
        })
        .bench_function("underwater_fireplace_paradox", |b| {
            let mut engine = rt.block_on(async {
                FireConsciousnessEngine::new(0.5).unwrap()
            });
            
            let paradox_input = "There is a fireplace burning underwater at the bottom of the ocean.";
            
            b.to_async(&rt).iter(|| async {
                let result = engine.process_input(black_box(paradox_input)).await.unwrap();
                black_box(result)
            });
        });
}

/// Benchmark oscillatory bio-metabolic RAG processing
fn bench_rag_system(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_group("rag_system")
        .bench_function("simple_query", |b| {
            let mut rag = rt.block_on(async {
                OscillatoryBioMetabolicRAG::new().await.unwrap()
            });
            
            let query = "What is consciousness?";
            
            b.to_async(&rt).iter(|| async {
                let result = rag.process_query(black_box(query)).await.unwrap();
                black_box(result)
            });
        })
        .bench_function("complex_biological_query", |b| {
            let mut rag = rt.block_on(async {
                OscillatoryBioMetabolicRAG::new().await.unwrap()
            });
            
            let query = "How do ion channels in biological membranes contribute to quantum coherence \
                        effects that enable consciousness emergence through cross-scale oscillatory \
                        coupling between molecular and cognitive hierarchy levels?";
            
            b.to_async(&rt).iter(|| async {
                let result = rag.process_query(black_box(query)).await.unwrap();
                black_box(result)
            });
        });
}

/// Benchmark categorical predeterminism engine
fn bench_predeterminism_engine(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_group("predeterminism")
        .bench_function("thermodynamic_necessity_analysis", |b| {
            let mut engine = rt.block_on(async {
                CategoricalPredeterminismEngine::new().await.unwrap()
            });
            
            let event = "The emergence of consciousness in biological systems";
            
            b.to_async(&rt).iter(|| async {
                let result = engine.analyze_thermodynamic_necessity(black_box(event)).await.unwrap();
                black_box(result)
            });
        })
        .bench_function("configuration_space_exploration", |b| {
            let mut engine = rt.block_on(async {
                CategoricalPredeterminismEngine::new().await.unwrap()
            });
            
            let dimensions = vec![0.5, 0.7, 0.3, 0.9, 0.1, 0.8, 0.4, 0.6, 0.2, 0.95];
            
            b.to_async(&rt).iter(|| async {
                let result = engine.explore_configuration_space(black_box(&dimensions)).await.unwrap();
                black_box(result)
            });
        })
        .bench_function("heat_death_trajectory", |b| {
            let mut engine = rt.block_on(async {
                CategoricalPredeterminismEngine::new().await.unwrap()
            });
            
            let current_entropy = 0.65;
            let time_horizon = 1000.0;
            
            b.to_async(&rt).iter(|| async {
                let result = engine.calculate_heat_death_trajectory(
                    black_box(current_entropy), 
                    black_box(time_horizon)
                ).await.unwrap();
                black_box(result)
            });
        });
}

/// Benchmark biological processing layers
fn bench_biological_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_group("biological_processing")
        .bench_function("context_layer", |b| {
            let processor = rt.block_on(async {
                BiologicalProcessor::new().await.unwrap()
            });
            
            let input = "Environmental assessment of fire-lit cave dwelling";
            
            b.to_async(&rt).iter(|| async {
                let result = processor.process_context_layer(black_box(input)).await.unwrap();
                black_box(result)
            });
        })
        .bench_function("reasoning_layer", |b| {
            let processor = rt.block_on(async {
                BiologicalProcessor::new().await.unwrap()
            });
            
            let context = "Fire provides warmth and light";
            let reasoning_input = "Should we maintain the fire through the night?";
            
            b.to_async(&rt).iter(|| async {
                let result = processor.process_reasoning_layer(
                    black_box(context), 
                    black_box(reasoning_input)
                ).await.unwrap();
                black_box(result)
            });
        })
        .bench_function("intuition_layer", |b| {
            let processor = rt.block_on(async {
                BiologicalProcessor::new().await.unwrap()
            });
            
            let context = "Approaching predator detected";
            let reasoning = "Fire may provide protection";
            let intuition_input = "Immediate action required";
            
            b.to_async(&rt).iter(|| async {
                let result = processor.process_intuition_layer(
                    black_box(context),
                    black_box(reasoning),
                    black_box(intuition_input)
                ).await.unwrap();
                black_box(result)
            });
        })
        .bench_function("full_three_layer_processing", |b| {
            let processor = rt.block_on(async {
                BiologicalProcessor::new().await.unwrap()
            });
            
            let input = "Complex social situation requiring fire circle decision making";
            
            b.to_async(&rt).iter(|| async {
                let result = processor.process_through_layers(
                    black_box(input), 
                    BiologicalLayer::All
                ).await.unwrap();
                black_box(result)
            });
        });
}

/// Benchmark full system integration
fn bench_full_system(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_group("full_system")
        .bench_function("complete_processing_pipeline", |b| {
            let mut system = rt.block_on(async {
                AutobahnSystem::new(0.5).await.unwrap()
            });
            
            let input = "The fire burns bright as consciousness emerges from the quantum \
                        coherence of biological membranes, creating a thermodynamically \
                        necessary pattern of information integration across oscillatory \
                        hierarchy levels.";
            
            b.to_async(&rt).iter(|| async {
                let result = system.process_input(black_box(input)).await.unwrap();
                black_box(result)
            });
        });
}

/// Benchmark different system configurations
fn bench_system_configurations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("system_configurations");
    
    // Test different evolutionary timeline configurations
    for timeline in [0.1, 0.5, 1.0, 2.0, 3.0].iter() {
        group.bench_with_input(
            BenchmarkId::new("evolutionary_timeline", timeline),
            timeline,
            |b, &timeline| {
                b.to_async(&rt).iter(|| async {
                    let mut system = AutobahnSystem::new(timeline).await.unwrap();
                    let input = "Fire consciousness emergence test";
                    let result = system.process_input(black_box(input)).await.unwrap();
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_group("memory_usage")
        .bench_function("large_dataset_processing", |b| {
            let mut system = rt.block_on(async {
                AutobahnSystem::new(0.5).await.unwrap()
            });
            
            // Create a large input to test memory efficiency
            let large_input = "consciousness emergence ".repeat(1000);
            
            b.to_async(&rt).iter(|| async {
                let result = system.process_input(black_box(&large_input)).await.unwrap();
                black_box(result)
            });
        })
        .bench_function("concurrent_processing", |b| {
            b.to_async(&rt).iter(|| async {
                let mut handles = Vec::new();
                
                for i in 0..10 {
                    let mut system = AutobahnSystem::new(0.5).await.unwrap();
                    let input = format!("Concurrent consciousness test {}", i);
                    
                    let handle = tokio::spawn(async move {
                        system.process_input(&input).await.unwrap()
                    });
                    handles.push(handle);
                }
                
                let results = futures::future::join_all(handles).await;
                black_box(results)
            });
        });
}

criterion_group!(
    benches,
    bench_fire_consciousness,
    bench_rag_system,
    bench_predeterminism_engine,
    bench_biological_processing,
    bench_full_system,
    bench_system_configurations,
    bench_memory_usage
);

criterion_main!(benches); 