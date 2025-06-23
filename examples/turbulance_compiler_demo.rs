/// Complete Turbulance Compiler System Demo
/// 
/// This example demonstrates the complete Turbulance language compiler
/// and its integration with autobahn's biological computing systems.

use autobahn::turbulance::{TurbulanceProcessor, TurbulanceValue};
use autobahn::turbulance::integration::{AutobahnTurbulanceIntegration, BMDIntegration, QuantumIntegration};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Turbulance Scientific Method Compiler Demo");
    println!("============================================\n");
    
    // Initialize the integrated Turbulance system
    let mut integration_system = AutobahnTurbulanceIntegration::new();
    
    // Example 1: Basic Scientific Proposition
    demonstrate_scientific_proposition(&mut integration_system)?;
    
    // Example 2: Biological Maxwell's Demon Integration
    demonstrate_bmd_integration(&mut integration_system)?;
    
    // Example 3: Goal-Oriented Scientific Computing
    demonstrate_goal_system(&mut integration_system)?;
    
    // Example 4: Complex Pattern-Based Analysis
    demonstrate_pattern_analysis(&mut integration_system)?;
    
    Ok(())
}

fn demonstrate_scientific_proposition(integration: &mut AutobahnTurbulanceIntegration) -> Result<(), Box<dyn std::error::Error>> {
    println!("📋 Example 1: Scientific Method Encoding");
    println!("-----------------------------------------");
    
    let turbulance_code = r#"
        # Define a scientific proposition about energy efficiency
        proposition EnergyOptimization:
            motion HighEfficiency("System achieves >90% energy conversion efficiency")
            motion StableOperation("System maintains stable operation under varying loads")
            motion ThermodynamicCompliance("All operations respect thermodynamic laws")
            
            # Evidence collection from autobahn biological systems
            evidence EfficiencyMeasurement from "biological_sensors":
                collect energy_conversion_data
                validate thermodynamic_consistency
            
            # Support conditions based on evidence
            given efficiency_measurement > 0.9:
                support HighEfficiency with_weight(0.95)
            
            given stability_variance < 0.05:
                support StableOperation with_weight(0.8)
            
            given entropy_change >= 0:
                support ThermodynamicCompliance with_weight(1.0)
        
        # Evaluate the proposition
        evaluate EnergyOptimization
        print("Energy optimization proposition evaluated")
    "#;
    
    let result = integration.execute_with_integration(turbulance_code)?;
    
    println!("✅ Proposition evaluation completed:");
    println!("   Success: {}", result.turbulance_result.success);
    println!("   Side effects: {} recorded", result.turbulance_result.side_effects.len());
    println!("   Evidence generated: {} pieces", result.turbulance_result.evidence_generated.len());
    println!("   Integration effects: {}", result.integration_effects.len());
    println!();
    
    Ok(())
}

fn demonstrate_bmd_integration(integration: &mut AutobahnTurbulanceIntegration) -> Result<(), Box<dyn std::error::Error>> {
    println!("🧬 Example 2: Biological Maxwell's Demon Integration");
    println!("----------------------------------------------------");
    
    // Register a BMD for Turbulance control
    let bmd_config = BMDIntegration {
        demon_id: "metabolic_demon_01".to_string(),
        turbulance_proposition: "MetabolicEfficiency".to_string(),
        energy_threshold: 2.5,
        information_extraction_rate: 1.8,
        active_goals: vec!["optimize_atp_production".to_string()],
    };
    
    integration.register_bmd_integration(bmd_config)?;
    
    // Create biological proposition for the BMD
    integration.create_biological_proposition("MetabolicEfficiency", "atp_synthesis_pathway")?;
    
    let turbulance_code = r#"
        # Control biological Maxwell's demon through Turbulance
        funxn optimize_metabolic_demon():
            # Process glucose molecule
            item energy_yield = process_molecule("glucose")
            
            # Harvest available energy
            item harvested = harvest_energy("glycolysis_pathway")
            
            # Extract information from the process
            item information = extract_information("metabolic_state")
            
            # Update membrane interface based on results
            given energy_yield > 2.0:
                update_membrane_state("high_permeability")
            otherwise:
                update_membrane_state("regulated_transport")
            
            # Metacognitive monitoring
            metacognitive MetabolicMonitor:
                track_reasoning("energy_optimization")
                evaluate_confidence()
                detect_bias("efficiency_bias")
            
            return [energy_yield, harvested, information]
        
        # Execute optimization
        item results = optimize_metabolic_demon()
        print("BMD optimization results: " + str(results))
    "#;
    
    let result = integration.execute_with_integration(turbulance_code)?;
    
    println!("✅ BMD integration completed:");
    println!("   Success: {}", result.turbulance_result.success);
    println!("   Goals updated: {}", result.turbulance_result.goals_updated.len());
    println!("   System updates: {}", result.system_updates.len());
    
    for update in result.system_updates.iter().take(3) {
        println!("   • {} - {}: {}", update.system_type, update.update_type, update.data);
    }
    println!();
    
    Ok(())
}

fn demonstrate_goal_system(integration: &mut AutobahnTurbulanceIntegration) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 Example 3: Goal-Oriented Scientific Computing");
    println!("-------------------------------------------------");
    
    // Create optimization goal
    integration.create_optimization_goal("system_optimization", 0.95)?;
    
    let turbulance_code = r#"
        # Goal-driven system optimization
        goal SystemPerformance:
            description: "Achieve optimal system performance across all metrics"
            success_threshold: 0.95
            metrics:
                energy_efficiency: 0.0
                processing_speed: 0.0
                information_quality: 0.0
        
        # Iterative optimization function
        funxn iterative_optimization(iterations):
            item current_performance = 0.0
            item step = 0
            
            while step < iterations and current_performance < 0.95:
                # Measure current system state
                item efficiency = harvest_energy("system_core") / 10.0
                item speed = process_molecule("test_substrate") / 5.0
                item quality = extract_information("system_state") / 2.0
                
                # Calculate overall performance
                current_performance = (efficiency + speed + quality) / 3.0
                
                # Update goal progress
                update_goal_progress("SystemPerformance", current_performance)
                
                # Adaptive behavior based on performance
                given current_performance < 0.7:
                    adapt_behavior("increase_resources")
                given current_performance > 0.9:
                    adapt_behavior("maintain_efficiency")
                
                step = step + 1
            
            # Evaluate final goal achievement
            item goal_achieved = evaluate_goal("SystemPerformance")
            return [current_performance, goal_achieved, step]
        
        # Run optimization
        item optimization_results = iterative_optimization(10)
        print("Optimization completed: " + str(optimization_results))
    "#;
    
    let result = integration.execute_with_integration(turbulance_code)?;
    
    println!("✅ Goal-oriented optimization completed:");
    println!("   Success: {}", result.turbulance_result.success);
    
    if let Some(return_value) = &result.turbulance_result.return_value {
        println!("   Return value: {:?}", return_value);
    }
    
    println!("   Goals updated: {}", result.turbulance_result.goals_updated.len());
    for goal_update in &result.turbulance_result.goals_updated {
        println!("   • Goal '{}': {} → {}", 
                 goal_update.goal_id, 
                 goal_update.previous_progress, 
                 goal_update.new_progress);
    }
    println!();
    
    Ok(())
}

fn demonstrate_pattern_analysis(integration: &mut AutobahnTurbulanceIntegration) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Example 4: Pattern-Based Scientific Analysis");
    println!("-----------------------------------------------");
    
    let turbulance_code = r#"
        # Pattern-based analysis of biological systems
        funxn analyze_biological_patterns():
            # Define pattern templates
            item efficiency_pattern = pattern("high_efficiency", oscillatory)
            item stability_pattern = pattern("stable_operation", temporal)
            item anomaly_pattern = pattern("anomalous_behavior", emergent)
            
            # Collect data from multiple sources
            item energy_data = []
            item stability_data = []
            
            # Simulate data collection
            for i in range(5):
                energy_data.append(harvest_energy("sensor_" + str(i)))
                stability_data.append(extract_information("stability_" + str(i)))
            
            # Pattern matching analysis
            within "pattern_analysis_scope":
                item efficiency_matches = energy_data matches efficiency_pattern
                item stability_matches = stability_data matches stability_pattern
                
                # Evidence-based reasoning
                evidence PatternEvidence from "analysis_results":
                    collect efficiency_matches
                    collect stability_matches
                
                # Proposition about system behavior
                proposition SystemBehavior:
                    motion PredictableOperation("System exhibits predictable operational patterns")
                    motion HighReliability("System maintains high reliability across conditions")
                    
                    given efficiency_matches == true:
                        support PredictableOperation with_weight(0.9)
                    
                    given stability_matches == true:
                        support HighReliability with_weight(0.85)
                
                # Evaluate the behavioral proposition
                item behavior_support = evaluate SystemBehavior
                
                # Metacognitive reflection on the analysis
                metacognitive AnalysisReflection:
                    track_reasoning("pattern_analysis")
                    confidence = evaluate_confidence()
                    bias_detected = detect_bias("pattern_bias")
                    
                    given confidence < 0.7:
                        adapt_behavior("increase_data_collection")
            
            return [efficiency_matches, stability_matches, behavior_support]
        
        # Execute pattern analysis
        item analysis_results = analyze_biological_patterns()
        print("Pattern analysis completed: " + str(analysis_results))
        
        # Final system assessment
        proposition FinalAssessment:
            motion SystemReadiness("System is ready for deployment")
            
            given analysis_results[2] > 0.8:  # High behavioral support
                support SystemReadiness with_weight(1.0)
        
        evaluate FinalAssessment
        print("Final system assessment complete")
    "#;
    
    let result = integration.execute_with_integration(turbulance_code)?;
    
    println!("✅ Pattern analysis completed:");
    println!("   Success: {}", result.turbulance_result.success);
    println!("   Evidence pieces: {}", result.turbulance_result.evidence_generated.len());
    println!("   Propositions evaluated: {}", result.turbulance_result.propositions_evaluated.len());
    
    for prop_eval in &result.turbulance_result.propositions_evaluated {
        println!("   • Proposition '{}': support={:.2}, confidence={:.2}", 
                 prop_eval.name, 
                 prop_eval.overall_support, 
                 prop_eval.confidence);
        
        for motion in &prop_eval.motions {
            println!("     - Motion '{}': support={:.2}", motion.name, motion.support_level);
        }
    }
    
    println!();
    println!("🎉 Turbulance Scientific Computing Demo Complete!");
    println!("The system successfully demonstrated:");
    println!("• Scientific method encoding in programming language");
    println!("• Integration with biological Maxwell's demons");
    println!("• Goal-oriented computational optimization");
    println!("• Pattern-based analysis with metacognitive monitoring");
    println!("• Evidence-based reasoning and proposition evaluation");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_turbulance_basic_functionality() {
        let mut processor = TurbulanceProcessor::new();
        
        let simple_code = r#"
            item x = 5
            item y = 10
            item result = x + y
            print("Result: " + str(result))
        "#;
        
        let result = processor.process_turbulance_code(simple_code);
        assert!(result.success, "Basic Turbulance code should execute successfully");
    }
    
    #[test]
    fn test_scientific_proposition_creation() {
        let mut integration = AutobahnTurbulanceIntegration::new();
        
        let result = integration.create_biological_proposition(
            "TestProposition", 
            "test_context"
        );
        
        assert!(result.is_ok(), "Should be able to create biological propositions");
    }
    
    #[test]
    fn test_bmd_integration_registration() {
        let mut integration = AutobahnTurbulanceIntegration::new();
        
        let bmd_config = BMDIntegration {
            demon_id: "test_demon".to_string(),
            turbulance_proposition: "TestProp".to_string(),
            energy_threshold: 1.0,
            information_extraction_rate: 0.5,
            active_goals: vec!["test_goal".to_string()],
        };
        
        let result = integration.register_bmd_integration(bmd_config);
        assert!(result.is_ok(), "Should be able to register BMD integration");
    }
} 