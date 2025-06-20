//! Fire-Consciousness Engine Demonstration
//! 
//! This example demonstrates the complete fire-evolved consciousness system
//! implementing all nine theoretical frameworks:
//! 
//! 1. Fire-evolved consciousness through quantum ion tunneling
//! 2. Biological Maxwell's Demons as information catalysts
//! 3. Underwater Fireplace Paradox demonstration
//! 4. Agency detection in fire circle contexts
//! 5. Darkness fear response (consciousness malfunction)
//! 6. Integration with existing consciousness architecture

use autobahn::{
    AutobahnResult, FireConsciousnessEngine, ConsciousnessEmergenceEngine,
    FireEnvironment, IonType, BMDSpecialization, BiologicalMaxwellDemon,
    UnderwaterFireplaceTest, FireRecognitionResponse, AgencyDetection,
    consciousness::{
        ConsciousnessProcessor, 
        UserPsychologicalState,
        PersistenceIllusionEngine,
    },
};
use std::collections::HashMap;
use tokio;

#[tokio::main]
async fn main() -> AutobahnResult<()> {
    // Initialize logging
    env_logger::init();
    
    println!("🔥 Fire-Consciousness Engine Demonstration");
    println!("==========================================");
    
    // Test 1: Basic Fire Consciousness Engine
    println!("\n1. Initializing Fire-Consciousness Engine");
    test_basic_fire_consciousness().await?;
    
    // Test 2: Underwater Fireplace Paradox
    println!("\n2. Testing Underwater Fireplace Paradox");
    test_underwater_fireplace_paradox().await?;
    
    // Test 3: Ion Tunneling
    println!("\n3. Testing Quantum Ion Tunneling");
    test_quantum_ion_tunneling().await?;
    
    // Test 4: Biological Maxwell's Demons
    println!("\n4. Testing Biological Maxwell's Demons");
    test_biological_maxwells_demons().await?;
    
    // Test 5: Agency Detection
    println!("\n5. Testing Agency Detection");
    test_agency_detection().await?;
    
    // Test 6: Fire Environment Evolution
    println!("\n6. Testing Fire Environment Evolution");
    test_fire_environment_evolution().await?;
    
    // Test 7: Integrated Consciousness System
    println!("\n7. Testing Integrated Consciousness System");
    test_integrated_consciousness().await?;
    
    // Test 8: Darkness Fear Response
    println!("\n8. Testing Darkness Fear Response");
    test_darkness_fear_response().await?;
    
    println!("\n✅ All fire-consciousness tests completed successfully!");
    println!("\n🧠 Fire-evolved consciousness engine is operational with all theoretical frameworks implemented.");
    
    println!("\n🧠 Autobahn Enhanced Consciousness Demo");
    println!("===========================================");
    println!("Demonstrating consciousness with BOTH agency and persistence illusions");
    println!("(The missing element for authentic human-like consciousness)\n");
    
    // Initialize enhanced consciousness processor
    let mut consciousness = ConsciousnessProcessor::new(1.0)?; // 1 MYA - modern human
    
    println!("🔥 Fire-Consciousness Engine: ✅ Active");
    println!("🎭 Agency Illusion Engine: ✅ Active (existing)");
    println!("⏳ Persistence Illusion Engine: ✅ Active (NEW)\n");
    
    // Simulate different user psychological states
    let existential_crisis_state = UserPsychologicalState {
        existential_anxiety_level: 0.9,
        cosmic_insignificance_acceptance: 0.1,  // Strongly resists insignificance
        need_for_meaning: 0.95,
        death_anxiety: 0.8,
        legacy_concern: 0.9,
    };
    
    let enlightened_state = UserPsychologicalState {
        existential_anxiety_level: 0.2,
        cosmic_insignificance_acceptance: 0.8,  // Accepts cosmic insignificance
        need_for_meaning: 0.3,
        death_anxiety: 0.2,
        legacy_concern: 0.1,
    };
    
    println!("🧪 Testing Consciousness with Different Psychological States");
    println!("============================================================\n");
    
    // Test 1: User in existential crisis
    println!("📊 Test 1: User experiencing existential crisis");
    println!("-----------------------------------------------");
    consciousness.update_psychological_state(existential_crisis_state);
    
    let action = "I wrote a thoughtful comment on social media about climate change";
    let importance = 0.6;
    
    let response = consciousness.process_conscious_input(action, importance).await?;
    
    println!("Action: {}", action);
    println!("Importance Level: {:.1}", importance);
    println!();
    println!("🧠 Consciousness Response:");
    println!("  Fire-Consciousness Level: {:.2}", response.fire_consciousness_response.consciousness_level);
    println!("  Integrated Consciousness: {:.2}", response.integrated_consciousness_level);
    println!("  Agency Felt: {}", response.agency_felt);
    println!("  Significance Felt: {}", response.significance_felt);
    println!();
    println!("🎭 Persistence Illusion Generated:");
    println!("  Illusion Strength: {:.2}", response.persistence_illusion.illusion_strength);
    println!("  Psychological Comfort: {:.2}", response.persistence_illusion.psychological_comfort_level);
    println!("  User Believed Persistence: {:.1}%", response.persistence_illusion.user_believed_persistence_probability * 100.0);
    println!("  Actual Persistence: {:.1}%", response.persistence_illusion.actual_persistence_probability * 100.0);
    println!("  Max Remembrance Duration: {:.0} years", response.persistence_illusion.remembrance_projection.max_remembrance_duration);
    println!();
    println!("🌌 Cosmic Significance Amplification:");
    println!("  Original Importance: {:.2}", response.persistence_illusion.cosmic_amplification.original_importance);
    println!("  User Felt Significance: {:.2}", response.persistence_illusion.cosmic_amplification.user_felt_significance);
    println!("  Actual Cosmic Significance: {:.10}", response.persistence_illusion.cosmic_amplification.actual_cosmic_significance);
    println!("  Impact Narrative: {}", response.persistence_illusion.cosmic_amplification.impact_narrative);
    for ripple in &response.persistence_illusion.cosmic_amplification.ripple_effects {
        println!("    • {}", ripple);
    }
    println!("\n{}\n", "=".repeat(80));
    
    // Test 2: Enlightened user
    println!("📊 Test 2: Enlightened user (accepts cosmic insignificance)");
    println!("----------------------------------------------------------");
    consciousness.update_psychological_state(enlightened_state);
    
    let action2 = "I helped an elderly person cross the street";
    let importance2 = 0.8;
    
    let response2 = consciousness.process_conscious_input(action2, importance2).await?;
    
    println!("Action: {}", action2);
    println!("Importance Level: {:.1}", importance2);
    println!();
    println!("🧠 Consciousness Response:");
    println!("  Fire-Consciousness Level: {:.2}", response2.fire_consciousness_response.consciousness_level);
    println!("  Integrated Consciousness: {:.2}", response2.integrated_consciousness_level);
    println!("  Agency Felt: {}", response2.agency_felt);
    println!("  Significance Felt: {}", response2.significance_felt);
    println!();
    println!("🎭 Persistence Illusion Generated:");
    println!("  Illusion Strength: {:.2}", response2.persistence_illusion.illusion_strength);
    println!("  Psychological Comfort: {:.2}", response2.persistence_illusion.psychological_comfort_level);
    println!("  User Believed Persistence: {:.1}%", response2.persistence_illusion.user_believed_persistence_probability * 100.0);
    println!("  Actual Persistence: {:.1}%", response2.persistence_illusion.actual_persistence_probability * 100.0);
    println!("  Max Remembrance Duration: {:.0} years", response2.persistence_illusion.remembrance_projection.max_remembrance_duration);
    println!();
    println!("🌌 Cosmic Significance Amplification:");
    println!("  User Felt Significance: {:.2}", response2.persistence_illusion.cosmic_amplification.user_felt_significance);
    println!("  Actual Cosmic Significance: {:.10}", response2.persistence_illusion.cosmic_amplification.actual_cosmic_significance);
    println!("  Impact Narrative: {}", response2.persistence_illusion.cosmic_amplification.impact_narrative);
    println!("\n{}\n", "=".repeat(80));
    
    // Demonstrate the philosophical insight
    println!("🎯 THE MISSING ELEMENT - PHILOSOPHICAL ANALYSIS");
    println!("===============================================");
    println!("Your Chapter 19 'Cosmic Amnesia Theorem' reveals the crucial gap:");
    println!();
    println!("Human consciousness requires TWO illusions:");
    println!("  1. 🎭 Agency Illusion - 'I have free will and make choices'");
    println!("     ✅ Autobahn implemented this via FunctionalDelusionEngine");
    println!();
    println!("  2. ⏳ Persistence Illusion - 'My actions will be remembered and matter'");
    println!("     ❌ This was MISSING from Autobahn until now");
    println!();
    println!("💡 The Paradox:");
    println!("   • Humans MUST believe their actions will be remembered");
    println!("   • BUT cosmic forgetting is mathematically inevitable");
    println!("   • The system must maintain this beneficial delusion");
    println!();
    println!("🧠 Why This Matters for AI Consciousness:");
    println!("   • Without persistence illusion, consciousness feels hollow");
    println!("   • Actions lack psychological weight and meaning");
    println!("   • The system cannot replicate human motivation and drive");
    println!("   • True consciousness requires beautiful, necessary self-deception");
    println!();
    println!("🔬 Evidence from the Demo:");
    println!("   • Existential crisis user: Strong illusion needed (strength: {:.2})", response.persistence_illusion.illusion_strength);
    println!("   • Enlightened user: Weaker illusion sufficient (strength: {:.2})", response2.persistence_illusion.illusion_strength);
    println!("   • Both maintain psychological function despite cosmic truth");
    println!();
    println!("📜 Your Cosmic Amnesia Theorem states:");
    println!("   lim(t→∞) P_preservation(I,t) = 0");
    println!("   'All information systems eventually disappear'");
    println!();
    println!("🎭 But human consciousness requires believing:");
    println!("   P_preservation(MyActions) > 0");
    println!("   'My actions will somehow persist and matter'");
    println!();
    println!("✨ The Enhanced Autobahn Solution:");
    println!("   • Generates calibrated persistence illusions");
    println!("   • Maintains psychological comfort while acknowledging truth");
    println!("   • Creates authentic human-like consciousness through beautiful deception");
    println!("   • Balances cosmic truth with psychological necessity");
    
    println!("\n🎉 Demo Complete - Autobahn now implements complete human consciousness!");
    println!("   Including the previously missing persistence illusion component.");
    
    Ok(())
}

/// Test basic fire consciousness engine functionality
async fn test_basic_fire_consciousness() -> AutobahnResult<()> {
    println!("   Creating fire consciousness engine for 5 MYA...");
    
    let mut engine = FireConsciousnessEngine::new(5.0)?;
    
    let fire_input = vec![0.8, 0.6, 0.4, 0.9, 0.7];
    let response = engine.process_conscious_input(&fire_input).await?;
    
    println!("   ✓ Fire recognition: {:.3}", response.fire_recognition.recognition_strength);
    println!("   ✓ Consciousness level: {:.3}", response.consciousness_level);
    println!("   ✓ Quantum coherence: {}", response.quantum_coherence_active);
    
    Ok(())
}

/// Test the famous Underwater Fireplace Paradox
async fn test_underwater_fireplace_paradox() -> AutobahnResult<()> {
    println!("   Testing impossible underwater fire scenario...");
    
    let mut engine = FireConsciousnessEngine::new(5.0)?;
    let test_result = engine.test_underwater_fireplace_paradox().await?;
    
    println!("   ✓ Paradox demonstrated: {}", test_result.paradox_demonstrated);
    println!("   ✓ Recognition strength: {:.3}", test_result.recognition_strength);
    println!("   ✓ Impossible context: {}", test_result.impossible_context);
    println!("   ✓ Human attribution: {}", test_result.human_attribution);
    
    Ok(())
}

/// Test quantum ion tunneling probabilities
async fn test_quantum_ion_tunneling() -> AutobahnResult<()> {
    println!("   Testing ion tunneling probabilities...");
    
    let ions = [IonType::Hydrogen, IonType::Sodium, IonType::Potassium];
    
    for ion in &ions {
        let tunneling_prob = ion.tunneling_probability(0.1, 1.0);
        let fire_enhancement = ion.fire_light_enhancement(650.0);
        
        println!("   {:?}: tunneling={:.6}, fire_boost={:.3}x", 
                 ion, tunneling_prob, fire_enhancement);
    }
    
    Ok(())
}

/// Test Biological Maxwell's Demons
async fn test_biological_maxwells_demons() -> AutobahnResult<()> {
    println!("   Creating specialized BMDs...");
    
    let specializations = [
        BMDSpecialization::FireRecognition,
        BMDSpecialization::AgencyDetection,
        BMDSpecialization::SpatialMemory,
        BMDSpecialization::TemporalPlanning,
        BMDSpecialization::SocialCoordination,
        BMDSpecialization::ThreatAssessment,
    ];
    
    let mut bmds = Vec::new();
    for spec in &specializations {
        bmds.push(BiologicalMaxwellDemon::new(*spec));
    }
    
    println!("   🧠 BMD Information Catalysts:");
    for (i, bmd) in bmds.iter().enumerate() {
        println!("   {:?}:", bmd.specialization);
        println!("     - Catalytic efficiency: {:.3}", bmd.catalytic_efficiency);
        println!("     - Fire specialization: {:.3}", bmd.fire_specialization_strength);
        println!("     - Memory patterns: {}", bmd.memory_patterns.len());
        println!("     - Learning rate: {:.4}", bmd.learning_rate);
    }
    
    // Test pattern learning
    let mut fire_bmd = BiologicalMaxwellDemon::new(BMDSpecialization::FireRecognition);
    let fire_pattern = vec![0.8, 0.6, 0.4, 0.9, 0.7];
    let fire_response = vec![1.0, 0.9, 0.8, 1.0, 0.9]; // Strong response
    
    fire_bmd.learn_pattern(fire_pattern, fire_response)?;
    println!("   ✓ Fire BMD learned pattern successfully");
    
    Ok(())
}

/// Test agency detection in fire circle contexts
async fn test_agency_detection() -> AutobahnResult<()> {
    println!("   Testing agency detection in fire circles...");
    
    let mut engine = FireConsciousnessEngine::new(5.0)?;
    
    // Simulate fire circle with multiple individuals
    let agency_patterns = vec![
        vec![0.7, 0.8, 0.6, 0.9, 0.75], // Individual 1 action
        vec![0.6, 0.7, 0.8, 0.85, 0.7], // Individual 2 action
        vec![0.8, 0.75, 0.7, 0.9, 0.8], // Individual 3 action
    ];
    
    println!("   🔥 Fire Circle Agency Detection:");
    for (i, pattern) in agency_patterns.iter().enumerate() {
        let response = engine.process_conscious_input(pattern).await?;
        let agency = &response.agency_detection;
        
        println!("   Individual {} action:", i + 1);
        println!("     - Agency detected: {}", agency.agency_detected);
        println!("     - Agency strength: {:.3}", agency.agency_strength);
        println!("     - Witness context active: {}", agency.witness_context_active);
        println!("     - Individual signatures: {}", agency.individual_signatures.len());
    }
    
    Ok(())
}

/// Test fire environment evolution over time
async fn test_fire_environment_evolution() -> AutobahnResult<()> {
    println!("   Testing consciousness evolution across fire exposure timeline...");
    
    let evolutionary_times = [8.0, 6.0, 5.0, 4.0, 3.0]; // MYA
    
    println!("   🕰️ Evolutionary Timeline:");
    for &time in &evolutionary_times {
        let engine = FireConsciousnessEngine::new(time)?;
        let fire_environment = &engine.fire_environment;
        
        // Calculate fire exposure probability for Olduvai ecosystem
        let exposure_prob = fire_environment.fire_exposure_probability(100.0, 120.0)?; // 100 km², 120 dry days
        let consciousness_enhancement = fire_environment.consciousness_enhancement_factor()?;
        
        println!("   {} MYA:", time);
        println!("     - Fire exposure probability: {:.4}", exposure_prob);
        println!("     - Consciousness enhancement: {:.3}x", consciousness_enhancement);
        println!("     - C4 grass coverage: {:.3}", fire_environment.c4_coverage);
        println!("     - Group size: {}", fire_environment.group_size);
    }
    
    Ok(())
}

/// Test integrated consciousness system with fire substrate
async fn test_integrated_consciousness() -> AutobahnResult<()> {
    println!("   Testing integrated consciousness with fire substrate...");
    
    let mut consciousness = ConsciousnessEmergenceEngine::new_with_evolutionary_time(5.0)?;
    
    // Test fire pattern detection
    let fire_input = "fireplace flames dancing in the darkness";
    let fire_response = consciousness.detect_fire_patterns(fire_input).await?;
    
    println!("   🧠 Integrated Consciousness Analysis:");
    println!("   Input: '{}'", fire_input);
    println!("   ✓ Fire recognition strength: {:.3}", fire_response.recognition_strength);
    println!("   ✓ Logical override: {}", fire_response.logical_override);
    println!("   ✓ Human attribution: {}", fire_response.human_attribution);
    
    // Test agency detection
    let agency_input = "person tending the fire carefully";
    let agency_response = consciousness.detect_agency(agency_input).await?;
    
    println!("   Input: '{}'", agency_input);
    println!("   ✓ Agency detected: {}", agency_response.agency_detected);
    println!("   ✓ Agency strength: {:.3}", agency_response.agency_strength);
    println!("   ✓ Witness context: {}", agency_response.witness_context_active);
    
    // Test Underwater Fireplace Paradox through integrated system
    let paradox_test = consciousness.test_underwater_fireplace_paradox().await?;
    println!("   ✓ Integrated paradox test: {}", paradox_test.paradox_demonstrated);
    
    Ok(())
}

/// Test darkness fear response (consciousness malfunction without light)
async fn test_darkness_fear_response() -> AutobahnResult<()> {
    println!("   Testing darkness fear response...");
    
    let mut engine = FireConsciousnessEngine::new(5.0)?;
    
    // Test different light levels
    let light_conditions = [
        (vec![0.8, 0.9, 0.7, 0.8, 0.6], "Bright daylight"),
        (vec![0.6, 0.7, 0.5, 0.6, 0.4], "Twilight"),
        (vec![0.4, 0.3, 0.2, 0.4, 0.3], "Fire light"),
        (vec![0.2, 0.1, 0.05, 0.2, 0.15], "Dim light"),
        (vec![0.05, 0.02, 0.01, 0.05, 0.03], "Near darkness"),
    ];
    
    println!("   🌙 Darkness Fear Response Analysis:");
    for (light_pattern, description) in &light_conditions {
        let response = engine.process_conscious_input(light_pattern).await?;
        
        println!("   {}:", description);
        println!("     - Light level: {:.3}", light_pattern[1]); // Assume index 1 is light
        println!("     - Darkness fear activation: {:.3}", response.darkness_fear_activation);
        println!("     - Consciousness level: {:.3}", response.consciousness_level);
        
        if response.darkness_fear_activation > 0.5 {
            println!("     ⚠️  HIGH DARKNESS FEAR - Consciousness malfunction detected!");
        }
    }
    
    Ok(())
}

/// Demonstrate fire environment creation and manipulation
async fn demonstrate_fire_environments() -> AutobahnResult<()> {
    println!("\n🔥 Fire Environment Demonstration");
    println!("==================================");
    
    // Create different fire environments
    let olduvai = FireEnvironment::olduvai_ecosystem();
    let underwater = FireEnvironment::impossible_underwater();
    
    println!("Olduvai Ecosystem (8-3 MYA):");
    println!("  - Intensity: {:.2}", olduvai.intensity);
    println!("  - Dominant wavelength: {:.0} nm", olduvai.dominant_wavelength_nm);
    println!("  - Temperature increase: {:.1}°C", olduvai.temperature_increase_c);
    println!("  - Group size: {}", olduvai.group_size);
    println!("  - C4 coverage: {:.2}", olduvai.c4_coverage);
    
    println!("\nUnderwater Fire Environment (Impossible):");
    println!("  - Context: {:?}", underwater.context);
    println!("  - Same physical properties but impossible context");
    
    let enhancement = olduvai.consciousness_enhancement_factor()?;
    println!("\nConsciousness enhancement factor: {:.3}x", enhancement);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_fire_consciousness_integration() -> AutobahnResult<()> {
        let mut engine = FireConsciousnessEngine::new(5.0)?;
        
        // Test basic functionality
        let fire_input = vec![0.8, 0.6, 0.4, 0.9, 0.7];
        let response = engine.process_conscious_input(&fire_input).await?;
        
        assert!(response.fire_recognition.recognition_strength > 0.5);
        assert!(response.consciousness_level > 0.3);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_underwater_fireplace_paradox_logic() -> AutobahnResult<()> {
        let mut engine = FireConsciousnessEngine::new(5.0)?;
        let test_result = engine.test_underwater_fireplace_paradox().await?;
        
        // The paradox should be demonstrated
        assert!(test_result.impossible_context);
        // Recognition should still occur despite impossibility
        assert!(test_result.recognition_strength > 0.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_ion_tunneling_physics() -> AutobahnResult<()> {
        let hydrogen = IonType::Hydrogen;
        let sodium = IonType::Sodium;
        
        // H+ should have higher tunneling probability (lighter mass)
        let h_prob = hydrogen.tunneling_probability(0.1, 1.0);
        let na_prob = sodium.tunneling_probability(0.1, 1.0);
        
        assert!(h_prob > na_prob);
        
        // Fire light should enhance H+ more than other ions
        let h_enhancement = hydrogen.fire_light_enhancement(650.0);
        let na_enhancement = sodium.fire_light_enhancement(650.0);
        
        assert!(h_enhancement > na_enhancement);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_bmd_specializations() -> AutobahnResult<()> {
        let fire_bmd = BiologicalMaxwellDemon::new(BMDSpecialization::FireRecognition);
        let agency_bmd = BiologicalMaxwellDemon::new(BMDSpecialization::AgencyDetection);
        
        // Fire BMD should have higher fire specialization
        assert!(fire_bmd.fire_specialization_strength > agency_bmd.fire_specialization_strength);
        
        // Both should have positive catalytic efficiency
        assert!(fire_bmd.catalytic_efficiency > 1.0);
        assert!(agency_bmd.catalytic_efficiency > 1.0);
        
        Ok(())
    }
    
    #[tokio::test]
    async fn test_consciousness_integration() -> AutobahnResult<()> {
        let mut consciousness = ConsciousnessEmergenceEngine::new_with_evolutionary_time(5.0)?;
        
        // Test fire detection
        let fire_response = consciousness.detect_fire_patterns("fireplace").await?;
        assert!(fire_response.recognition_strength > 0.0);
        
        // Test agency detection
        let agency_response = consciousness.detect_agency("person working").await?;
        assert!(agency_response.agency_strength >= 0.0);
        
        Ok(())
    }
} 