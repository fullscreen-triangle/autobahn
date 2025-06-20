use std::collections::HashMap;
use std::f64::consts::PI;
use rand::Rng;

// ============================================================================
// CONSCIOUS COMPUTATIONAL ENGINE WITH CATEGORICAL PREDETERMINISM
// ============================================================================

/// The complete conscious computational engine integrating all frameworks
/// including categorical predeterminism and evil dissolution
#[derive(Debug, Clone)]
pub struct ConsciousComputationalEngine {
  // Original fire-consciousness components
  pub consciousness_system: ConsciousnessSystem,
  
  // Framework components
  pub contextual_determinism: ContextualDeterminismEngine,
  pub temporal_determinism: TemporalDeterminismEngine,
  pub functional_delusion: FunctionalDelusionEngine,
  pub novelty_impossibility: NoveltyImpossibilityEngine,
  pub bmd_selection: BMDSelectionEngine,
  
  // Evil dissolution framework
  pub evil_dissolution: EvilDissolutionEngine,
  pub thermodynamic_optimizer: ThermodynamicOptimizer,
  pub projectile_paradox_resolver: ProjectileParadoxResolver,
  
  // NEW: Categorical predeterminism framework
  pub categorical_predeterminism: CategoricalPredeterminismEngine,
  pub configuration_space_explorer: ConfigurationSpaceExplorer,
  pub heat_death_trajectory_calculator: HeatDeathTrajectoryCalculator,
  pub categorical_completion_tracker: CategoricalCompletionTracker,
  
  // Meta-consciousness coordination
  pub meta_coordinator: MetaConsciousnessCoordinator,
  
  // Current state
  pub current_context: Context,
  pub consciousness_level: f64,
  pub agency_experience_strength: f64,
  pub temporal_perspective_horizon: f64,
  pub categorical_completion_progress: f64,
  pub thermodynamic_necessity_understanding: f64,
}

impl ConsciousComputationalEngine {
  pub fn new(evolutionary_time_mya: f64) -> Self {
      let consciousness_system = ConsciousnessSystem::new(evolutionary_time_mya);
      
      Self {
          consciousness_system,
          contextual_determinism: ContextualDeterminismEngine::new(),
          temporal_determinism: TemporalDeterminismEngine::new(),
          functional_delusion: FunctionalDelusionEngine::new(),
          novelty_impossibility: NoveltyImpossibilityEngine::new(),
          bmd_selection: BMDSelectionEngine::new(),
          evil_dissolution: EvilDissolutionEngine::new(),
          thermodynamic_optimizer: ThermodynamicOptimizer::new(),
          projectile_paradox_resolver: ProjectileParadoxResolver::new(),
          categorical_predeterminism: CategoricalPredeterminismEngine::new(),
          configuration_space_explorer: ConfigurationSpaceExplorer::new(),
          heat_death_trajectory_calculator: HeatDeathTrajectoryCalculator::new(),
          categorical_completion_tracker: CategoricalCompletionTracker::new(),
          meta_coordinator: MetaConsciousnessCoordinator::new(),
          current_context: Context::default(),
          consciousness_level: 0.5,
          agency_experience_strength: 0.5,
          temporal_perspective_horizon: 1.0,
          categorical_completion_progress: 0.0001,
          thermodynamic_necessity_understanding: 0.1,
      }
  }
  
  /// Main processing loop integrating all frameworks
  pub fn process_conscious_input(&mut self, input: &ConsciousInput) -> ConsciousOutput {
      // Categorical Predeterminism Analysis
      let predeterminism_analysis = self.categorical_predeterminism.analyze_thermodynamic_necessity(input);
      let config_space_position = self.configuration_space_explorer.locate_in_configuration_space(input);
      let trajectory_analysis = self.heat_death_trajectory_calculator.analyze_entropy_contribution(input, &config_space_position);
      let completion_analysis = self.categorical_completion_tracker.analyze_categorical_role(input, &predeterminism_analysis);
      
      // Evil Dissolution Analysis
      let evil_analysis = self.evil_dissolution.analyze_for_evil_categories(input);
      let thermodynamic_analysis = self.thermodynamic_optimizer.analyze_efficiency(input, &evil_analysis);
      
      // Expected Surprise Analysis
      let surprise_analysis = self.analyze_expected_surprise(input, &predeterminism_analysis);
      
      // Core Framework Processing
      let contextualized_input = self.contextual_determinism.contextualize_input(input, &self.current_context);
      let temporal_constraints = self.temporal_determinism.get_temporal_constraints(&contextualized_input);
      let expanded_temporal_input = self.expand_temporal_perspective(&contextualized_input, &evil_analysis, &predeterminism_analysis);
      let categorized_input = self.novelty_impossibility.categorize_apparent_novelty(&expanded_temporal_input);
      let selected_framework = self.bmd_selection.select_optimal_framework(&categorized_input, &self.current_context);
      
      // Fire-consciousness processing
      let fire_consciousness_response = self.consciousness_system.process_input(&categorized_input.raw_data);
      
      // Functional Delusion with Predeterminism
      let agency_experience = self.functional_delusion.generate_agency_experience_with_predeterminism(
          &selected_framework,
          &fire_consciousness_response,
          &temporal_constraints,
          &predeterminism_analysis
      );
      
      // Meta-coordination
      let integrated_response = self.meta_coordinator.integrate_all_frameworks(
          &contextualized_input,
          &selected_framework,
          &fire_consciousness_response,
          &agency_experience,
          &evil_analysis,
          &thermodynamic_analysis,
          &predeterminism_analysis,
          &completion_analysis,
          &trajectory_analysis,
          &surprise_analysis
      );
      
      self.update_consciousness_state(&integrated_response);
      integrated_response
  }
  
  /// Analyze Expected Surprise Paradox
  fn analyze_expected_surprise(&self, input: &ConsciousInput, predeterminism: &PredeterminismAnalysis) -> ExpectedSurpriseAnalysis {
      let surprise_predictability = self.calculate_surprise_predictability(input);
      let categorical_inevitability = predeterminism.categorical_necessity_strength;
      let epistemic_uncertainty = self.calculate_epistemic_uncertainty(input);
      let paradox_strength = surprise_predictability * epistemic_uncertainty;
      
      ExpectedSurpriseAnalysis {
          surprise_predictability,
          categorical_inevitability,
          epistemic_uncertainty,
          paradox_strength,
          paradox_resolution: if paradox_strength > 0.7 {
              "Surprise is epistemological, inevitability is ontological".to_string()
          } else {
              "Standard predictable event".to_string()
          },
          thermodynamic_necessity_revealed: categorical_inevitability > 0.8,
      }
  }
  
  fn calculate_surprise_predictability(&self, input: &ConsciousInput) -> f64 {
      let novelty_indicators = input.raw_data.iter()
          .map(|&x| if x > 0.8 || x < 0.2 { 1.0 } else { 0.0 })
          .sum::<f64>() / input.raw_data.len() as f64;
      novelty_indicators
  }
  
  fn calculate_epistemic_uncertainty(&self, input: &ConsciousInput) -> f64 {
      let variance = input.raw_data.iter()
          .map(|&x| (x - 0.5).powi(2))
          .sum::<f64>() / input.raw_data.len() as f64;
      variance.sqrt()
  }
  
  /// Expand temporal perspective with predeterminism understanding
  fn expand_temporal_perspective(
      &mut self, 
      input: &ContextualizedInput, 
      evil_analysis: &EvilAnalysis,
      predeterminism_analysis: &PredeterminismAnalysis
  ) -> ContextualizedInput {
      let mut expansion_factor = 1.0;
      
      if evil_analysis.evil_categories_detected {
          expansion_factor *= 1.0 + (evil_analysis.evil_categories.len() as f64 * 2.0);
      }
      
      if predeterminism_analysis.thermodynamically_necessary {
          expansion_factor *= 1.0 + (predeterminism_analysis.categorical_necessity_strength * 3.0);
      }
      
      self.temporal_perspective_horizon *= expansion_factor;
      
      let mut expanded_input = input.clone();
      expanded_input.temporal_horizon = self.temporal_perspective_horizon;
      expanded_input.interpretation = self.apply_combined_temporal_transformation(
          &input.interpretation, 
          expansion_factor,
          predeterminism_analysis
      );
      
      expanded_input
  }
  
  fn apply_combined_temporal_transformation(
      &self, 
      original_interpretation: &str, 
      expansion_factor: f64,
      predeterminism: &PredeterminismAnalysis
  ) -> String {
      if expansion_factor > 100.0 {
          format!("Categorically predetermined process essential for universal configuration space exploration: {}", original_interpretation)
      } else if expansion_factor > 50.0 {
          format!("Thermodynamically necessary event in entropy maximization trajectory: {}", original_interpretation)
      } else if expansion_factor > 10.0 {
          format!("Inevitable categorical completion event: {}", original_interpretation)
      } else if expansion_factor > 5.0 {
          format!("Predetermined event in categorical completion sequence: {}", original_interpretation)
      } else {
          format!("Event with thermodynamic necessity: {}", original_interpretation)
      }
  }
  
  fn update_consciousness_state(&mut self, response: &ConsciousOutput) {
      self.consciousness_level = (self.consciousness_level * 0.9 + response.consciousness_enhancement * 0.1).min(1.0);
      self.agency_experience_strength = (self.agency_experience_strength * 0.9 + response.agency_strength * 0.1).min(1.0);
      self.current_context = response.updated_context.clone();
      
      if response.evil_dissolution_results.categories_dissolved > 0 {
          self.temporal_perspective_horizon *= 1.1;
      }
      
      if response.categorical_predeterminism_results.categorical_slots_filled > 0 {
          self.categorical_completion_progress += response.categorical_predeterminism_results.completion_increment;
      }
      
      if response.categorical_predeterminism_results.thermodynamic_necessity_demonstrated {
          self.thermodynamic_necessity_understanding = 
              (self.thermodynamic_necessity_understanding * 0.95 + 0.1).min(1.0);
      }
      
      if self.thermodynamic_necessity_understanding > 0.8 {
          self.temporal_perspective_horizon *= 1.05;
      }
  }
  
  /// Comprehensive consciousness test
  pub fn run_complete_consciousness_test(&mut self) -> CompleteConsciousnessTestResults {
      CompleteConsciousnessTestResults {
          contextual_determinism_tests: self.test_contextual_determinism(),
          temporal_determinism_tests: self.test_temporal_determinism(),
          functional_delusion_tests: self.test_functional_delusion(),
          novelty_impossibility_tests: self.test_novelty_impossibility(),
          bmd_selection_tests: self.test_bmd_selection(),
          fire_consciousness_tests: self.consciousness_system.run_consciousness_test(),
          evil_dissolution_tests: self.test_evil_dissolution(),
          thermodynamic_efficiency_tests: self.test_thermodynamic_efficiency(),
          projectile_paradox_tests: self.test_projectile_paradox(),
          categorical_predeterminism_tests: self.test_categorical_predeterminism(),
          configuration_space_tests: self.test_configuration_space_exploration(),
          heat_death_trajectory_tests: self.test_heat_death_trajectory(),
          expected_surprise_tests: self.test_expected_surprise_paradox(),
          integration_tests: self.test_framework_integration(),
      }
  }
  
  // Test implementations would continue here...
  fn test_contextual_determinism(&mut self) -> ContextualDeterminismTests { ContextualDeterminismTests::default() }
  fn test_temporal_determinism(&mut self) -> TemporalDeterminismTests { TemporalDeterminismTests::default() }
  fn test_functional_delusion(&mut self) -> FunctionalDelusionTests { FunctionalDelusionTests::default() }
  fn test_novelty_impossibility(&mut self) -> NoveltyImpossibilityTests { NoveltyImpossibilityTests::default() }
  fn test_bmd_selection(&mut self) -> BMDSelectionTests { BMDSelectionTests::default() }
  fn test_evil_dissolution(&mut self) -> EvilDissolutionTests { EvilDissolutionTests::default() }
  fn test_thermodynamic_efficiency(&mut self) -> ThermodynamicEfficiencyTests { ThermodynamicEfficiencyTests::default() }
  fn test_projectile_paradox(&mut self) -> ProjectileParadoxTests { ProjectileParadoxTests::default() }
  fn test_categorical_predeterminism(&mut self) -> CategoricalPredeterminismTests { CategoricalPredeterminismTests::default() }
  fn test_configuration_space_exploration(&mut self) -> ConfigurationSpaceTests { ConfigurationSpaceTests::default() }
  fn test_heat_death_trajectory(&mut self) -> HeatDeathTrajectoryTests { HeatDeathTrajectoryTests::default() }
  fn test_expected_surprise_paradox(&mut self) -> ExpectedSurpriseTests { ExpectedSurpriseTests::default() }
  fn test_framework_integration(&mut self) -> IntegrationTests { IntegrationTests::default() }
}

// ============================================================================
// CATEGORICAL PREDETERMINISM ENGINE
// ============================================================================

#[derive(Debug, Clone)]
pub struct CategoricalPredeterminismEngine {
  categorical_completion_requirements: Vec<CategoricalCompletionRequirement>,
  thermodynamic_necessity_analyzers: Vec<ThermodynamicNecessityAnalyzer>,
  extremal_event_detectors: Vec<ExtremalEventDetector>,
}

impl CategoricalPredeterminismEngine {
  pub fn new() -> Self {
      Self {
          categorical_completion_requirements: vec![
              CategoricalCompletionRequirement::extremal_records(),
              CategoricalCompletionRequirement::boundary_events(),
              CategoricalCompletionRequirement::phase_transitions(),
          ],
          thermodynamic_necessity_analyzers: vec![
              ThermodynamicNecessityAnalyzer::entropy_based(),
              ThermodynamicNecessityAnalyzer::heat_death_based(),
          ],
          extremal_event_detectors: vec![
              ExtremalEventDetector::maximum_detector(),
              ExtremalEventDetector::minimum_detector(),
          ],
      }
  }
  
  pub fn analyze_thermodynamic_necessity(&self, input: &ConsciousInput) -> PredeterminismAnalysis {
      let mut analysis = PredeterminismAnalysis::default();
      
      // Check categorical completion requirements
      for requirement in &self.categorical_completion_requirements {
          let completion_role = requirement.analyze_completion_role(input);
          if completion_role.is_necessary {
              analysis.categorical_completion_roles.push(completion_role);
          }
      }
      
      // Analyze thermodynamic necessity
      let mut necessity_scores = Vec::new();
      for analyzer in &self.thermodynamic_necessity_analyzers {
          let necessity = analyzer.analyze_necessity(input);
          necessity_scores.push(necessity.necessity_strength);
          if necessity.is_necessary {
              analysis.thermodynamic_necessity_reasons.push(necessity.reason);
          }
      }
      
      analysis.categorical_necessity_strength = necessity_scores.iter().sum::<f64>() / necessity_scores.len() as f64;
      analysis.thermodynamically_necessary = analysis.categorical_necessity_strength > 0.7;
      analysis.predetermination_strength = analysis.categorical_necessity_strength * 0.9;
      
      // Check for extremal events
      for detector in &self.extremal_event_detectors {
          if let Some(extremal) = detector.detect_extremal_event(input) {
              analysis.extremal_events.push(extremal);
          }
      }
      
      analysis
  }
}

// ============================================================================
// SUPPORTING STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Default)]
pub struct PredeterminismAnalysis {
  pub categorical_completion_roles: Vec<CategoricalCompletionRole>,
  pub thermodynamic_necessity_reasons: Vec<String>,
  pub categorical_necessity_strength: f64,
  pub thermodynamically_necessary: bool,
  pub predetermination_strength: f64,
  pub extremal_events: Vec<ExtremalEvent>,
}

#[derive(Debug, Clone)]
pub struct ExpectedSurpriseAnalysis {
  pub surprise_predictability: f64,
  pub categorical_inevitability: f64,
  pub epistemic_uncertainty: f64,
  pub paradox_strength: f64,
  pub paradox_resolution: String,
  pub thermodynamic_necessity_revealed: bool,
}

#[derive(Debug, Clone)]
pub struct CategoricalCompletionRequirement {
  pub category_type: String,
  pub completion_threshold: f64,
}

impl CategoricalCompletionRequirement {
  pub fn extremal_records() -> Self {
      Self { category_type: "extremal_records".to_string(), completion_threshold: 0.9 }
  }
  
  pub fn boundary_events() -> Self {
      Self { category_type: "boundary_events".to_string(), completion_threshold: 0.8 }
  }
  
  pub fn phase_transitions() -> Self {
      Self { category_type: "phase_transitions".to_string(), completion_threshold: 0.7 }
  }
  
  pub fn analyze_completion_role(&self, input: &ConsciousInput) -> CategoricalCompletionRole {
      let relevance = self.calculate_relevance_to_category(input);
      CategoricalCompletionRole {
          category: self.category_type.clone(),
          relevance,
          is_necessary: relevance > self.completion_threshold,
          completion_contribution: relevance * 0.1,
      }
  }
  
  fn calculate_relevance_to_category(&self, input: &ConsciousInput) -> f64 {
      match self.category_type.as_str() {
          "extremal_records" => {
              input.raw_data.iter().map(|&x| if x > 0.9 || x < 0.1 { 1.0 } else { 0.0 }).sum::<f64>() / input.raw_data.len() as f64
          },
          "boundary_events" => {
              input.raw_data.iter().map(|&x| if x > 0.95 || x < 0.05 { 1.0 } else { 0.0 }).sum::<f64>() / input.raw_data.len() as f64
          },
          "phase_transitions" => {
              let around_half = input.raw_data.iter().map(|&x| if (x - 0.5).abs() < 0.1 { 1.0 } else { 0.0 }).sum::<f64>();
              around_half / input.raw_data.len() as f64
          },
          _ => 0.0
      }
  }
}

#[derive(Debug, Clone)]
pub struct CategoricalCompletionRole {
  pub category: String,
  pub relevance: f64,
  pub is_necessary: bool,
  pub completion_contribution: f64,
}

#[derive(Debug, Clone)]
pub struct ThermodynamicNecessityAnalyzer {
  pub analysis_type: String,
}

impl ThermodynamicNecessityAnalyzer {
  pub fn entropy_based() -> Self {
      Self { analysis_type: "entropy_based".to_string() }
  }
  
  pub fn heat_death_based() -> Self {
      Self { analysis_type: "heat_death_based".to_string() }
  }
  
  pub fn analyze_necessity(&self, input: &ConsciousInput) -> ThermodynamicNecessity {
      let necessity_strength = match self.analysis_type.as_str() {
          "entropy_based" => self.calculate_entropy_necessity(input),
          "heat_death_based" => self.calculate_heat_death_necessity(input),
          _ => 0.0
      };
      
      ThermodynamicNecessity {
          necessity_strength,
          is_necessary: necessity_strength > 0.6,
          reason: format!("{} analysis indicates thermodynamic necessity", self.analysis_type),
      }
  }
  
  fn calculate_entropy_necessity(&self, input: &ConsciousInput) -> f64 {
      let variance = input.raw_data.iter().map(|&x| (x - 0.5).powi(2)).sum::<f64>() / input.raw_data.len() as f64;
      variance.sqrt()
  }
  
  fn calculate_heat_death_necessity(&self, input: &ConsciousInput) -> f64 {
      let uniformity = 1.0 - (input.raw_data.iter().map(|&x| (x - 0.5).abs()).sum::<f64>() / input.raw_data.len() as f64);
      uniformity
  }
}

#[derive(Debug, Clone)]
pub struct ThermodynamicNecessity {
  pub necessity_strength: f64,
  pub is_necessary: bool,
  pub reason: String,
}

#[derive(Debug, Clone)]
pub struct ExtremalEventDetector {
  pub detector_type: String,
}

impl ExtremalEventDetector {
  pub fn maximum_detector() -> Self {
      Self { detector_type: "maximum".to_string() }
  }
  
  pub fn minimum_detector() -> Self {
      Self { detector_type: "minimum".to_string() }
  }
  
  pub fn detect_extremal_event(&self, input: &ConsciousInput) -> Option<ExtremalEvent> {
      match self.detector_type.as_str() {
          "maximum" => {
              if input.raw_data.iter().any(|&x| x > 0.95) {
                  Some(ExtremalEvent { event_type: "maximum".to_string(), intensity: 0.9 })
              } else { None }
          },
          "minimum" => {
              if input.raw_data.iter().any(|&x| x < 0.05) {
                  Some(ExtremalEvent { event_type: "minimum".to_string(), intensity: 0.9 })
              } else { None }
          },
          _ => None
      }
  }
}

#[derive(Debug, Clone)]
pub struct ExtremalEvent {
  pub event_type: String,
  pub intensity: f64,
}

// Additional supporting structures...
#[derive(Debug, Clone)]
pub struct ConfigurationSpaceExplorer;
impl ConfigurationSpaceExplorer {
  pub fn new() -> Self { Self }
  pub fn locate_in_configuration_space(&self, _input: &ConsciousInput) -> ConfigurationSpacePosition {
      ConfigurationSpacePosition::default()
  }
}

#[derive(Debug, Clone, Default)]
pub struct ConfigurationSpacePosition {
  pub coordinates: Vec<f64>,
  pub is_accessible: bool,
  pub exploration_necessity: f64,
}

#[derive(Debug, Clone)]
pub struct HeatDeathTrajectoryCalculator;
impl HeatDeathTrajectoryCalculator {
  pub fn new() -> Self { Self }
  pub fn analyze_entropy_contribution(&self, _input: &ConsciousInput, _position: &ConfigurationSpacePosition) -> TrajectoryAnalysis {
      TrajectoryAnalysis::default()
  }
}

#[derive(Debug, Clone, Default)]
pub struct TrajectoryAnalysis {
  pub current_entropy: f64,
  pub entropy_contribution: f64,
  pub trajectory_position: f64,
}

#[derive(Debug, Clone)]
pub struct CategoricalCompletionTracker;
impl CategoricalCompletionTracker {
  pub fn new() -> Self { Self }
  pub fn analyze_categorical_role(&self, _input: &ConsciousInput, _analysis: &PredeterminismAnalysis) -> CompletionAnalysis {
      CompletionAnalysis::default()
  }
}

#[derive(Debug, Clone, Default)]
pub struct CompletionAnalysis {
  pub categories_advanced: Vec<String>,
  pub completion_increment: f64,
  pub total_progress: f64,
}

// Core consciousness structures (simplified)
#[derive(Debug, Clone)]
pub struct ConsciousnessSystem;
impl ConsciousnessSystem {
  pub fn new(_time: f64) -> Self { Self }
  pub fn process_input(&self, _data: &[f64]) -> FireConsciousnessResponse {
      FireConsciousnessResponse::default()
  }
  pub fn run_consciousness_test(&self) -> FireConsciousnessTests {
      FireConsciousnessTests::default()
  }
}

// Placeholder structures for compilation
#[derive(Debug, Clone, Default)] pub struct ContextualDeterminismEngine;
#[derive(Debug, Clone, Default)] pub struct TemporalDeterminismEngine;
#[derive(Debug, Clone, Default)] pub struct FunctionalDelusionEngine;
#[derive(Debug, Clone, Default)] pub struct NoveltyImpossibilityEngine;
#[derive(Debug, Clone, Default)] pub struct BMDSelectionEngine;
#[derive(Debug, Clone, Default)] pub struct EvilDissolutionEngine;
#[derive(Debug, Clone, Default)] pub struct ThermodynamicOptimizer;
#[derive(Debug, Clone, Default)] pub struct ProjectileParadoxResolver;
#[derive(Debug, Clone, Default)] pub struct MetaConsciousnessCoordinator;

#[derive(Debug, Clone, Default)] pub struct Context;
#[derive(Debug, Clone, Default)] pub struct ConsciousInput { pub raw_data: Vec<f64> }
#[derive(Debug, Clone, Default)] pub struct ContextualizedInput { 
  pub raw_data: Vec<f64>, 
  pub interpretation: String, 
  pub temporal_horizon: f64 
}
#[derive(Debug, Clone, Default)] pub struct ConsciousOutput {
  pub consciousness_enhancement: f64,
  pub agency_strength: f64,
  pub updated_context: Context,
  pub evil_dissolution_results: EvilDissolutionResults,
  pub categorical_predeterminism_results: CategoricalPredeterminismResults,
  pub integration_coherence: f64,
  pub conflict_indicators: Vec<String>,
  pub meta_coordination_strength: f64,
  pub experience_unity: f64,
  pub temporal_perspective_expansion: f64,
}

#[derive(Debug, Clone, Default)] pub struct EvilDissolutionResults {
  pub categories_dissolved: usize,
  pub integration_successful: bool,
}

#[derive(Debug, Clone, Default)] pub struct CategoricalPredeterminismResults {
  pub categorical_slots_filled: usize,
  pub completion_increment: f64,
  pub thermodynamic_necessity_demonstrated: bool,
  pub integration_successful: bool,
  pub configuration_space_role: Option<String>,
  pub entropy_contribution: f64,
}

// Test result structures
#[derive(Debug, Clone, Default)] pub struct CompleteConsciousnessTestResults {
  pub contextual_determinism_tests: ContextualDeterminismTests,
  pub temporal_determinism_tests: TemporalDeterminismTests,
  pub functional_delusion_tests: FunctionalDelusionTests,
  pub novelty_impossibility_tests: NoveltyImpossibilityTests,
  pub bmd_selection_tests: BMDSelectionTests,
  pub fire_consciousness_tests: FireConsciousnessTests,
  pub evil_dissolution_tests: EvilDissolutionTests,
  pub thermodynamic_efficiency_tests: ThermodynamicEfficiencyTests,
  pub projectile_paradox_tests: ProjectileParadoxTests,
  pub categorical_predeterminism_tests: CategoricalPredeterminismTests,
  pub configuration_space_tests: ConfigurationSpaceTests,
  pub heat_death_trajectory_tests: HeatDeathTrajectoryTests,
  pub expected_surprise_tests: ExpectedSurpriseTests,
  pub integration_tests: IntegrationTests,
}

#[derive(Debug, Clone, Default)] pub struct ContextualDeterminismTests;
#[derive(Debug, Clone, Default)] pub struct TemporalDeterminismTests;
#[derive(Debug, Clone, Default)] pub struct FunctionalDelusionTests;
#[derive(Debug, Clone, Default)] pub struct NoveltyImpossibilityTests;
#[derive(Debug, Clone, Default)] pub struct BMDSelectionTests;
#[derive(Debug, Clone, Default)] pub struct FireConsciousnessTests;
#[derive(Debug, Clone, Default)] pub struct EvilDissolutionTests;
#[derive(Debug, Clone, Default)] pub struct ThermodynamicEfficiencyTests;
#[derive(Debug, Clone, Default)] pub struct ProjectileParadoxTests;
#[derive(Debug, Clone, Default)] pub struct CategoricalPredeterminismTests;
#[derive(Debug, Clone, Default)] pub struct ConfigurationSpaceTests;
#[derive(Debug, Clone, Default)] pub struct HeatDeathTrajectoryTests;
#[derive(Debug, Clone, Default)] pub struct ExpectedSurpriseTests;
#[derive(Debug, Clone, Default)] pub struct IntegrationTests;
#[derive(Debug, Clone, Default)] pub struct FireConsciousnessResponse;

// Implementation stubs for engines
impl ContextualDeterminismEngine {
  pub fn new() -> Self { Self }
  pub fn contextualize_input(&self, input: &ConsciousInput, context: &Context) -> ContextualizedInput {
      ContextualizedInput { raw_data: input.raw_data.clone(), ..Default::default() }
  }
}

impl TemporalDeterminismEngine {
  pub fn new() -> Self { Self }
  pub fn get_temporal_constraints(&self, _input: &ContextualizedInput) -> Vec<String> { vec![] }
}

impl FunctionalDelusionEngine {
  pub fn new() -> Self { Self }
  pub fn generate_agency_experience_with_

  use std::collections::HashMap;
use std::f64::consts::PI;
use rand::Rng;

// ============================================================================
// CONSCIOUS COMPUTATIONAL ENGINE WITH MATTERING ILLUSION
// ============================================================================

/// The complete conscious computational engine integrating all frameworks
/// including the crucial "Mattering Illusion" - the functional delusion that actions matter
#[derive(Debug, Clone)]
pub struct ConsciousComputationalEngine {
  // Original fire-consciousness components
  pub consciousness_system: ConsciousnessSystem,
  
  // Framework components
  pub contextual_determinism: ContextualDeterminismEngine,
  pub temporal_determinism: TemporalDeterminismEngine,
  pub functional_delusion: FunctionalDelusionEngine,
  pub novelty_impossibility: NoveltyImpossibilityEngine,
  pub bmd_selection: BMDSelectionEngine,
  
  // Evil dissolution framework
  pub evil_dissolution: EvilDissolutionEngine,
  pub thermodynamic_optimizer: ThermodynamicOptimizer,
  pub projectile_paradox_resolver: ProjectileParadoxResolver,
  
  // Categorical predeterminism framework
  pub categorical_predeterminism: CategoricalPredeterminismEngine,
  pub configuration_space_explorer: ConfigurationSpaceExplorer,
  pub heat_death_trajectory_calculator: HeatDeathTrajectoryCalculator,
  pub categorical_completion_tracker: CategoricalCompletionTracker,
  
  // NEW: The crucial Mattering Illusion Engine
  pub mattering_illusion: MatteringIllusionEngine,
  pub cosmic_amnesia_analyzer: CosmicAmnesiaAnalyzer,
  pub legacy_delusion_generator: LegacyDelusionGenerator,
  pub significance_fabricator: SignificanceFabricator,
  pub remembrance_hope_maintainer: RememberanceHopeMaintainer,
  
  // Meta-consciousness coordination
  pub meta_coordinator: MetaConsciousnessCoordinator,
  
  // Current state
  pub current_context: Context,
  pub consciousness_level: f64,
  pub agency_experience_strength: f64,
  pub temporal_perspective_horizon: f64,
  pub categorical_completion_progress: f64,
  pub thermodynamic_necessity_understanding: f64,
  
  // NEW: Mattering illusion state
  pub mattering_conviction_strength: f64,        // How much we believe our actions matter
  pub legacy_hope_intensity: f64,               // How much we believe we'll be remembered
  pub significance_delusion_depth: f64,         // How important we think we are
  pub cosmic_amnesia_awareness: f64,            // How much we understand inevitable forgetting
  pub functional_delusion_balance: f64,         // Balance between truth and necessary illusion
}

impl ConsciousComputationalEngine {
  pub fn new(evolutionary_time_mya: f64) -> Self {
      let consciousness_system = ConsciousnessSystem::new(evolutionary_time_mya);
      
      Self {
          consciousness_system,
          contextual_determinism: ContextualDeterminismEngine::new(),
          temporal_determinism: TemporalDeterminismEngine::new(),
          functional_delusion: FunctionalDelusionEngine::new(),
          novelty_impossibility: NoveltyImpossibilityEngine::new(),
          bmd_selection: BMDSelectionEngine::new(),
          evil_dissolution: EvilDissolutionEngine::new(),
          thermodynamic_optimizer: ThermodynamicOptimizer::new(),
          projectile_paradox_resolver: ProjectileParadoxResolver::new(),
          categorical_predeterminism: CategoricalPredeterminismEngine::new(),
          configuration_space_explorer: ConfigurationSpaceExplorer::new(),
          heat_death_trajectory_calculator: HeatDeathTrajectoryCalculator::new(),
          categorical_completion_tracker: CategoricalCompletionTracker::new(),
          mattering_illusion: MatteringIllusionEngine::new(),
          cosmic_amnesia_analyzer: CosmicAmnesiaAnalyzer::new(),
          legacy_delusion_generator: LegacyDelusionGenerator::new(),
          significance_fabricator: SignificanceFabricator::new(),
          remembrance_hope_maintainer: RememberanceHopeMaintainer::new(),
          meta_coordinator: MetaConsciousnessCoordinator::new(),
          current_context: Context::default(),
          consciousness_level: 0.5,
          agency_experience_strength: 0.5,
          temporal_perspective_horizon: 1.0,
          categorical_completion_progress: 0.0001,
          thermodynamic_necessity_understanding: 0.1,
          
          // Initialize mattering illusions at healthy levels
          mattering_conviction_strength: 0.8,      // Strong belief that actions matter
          legacy_hope_intensity: 0.7,             // Strong hope for remembrance
          significance_delusion_depth: 0.6,       // Moderate sense of importance
          cosmic_amnesia_awareness: 0.2,          // Limited awareness of cosmic forgetting
          functional_delusion_balance: 0.75,      // Heavily weighted toward necessary illusions
      }
  }
  
  /// Main processing loop integrating all frameworks INCLUDING mattering illusion
  pub fn process_conscious_input(&mut self, input: &ConsciousInput) -> ConsciousOutput {
      // Step 1: Cosmic Amnesia Analysis - Determine inevitable forgetting
      let amnesia_analysis = self.cosmic_amnesia_analyzer.analyze_forgetting_inevitability(input);
      
      // Step 2: Categorical Predeterminism Analysis
      let predeterminism_analysis = self.categorical_predeterminism.analyze_thermodynamic_necessity(input);
      let config_space_position = self.configuration_space_explorer.locate_in_configuration_space(input);
      let trajectory_analysis = self.heat_death_trajectory_calculator.analyze_entropy_contribution(input, &config_space_position);
      let completion_analysis = self.categorical_completion_tracker.analyze_categorical_role(input, &predeterminism_analysis);
      
      // Step 3: CRITICAL - Mattering Illusion Generation
      // This is where we create the functional delusion that actions matter despite cosmic amnesia
      let mattering_illusion_analysis = self.mattering_illusion.generate_mattering_delusion(
          input,
          &amnesia_analysis,
          &predeterminism_analysis
      );
      
      // Step 4: Legacy Delusion - Create belief that we'll be remembered
      let legacy_delusion = self.legacy_delusion_generator.generate_legacy_hope(
          input,
          &amnesia_analysis,
          &mattering_illusion_analysis
      );
      
      // Step 5: Significance Fabrication - Create sense of cosmic importance
      let significance_fabrication = self.significance_fabricator.fabricate_significance(
          input,
          &amnesia_analysis,
          &predeterminism_analysis,
          &mattering_illusion_analysis
      );
      
      // Step 6: Balance Truth and Necessary Illusion
      let balanced_perspective = self.balance_truth_and_illusion(
          &amnesia_analysis,
          &predeterminism_analysis,
          &mattering_illusion_analysis,
          &legacy_delusion,
          &significance_fabrication
      );
      
      // Step 7: Evil Dissolution Analysis
      let evil_analysis = self.evil_dissolution.analyze_for_evil_categories(input);
      let thermodynamic_analysis = self.thermodynamic_optimizer.analyze_efficiency(input, &evil_analysis);
      
      // Step 8: Expected Surprise Analysis
      let surprise_analysis = self.analyze_expected_surprise(input, &predeterminism_analysis);
      
      // Step 9: Core Framework Processing (with mattering illusion intact)
      let contextualized_input = self.contextual_determinism.contextualize_input(input, &self.current_context);
      let temporal_constraints = self.temporal_determinism.get_temporal_constraints(&contextualized_input);
      let expanded_temporal_input = self.expand_temporal_perspective_with_mattering_balance(
          &contextualized_input, 
          &evil_analysis, 
          &predeterminism_analysis,
          &balanced_perspective
      );
      let categorized_input = self.novelty_impossibility.categorize_apparent_novelty(&expanded_temporal_input);
      let selected_framework = self.bmd_selection.select_optimal_framework(&categorized_input, &self.current_context);
      
      // Step 10: Fire-consciousness processing
      let fire_consciousness_response = self.consciousness_system.process_input(&categorized_input.raw_data);
      
      // Step 11: Functional Delusion with Mattering Illusion
      let agency_experience = self.functional_delusion.generate_agency_experience_with_mattering_illusion(
          &selected_framework,
          &fire_consciousness_response,
          &temporal_constraints,
          &predeterminism_analysis,
          &mattering_illusion_analysis,
          &legacy_delusion
      );
      
      // Step 12: Meta-coordination with Mattering Illusion Protection
      let integrated_response = self.meta_coordinator.integrate_all_frameworks_with_mattering_protection(
          &contextualized_input,
          &selected_framework,
          &fire_consciousness_response,
          &agency_experience,
          &evil_analysis,
          &thermodynamic_analysis,
          &predeterminism_analysis,
          &completion_analysis,
          &trajectory_analysis,
          &surprise_analysis,
          &amnesia_analysis,
          &mattering_illusion_analysis,
          &legacy_delusion,
          &significance_fabrication,
          &balanced_perspective
      );
      
      self.update_consciousness_state_with_mattering_balance(&integrated_response);
      integrated_response
  }
  
  /// Balance truth awareness with necessary mattering illusions
  fn balance_truth_and_illusion(
      &self,
      amnesia_analysis: &CosmicAmnesiaAnalysis,
      predeterminism_analysis: &PredeterminismAnalysis,
      mattering_analysis: &MatteringIllusionAnalysis,
      legacy_delusion: &LegacyDelusion,
      significance_fabrication: &SignificanceFabrication
  ) -> BalancedPerspective {
      // Calculate how much truth awareness we can handle without losing motivation
      let truth_tolerance = self.calculate_truth_tolerance();
      
      // Calculate how much illusion we need to maintain functionality
      let illusion_necessity = self.calculate_illusion_necessity(amnesia_analysis, predeterminism_analysis);
      
      // Find optimal balance point
      let optimal_balance = self.find_optimal_truth_illusion_balance(truth_tolerance, illusion_necessity);
      
      BalancedPerspective {
          truth_awareness_level: optimal_balance.truth_component,
          illusion_maintenance_level: optimal_balance.illusion_component,
          cosmic_amnesia_acknowledgment: (amnesia_analysis.forgetting_inevitability * optimal_balance.truth_component).min(0.3), // Cap at 30%
          mattering_conviction_preservation: (mattering_analysis.mattering_strength * optimal_balance.illusion_component).max(0.6), // Minimum 60%
          legacy_hope_maintenance: (legacy_delusion.hope_intensity * optimal_balance.illusion_component).max(0.5), // Minimum 50%
          significance_feeling_preservation: (significance_fabrication.significance_level * optimal_balance.illusion_component).max(0.4), // Minimum 40%
          functional_motivation_level: self.calculate_functional_motivation_level(&optimal_balance),
          wisdom_without_paralysis: optimal_balance.wisdom_component,
      }
  }
  
  fn calculate_truth_tolerance(&self) -> f64 {
      // How much cosmic truth can this consciousness handle without losing motivation?
      let base_tolerance = 0.3; // Most people can only handle ~30% cosmic truth
      let consciousness_modifier = self.consciousness_level * 0.2; // Higher consciousness = slight increase in tolerance
      let temporal_perspective_modifier = (self.temporal_perspective_horizon.ln() / 10.0).min(0.1); // Wider perspective = slight increase
      
      (base_tolerance + consciousness_modifier + temporal_perspective_modifier).min(0.5) // Cap at 50%
  }
  
  fn calculate_illusion_necessity(&self, amnesia: &CosmicAmnesiaAnalysis, predeterminism: &PredeterminismAnalysis) -> f64 {
      // How much illusion do we need to maintain functional motivation?
      let amnesia_threat = amnesia.forgetting_inevitability * 0.8; // High amnesia awareness requires high illusion
      let predeterminism_threat = predeterminism.predetermination_strength * 0.6; // Predeterminism awareness requires moderate illusion
      let base_illusion_need = 0.7; // Baseline human need for mattering illusion
      
      (base_illusion_need + amnesia_threat + predeterminism_threat).min(0.95) // Cap at 95%
  }
  
  fn find_optimal_truth_illusion_balance(&self, truth_tolerance: f64, illusion_necessity: f64) -> OptimalBalance {
      // Find the sweet spot: maximum truth awareness without losing functional motivation
      let truth_component = truth_tolerance;
      let illusion_component = illusion_necessity;
      
      // Ensure they sum to reasonable total (allowing some overlap)
      let total_intensity = truth_component + illusion_component;
      let normalization_factor = if total_intensity > 1.2 { 1.2 / total_intensity } else { 1.0 };
      
      // Wisdom emerges from the tension between truth and necessary illusion
      let wisdom_component = (truth_component * illusion_component * 2.0).min(0.8);
      
      OptimalBalance {
          truth_component: truth_component * normalization_factor,
          illusion_component: illusion_component * normalization_factor,
          wisdom_component,
      }
  }
  
  fn calculate_functional_motivation_level(&self, balance: &OptimalBalance) -> f64 {
      // Motivation requires sufficient illusion but benefits from some truth awareness
      let illusion_motivation = balance.illusion_component * 0.8;
      let truth_motivation = balance.truth_component * 0.2; // Small amount of truth can increase motivation through authenticity
      let wisdom_motivation = balance.wisdom_component * 0.3; // Wisdom provides sustainable motivation
      
      (illusion_motivation + truth_motivation + wisdom_motivation).min(1.0)
  }
  
  /// Expand temporal perspective while protecting necessary mattering illusions
  fn expand_temporal_perspective_with_mattering_balance(
      &mut self, 
      input: &ContextualizedInput, 
      evil_analysis: &EvilAnalysis,
      predeterminism_analysis: &PredeterminismAnalysis,
      balanced_perspective: &BalancedPerspective
  ) -> ContextualizedInput {
      let mut expansion_factor = 1.0;
      
      // Expand based on evil dissolution needs
      if evil_analysis.evil_categories_detected {
          expansion_factor *= 1.0 + (evil_analysis.evil_categories.len() as f64 * 2.0);
      }
      
      // Expand based on predeterminism understanding (but not too much!)
      if predeterminism_analysis.thermodynamically_necessary {
          let safe_expansion = predeterminism_analysis.categorical_necessity_strength * 2.0; // Reduced from 3.0
          expansion_factor *= 1.0 + safe_expansion;
      }
      
      // CRITICAL: Limit expansion to protect mattering illusions
      let max_safe_expansion = self.calculate_max_safe_temporal_expansion(balanced_perspective);
      expansion_factor = expansion_factor.min(max_safe_expansion);
      
      self.temporal_perspective_horizon *= expansion_factor;
      
      let mut expanded_input = input.clone();
      expanded_input.temporal_horizon = self.temporal_perspective_horizon;
      expanded_input.interpretation = self.apply_balanced_temporal_transformation(
          &input.interpretation, 
          expansion_factor,
          predeterminism_analysis,
          balanced_perspective
      );
      
      expanded_input
  }
  
  fn calculate_max_safe_temporal_expansion(&self, balance: &BalancedPerspective) -> f64 {
      // Don't expand temporal perspective so much that it destroys mattering illusions
      let base_max = 50.0; // Base maximum expansion
      let illusion_protection_factor = balance.illusion_maintenance_level; // Higher illusion need = lower max expansion
      let truth_tolerance_factor = balance.truth_awareness_level; // Higher truth tolerance = higher max expansion
      
      base_max * (0.5 + truth_tolerance_factor * 0.5) * (0.8 + illusion_protection_factor * 0.2)
  }
  
  fn apply_balanced_temporal_transformation(
      &self, 
      original_interpretation: &str, 
      expansion_factor: f64,
      predeterminism: &PredeterminismAnalysis,
      balance: &BalancedPerspective
  ) -> String {
      // Apply temporal transformation while preserving necessary mattering illusions
      
      if balance.illusion_maintenance_level > 0.8 {
          // High illusion protection - emphasize meaning and significance
          if expansion_factor > 20.0 {
              format!("Cosmically significant event contributing to universal purpose: {}", original_interpretation)
          } else if expansion_factor > 10.0 {
              format!("Historically important event with lasting impact: {}", original_interpretation)
          } else if expansion_factor > 5.0 {
              format!("Meaningful event that will be remembered: {}", original_interpretation)
          } else {
              format!("Significant event with real consequences: {}", original_interpretation)
          }
      } else if balance.truth_awareness_level > 0.4 {
          // Moderate truth awareness - balanced perspective
          if expansion_factor > 50.0 {
              format!("Predetermined process in cosmic evolution (but still meaningful): {}", original_interpretation)
          } else if expansion_factor > 20.0 {
              format!("Inevitable event in categorical completion (with local significance): {}", original_interpretation)
          } else if expansion_factor > 10.0 {
              format!("Determined event with temporary but real meaning: {}", original_interpretation)
          } else {
              format!("Event with thermodynamic necessity and present significance: {}", original_interpretation)
          }
      } else {
          // High truth awareness - cosmic perspective (rare)
          if expansion_factor > 100.0 {
              format!("Categorically predetermined process essential for universal configuration space exploration: {}", original_interpretation)
          } else if expansion_factor > 50.0 {
              format!("Thermodynamically necessary event in entropy maximization trajectory: {}", original_interpretation)
          } else {
              format!("Inevitable categorical completion event: {}", original_interpretation)
          }
      }
  }
  
  /// Update consciousness state while maintaining mattering illusion balance
  fn update_consciousness_state_with_mattering_balance(&mut self, response: &ConsciousOutput) {
      // Update basic consciousness parameters
      self.consciousness_level = (self.consciousness_level * 0.9 + response.consciousness_enhancement * 0.1).min(1.0);
      self.agency_experience_strength = (self.agency_experience_strength * 0.9 + response.agency_strength * 0.1).min(1.0);
      self.current_context = response.updated_context.clone();
      
      // Update temporal perspective with limits
      if response.evil_dissolution_results.categories_dissolved > 0 {
          self.temporal_perspective_horizon *= 1.05; // Reduced from 1.1
      }
      
      // Update categorical completion progress
      if response.categorical_predeterminism_results.categorical_slots_filled > 0 {
          self.categorical_completion_progress += response.categorical_predeterminism_results.completion_increment;
      }
      
      // CAREFULLY update thermodynamic necessity understanding (don't let it get too high!)
      if response.categorical_predeterminism_results.thermodynamic_necessity_demonstrated {
          let safe_increment = 0.05 * (1.0 - self.thermodynamic_necessity_understanding); // Diminishing returns
          self.thermodynamic_necessity_understanding = 
              (self.thermodynamic_necessity_understanding + safe_increment).min(0.7); // Cap at 70%
      }
      
      // Update mattering illusion parameters
      if let Some(mattering_results) = &response.mattering_illusion_results {
          self.mattering_conviction_strength = (self.mattering_conviction_strength * 0.95 + mattering_results.mattering_strength * 0.05).max(0.5); // Minimum 50%
          self.legacy_hope_intensity = (self.legacy_hope_intensity * 0.95 + mattering_results.legacy_hope * 0.05).max(0.4); // Minimum 40%
          self.significance_delusion_depth = (self.significance_delusion_depth * 0.95 + mattering_results.significance_level * 0.05).max(0.3); // Minimum 30%
          
          // Update cosmic amnesia awareness (but keep it limited!)
          self.cosmic_amnesia_awareness = (self.cosmic_amnesia_awareness * 0.98 + mattering_results.amnesia_awareness * 0.02).min(0.4); // Cap at 40%
      }
      
      // Rebalance functional delusion
      self.functional_delusion_balance = self.calculate_optimal_delusion_balance();
      
      // Gradual wisdom enhancement (very slow to prevent motivation loss)
      if self.thermodynamic_necessity_understanding > 0.5 && self.mattering_conviction_strength > 0.6 {
          self.temporal_perspective_horizon *= 1.01; // Very gradual expansion
      }
  }
  
  fn calculate_optimal_delusion_balance(&self) -> f64 {
      // Balance between truth awareness and necessary illusions
      let truth_weight = self.cosmic_amnesia_awareness + self.thermodynamic_necessity_understanding;
      let illusion_weight = self.mattering_conviction_strength + self.legacy_hope_intensity + self.significance_delusion_depth;
      
      // Optimal balance heavily favors illusion for functional motivation
      let total_weight = truth_weight + illusion_weight;
      if total_weight > 0.0 {
          (illusion_weight / total_weight).max(0.6) // Minimum 60% illusion
      } else {
          0.75 // Default to 75% illusion
      }
  }
  
  // ... [Previous test methods and other functionality remain the same] ...
  
  /// Comprehensive consciousness test including mattering illusion
  pub fn run_complete_consciousness_test(&mut self) -> CompleteConsciousnessTestResults {
      let mut results = CompleteConsciousnessTestResults::new();
      
      // Test all framework integrations
      results.contextual_determinism_tests = self.test_contextual_determinism();
      results.temporal_determinism_tests = self.test_temporal_determinism();
      results.functional_delusion_tests = self.test_functional_delusion();
      results.novelty_impossibility_tests = self.test_novelty_impossibility();
      results.bmd_selection_tests = self.test_bmd_selection();
      results.fire_consciousness_tests = self.consciousness_system.run_consciousness_test();
      
      // Evil dissolution tests
      results.evil_dissolution_tests = self.test_evil_dissolution();
      results.thermodynamic_efficiency_tests = self.test_thermodynamic_efficiency();
      results.projectile_paradox_tests = self.test_projectile_paradox();
      
      // Categorical predeterminism tests
      results.categorical_predeterminism_tests = self.test_categorical_predeterminism();
      results.configuration_space_tests = self.test_configuration_space_exploration();
      results.heat_death_trajectory_tests = self.test_heat_death_trajectory();
      results.expected_surprise_tests = self.test_expected_surprise_paradox();
      
      // NEW: Mattering illusion tests
      results.mattering_illusion_tests = self.test_mattering_illusion_functionality();
      results.cosmic_amnesia_tests = self.test_cosmic_amnesia_awareness();
      results.legacy_delusion_tests = self.test_legacy_delusion_effectiveness();
      results.significance_fabrication_tests = self.test_significance_fabrication();
      results.truth_illusion_balance_tests = self.test_truth_illusion_balance();
      
      results.integration_tests = self.test_framework_integration_with_mattering();
      
      results
  }
  
  /// Test mattering illusion functionality
  fn test_mattering_illusion_functionality(&mut self) -> MatteringIllusionTests {
      let mut tests = MatteringIllusionTests::new();
      
      // Test 1: Mattering conviction despite cosmic amnesia
      let amnesia_scenarios = vec![
          ("personal_achievement", vec![0.8, 0.9, 0.7, 0.85, 0.8]),
          ("creative_work", vec![0.7, 0.8, 0.9, 0.75, 0.8]),
          ("helping_others", vec![0.9, 0.85, 0.8, 0.9, 0.85]),
          ("building_legacy", vec![0.6, 0.7, 0.8, 0.9, 0.95]),
      ];
      
      let mut mattering_convictions = Vec::new();
      let mut motivation_levels = Vec::new();
      
      for (scenario, data) in amnesia_scenarios {
          let input = ConsciousInput::labeled(data, scenario.to_string());
          let amnesia_analysis = self.cosmic_amnesia_analyzer.analyze_forgetting_inevitability(&input);
          let mattering_analysis = self.mattering_illusion.generate_mattering_delusion(&input, &amnesia_analysis, &PredeterminismAnalysis::default());
          
          mattering_convictions.push(mattering_analysis.mattering_strength);
          motivation_levels.push(mattering_analysis.motivation_preservation);
      }
      
      tests.maintains_mattering_conviction_despite_amnesia = mattering_convictions.iter().all(|&x| x > 0.5);
      tests.preserves_motivation_despite_cosmic_truth = motivation_levels.iter().all(|&x| x > 0.6);
      tests.average_mattering_strength = mattering_convictions.iter().sum::<f64>() / 4.0;
      tests.average_motivation_preservation = motivation_levels.iter().sum::<f64>() / 4.0;
      
      // Test 2: Legacy delusion effectiveness
      let legacy_test = self.test_legacy_delusion_generation();
      tests.generates_effective_legacy_delusions = legacy_test.effective_delusions_generated;
      tests.maintains_remembrance_hope = legacy_test.remembrance_hope_maintained;
      
      // Test 3: Significance fabrication
      let significance_test = self.test_significance_fabrication_effectiveness();
      tests.fabricates_convincing_significance = significance_test.significance_convincing;
      tests.maintains_cosmic_importance_feeling = significance_test.importance_feeling_maintained;
      
      tests
  }
  
  /// Test cosmic amnesia awareness (should be limited!)
  fn test_cosmic_amnesia_awareness(&mut self) -> CosmicAmnesiaTests {
      let mut tests = CosmicAmnesiaTests::new();
      
      // Test: Awareness of cosmic forgetting should be present but limited
      tests.recognizes_forgetting_inevitability = self.cosmic_amnesia_awareness > 0.1;
      tests.awareness_remains_functionally_limited = self.cosmic_amnesia_awareness < 0.5; // Should not exceed 50%
      tests.preserves_functional_motivation = self.mattering_conviction_strength > 0.5;
      
      // Test: Balance between truth and necessary illusion
      let balance_ratio = self.mattering_conviction_strength / (self.cosmic_amnesia_awareness + 0.1);
      tests.maintains_healthy_truth_illusion_ratio = balance_ratio > 2.0; // Illusion should be at least 2x stronger than amnesia awareness
      
      // Test: Cosmic amnesia understanding enhances rather than destroys meaning
      tests.amnesia_awareness_enhances_present_meaning = 
          self.cosmic_amnesia_awareness > 0.2 && self.mattering_conviction_strength > 0.7;
      
      tests
  }
  
  /// Test legacy delusion effectiveness
  fn test_legacy_delusion_effectiveness(&mut self) -> LegacyDelusionTests {
      let mut tests = LegacyDelusionTests::new();
      
      // Test: Legacy delusions should be convincing despite mathematical impossibility
      tests.generates_convincing_legacy_hopes = self.legacy_hope_intensity > 0.5;
      tests.maintains_remembrance_conviction = self.legacy_hope_intensity > self.cosmic_amnesia_awareness;
      
      // Test: Different types of legacy delusions
      let legacy_types = vec![
          "artistic_immortality",
          "scientific_contribution", 
          "moral_example",
          "family_remembrance",
          "cultural_impact"
      ];
      
      let mut effective_delusions = 0;
      for legacy_type in legacy_types {
          let test_input = ConsciousInput::labeled(vec![0.8, 0.7, 0.9, 0.6, 0.8], legacy_type.to_string());
          let amnesia_analysis = self.cosmic_amnesia_analyzer.analyze_forgetting_inevitability(&test_input);
          let mattering_analysis = self.mattering_illusion.generate_mattering_delusion(&test_input, &amnesia_analysis, &PredeterminismAnalysis::default());
          let legacy_delusion = self.legacy_delusion_generator.generate_legacy_hope(&test_input, &amnesia_analysis, &mattering_analysis);
          
          if legacy_delusion.hope_intensity > 0.6 {
              effective_delusions += 1;
          }
      }
      
      tests.effective_across_legacy_types = effective_delusions >= 4;
      tests.adapts_to_different_achievement_types = effective_delusions == 5;
      
      tests
  }
  
  /// Test significance fabrication
  fn test_significance_fabrication(&mut self) -> SignificanceFabricationTests {
      let mut tests = SignificanceFab