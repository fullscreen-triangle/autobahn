/// Turbulance Language Compiler/Parser for Autobahn
/// 
/// This module provides a complete compiler and parser for the Turbulance language,
/// converting scientific method encoding syntax into executable autobahn operations.

pub mod lexer;
pub mod parser;
pub mod ast;
pub mod compiler;
pub mod executor;
pub mod integration;

use crate::types::*;
use std::collections::HashMap;

/// Main Turbulance processor for autobahn
#[derive(Debug, Clone)]
pub struct TurbulanceProcessor {
    pub lexer: lexer::TurbulanceLexer,
    pub parser: parser::TurbulanceParser,
    pub compiler: compiler::TurbulanceCompiler,
    pub executor: executor::TurbulanceExecutor,
    pub symbol_table: HashMap<String, TurbulanceValue>,
    pub active_propositions: Vec<ast::Proposition>,
    pub evidence_collectors: Vec<ast::EvidenceCollector>,
    pub metacognitive_monitors: Vec<ast::MetacognitiveMonitor>,
    pub goal_systems: Vec<ast::GoalSystem>,
}

/// Turbulance value types that can be used in autobahn
#[derive(Debug, Clone)]
pub enum TurbulanceValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Array(Vec<TurbulanceValue>),
    Dictionary(HashMap<String, TurbulanceValue>),
    Function(ast::Function),
    Proposition(ast::Proposition),
    Evidence(ast::Evidence),
    Pattern(ast::Pattern),
    Goal(ast::Goal),
    BiologicalEntity(BiologicalEntity),
    QuantumState(QuantumState),
    EnergyState(EnergyState),
}

/// Results from executing Turbulance code
#[derive(Debug, Clone)]
pub struct TurbulanceExecutionResult {
    pub success: bool,
    pub return_value: Option<TurbulanceValue>,
    pub side_effects: Vec<SideEffect>,
    pub evidence_generated: Vec<ast::Evidence>,
    pub propositions_evaluated: Vec<PropositionEvaluation>,
    pub goals_updated: Vec<GoalUpdate>,
    pub errors: Vec<TurbulanceError>,
}

#[derive(Debug, Clone)]
pub enum SideEffect {
    VariableAssignment(String, TurbulanceValue),
    FunctionCall(String, Vec<TurbulanceValue>),
    PropositionEvaluation(String, f64),
    EvidenceCollection(String, ast::Evidence),
    GoalProgress(String, f64),
    BiologicalOperation(String, BiologicalOperationResult),
}

#[derive(Debug, Clone)]
pub struct PropositionEvaluation {
    pub name: String,
    pub motions: Vec<MotionEvaluation>,
    pub overall_support: f64,
    pub confidence: f64,
    pub evidence_quality: f64,
}

#[derive(Debug, Clone)]
pub struct MotionEvaluation {
    pub name: String,
    pub support_level: f64,
    pub confidence: f64,
    pub evidence_used: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GoalUpdate {
    pub goal_id: String,
    pub previous_progress: f64,
    pub new_progress: f64,
    pub metrics_updated: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum TurbulanceError {
    LexicalError(String),
    ParseError(String),
    CompileError(String),
    RuntimeError(String),
    TypeMismatch(String, String),
    UndefinedVariable(String),
    InvalidOperation(String),
    PropositionError(String),
    EvidenceError(String),
    GoalError(String),
}

impl TurbulanceProcessor {
    /// Create a new Turbulance processor
    pub fn new() -> Self {
        Self {
            lexer: lexer::TurbulanceLexer::new(),
            parser: parser::TurbulanceParser::new(),
            compiler: compiler::TurbulanceCompiler::new(),
            executor: executor::TurbulanceExecutor::new(),
            symbol_table: HashMap::new(),
            active_propositions: Vec::new(),
            evidence_collectors: Vec::new(),
            metacognitive_monitors: Vec::new(),
            goal_systems: Vec::new(),
        }
    }

    /// Process Turbulance code and execute it in autobahn context
    pub fn process_turbulance_code(&mut self, source_code: &str) -> TurbulanceExecutionResult {
        // Step 1: Lexical analysis
        let tokens = match self.lexer.tokenize(source_code) {
            Ok(tokens) => tokens,
            Err(e) => return TurbulanceExecutionResult {
                success: false,
                return_value: None,
                side_effects: Vec::new(),
                evidence_generated: Vec::new(),
                propositions_evaluated: Vec::new(),
                goals_updated: Vec::new(),
                errors: vec![TurbulanceError::LexicalError(e)],
            }
        };

        // Step 2: Parsing
        let ast = match self.parser.parse(tokens) {
            Ok(ast) => ast,
            Err(e) => return TurbulanceExecutionResult {
                success: false,
                return_value: None,
                side_effects: Vec::new(),
                evidence_generated: Vec::new(),
                propositions_evaluated: Vec::new(),
                goals_updated: Vec::new(),
                errors: vec![TurbulanceError::ParseError(e)],
            }
        };

        // Step 3: Compilation
        let executable = match self.compiler.compile(ast) {
            Ok(executable) => executable,
            Err(e) => return TurbulanceExecutionResult {
                success: false,
                return_value: None,
                side_effects: Vec::new(),
                evidence_generated: Vec::new(),
                propositions_evaluated: Vec::new(),
                goals_updated: Vec::new(),
                errors: vec![TurbulanceError::CompileError(e)],
            }
        };

        // Step 4: Execution
        match self.executor.execute(executable, &mut self.symbol_table) {
            Ok(result) => {
                self.update_internal_state(&result);
                result
            },
            Err(e) => TurbulanceExecutionResult {
                success: false,
                return_value: None,
                side_effects: Vec::new(),
                evidence_generated: Vec::new(),
                propositions_evaluated: Vec::new(),
                goals_updated: Vec::new(),
                errors: vec![TurbulanceError::RuntimeError(e)],
            }
        }
    }

    /// Process a Turbulance file
    pub fn process_file(&mut self, filename: &str) -> Result<TurbulanceExecutionResult, std::io::Error> {
        let source_code = std::fs::read_to_string(filename)?;
        Ok(self.process_turbulance_code(&source_code))
    }

    /// Evaluate a Turbulance expression interactively
    pub fn eval_expression(&mut self, expression: &str) -> TurbulanceExecutionResult {
        // Wrap expression in a minimal program structure
        let wrapped_code = format!("item result = {}", expression);
        self.process_turbulance_code(&wrapped_code)
    }

    /// Get current state of propositions
    pub fn get_active_propositions(&self) -> &[ast::Proposition] {
        &self.active_propositions
    }

    /// Get current evidence collectors
    pub fn get_evidence_collectors(&self) -> &[ast::EvidenceCollector] {
        &self.evidence_collectors
    }

    /// Get current goal systems
    pub fn get_goal_systems(&self) -> &[ast::GoalSystem] {
        &self.goal_systems
    }

    /// Register a new biological entity for Turbulance operations
    pub fn register_biological_entity(&mut self, name: String, entity: BiologicalEntity) {
        self.symbol_table.insert(name, TurbulanceValue::BiologicalEntity(entity));
    }

    /// Register a quantum state for Turbulance operations
    pub fn register_quantum_state(&mut self, name: String, state: QuantumState) {
        self.symbol_table.insert(name, TurbulanceValue::QuantumState(state));
    }

    /// Create a new proposition from Turbulance code
    pub fn create_proposition(&mut self, name: &str, definition: &str) -> Result<(), TurbulanceError> {
        let proposition_code = format!("proposition {}:\n{}", name, definition);
        let result = self.process_turbulance_code(&proposition_code);
        
        if result.success {
            Ok(())
        } else {
            Err(result.errors.into_iter().next().unwrap_or(
                TurbulanceError::PropositionError("Failed to create proposition".to_string())
            ))
        }
    }

    /// Update internal state based on execution results
    fn update_internal_state(&mut self, result: &TurbulanceExecutionResult) {
        // Update active propositions
        for prop_eval in &result.propositions_evaluated {
            if let Some(existing) = self.active_propositions.iter_mut()
                .find(|p| p.name == prop_eval.name) {
                existing.confidence = prop_eval.confidence;
                existing.support_level = prop_eval.overall_support;
            }
        }

        // Update goal systems
        for goal_update in &result.goals_updated {
            for goal_system in &mut self.goal_systems {
                if let Some(goal) = goal_system.goals.iter_mut()
                    .find(|g| g.id == goal_update.goal_id) {
                    goal.current_progress = goal_update.new_progress;
                    for (metric, value) in &goal_update.metrics_updated {
                        goal.metrics.insert(metric.clone(), *value);
                    }
                }
            }
        }

        // Add new evidence
        for evidence in &result.evidence_generated {
            if !self.evidence_collectors.iter().any(|ec| ec.id == evidence.source) {
                // This is simplified - in practice, we'd create proper evidence collectors
            }
        }
    }
}

/// Integration traits for connecting Turbulance with autobahn systems
pub trait TurbulanceIntegration {
    fn to_turbulance_value(&self) -> TurbulanceValue;
    fn from_turbulance_value(value: TurbulanceValue) -> Result<Self, TurbulanceError> where Self: Sized;
}

/// Biological entity wrapper for Turbulance integration
#[derive(Debug, Clone)]
pub struct BiologicalEntity {
    pub id: String,
    pub entity_type: String,
    pub properties: HashMap<String, f64>,
    pub state: EntityState,
}

#[derive(Debug, Clone)]
pub enum EntityState {
    Active,
    Inactive,
    Processing,
    Analyzing,
}

/// Quantum state wrapper for Turbulance integration
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub amplitude: f64,
    pub phase: f64,
    pub entanglement_partners: Vec<String>,
    pub coherence_time: f64,
}

/// Energy state wrapper for Turbulance integration
#[derive(Debug, Clone)]
pub struct EnergyState {
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub total_energy: f64,
    pub entropy: f64,
    pub temperature: f64,
}

/// Result of a biological operation executed through Turbulance
#[derive(Debug, Clone)]
pub struct BiologicalOperationResult {
    pub operation_type: String,
    pub success: bool,
    pub energy_change: f64,
    pub information_generated: f64,
    pub entropy_change: f64,
    pub products: Vec<String>,
}

// Export the main processor
pub use TurbulanceProcessor as Processor; 