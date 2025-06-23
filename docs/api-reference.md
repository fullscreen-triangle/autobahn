---
layout: default
title: "API Reference"
description: "Complete API reference for the Turbulance compiler system and biological integration interfaces"
---

# Turbulance API Reference

This document provides comprehensive API reference documentation for the Turbulance compiler system, including core language APIs, biological integration interfaces, and system management functions.

---

## Table of Contents

1. [Core Compiler API](#core-compiler-api)
2. [Language Processor](#language-processor)
3. [Biological Integration API](#biological-integration-api)
4. [Scientific Method APIs](#scientific-method-apis)
5. [Quantum System Interface](#quantum-system-interface)
6. [Energy Management API](#energy-management-api)
7. [Monitoring and Diagnostics](#monitoring-and-diagnostics)
8. [Utility Functions](#utility-functions)

---

## Core Compiler API

### TurbulanceProcessor

The main interface for processing Turbulance code and managing execution.

#### Constructor

```rust
impl TurbulanceProcessor {
    pub fn new() -> Self
}
```

Creates a new Turbulance processor with default configuration.

**Returns**: `TurbulanceProcessor` instance

#### Core Methods

##### process_turbulance_code

```rust
pub fn process_turbulance_code(&mut self, code: &str) -> TurbulanceExecutionResult
```

Processes and executes Turbulance source code.

**Parameters**:
- `code: &str` - Turbulance source code string

**Returns**: `TurbulanceExecutionResult` containing execution results

**Example**:
```turbulance
let result = processor.process_turbulance_code(r#"
    item x = 5
    item y = 10
    print("Result: " + str(x + y))
"#);
```

##### create_proposition

```rust
pub fn create_proposition(&mut self, name: &str, proposition_code: &str) -> Result<(), String>
```

Creates a scientific proposition for hypothesis testing.

**Parameters**:
- `name: &str` - Unique proposition identifier
- `proposition_code: &str` - Turbulance code defining the proposition

**Returns**: `Result<(), String>` indicating success or error

##### register_biological_entity

```rust
pub fn register_biological_entity(&mut self, entity_id: String, entity: BiologicalEntity)
```

Registers a biological entity for Turbulance control.

**Parameters**:
- `entity_id: String` - Unique entity identifier
- `entity: BiologicalEntity` - Entity configuration

### TurbulanceExecutionResult

Contains the results of Turbulance code execution.

```rust
pub struct TurbulanceExecutionResult {
    pub success: bool,
    pub return_value: Option<TurbulanceValue>,
    pub side_effects: Vec<SideEffect>,
    pub evidence_generated: Vec<Evidence>,
    pub propositions_evaluated: Vec<PropositionEvaluation>,
    pub goals_updated: Vec<GoalUpdate>,
    pub errors: Vec<TurbulanceError>,
}
```

#### Fields

- `success: bool` - Whether execution completed successfully
- `return_value: Option<TurbulanceValue>` - Final return value if any
- `side_effects: Vec<SideEffect>` - List of side effects produced
- `evidence_generated: Vec<Evidence>` - Scientific evidence collected
- `propositions_evaluated: Vec<PropositionEvaluation>` - Proposition evaluation results
- `goals_updated: Vec<GoalUpdate>` - Goal progress updates
- `errors: Vec<TurbulanceError>` - Any errors encountered

---

## Language Processor

### Lexer

Tokenizes Turbulance source code into a stream of tokens.

#### TurbulanceLexer

```rust
impl TurbulanceLexer {
    pub fn new(input: &str) -> Self
    pub fn tokenize(&mut self) -> Result<Vec<Token>, String>
    pub fn next_token(&mut self) -> Result<Token, String>
}
```

##### tokenize

```rust
pub fn tokenize(&mut self) -> Result<Vec<Token>, String>
```

Converts source code into a complete token stream.

**Returns**: `Result<Vec<Token>, String>` containing all tokens or error

#### Token Types

```rust
pub enum TokenType {
    // Keywords
    Item, Funxn, Proposition, Motion, Evidence, Goal, Metacognitive,
    Given, Otherwise, Within, While, For, Return,
    
    // Scientific operators
    Support, Contradict, WithWeight, Matches, Evaluate,
    
    // Biological functions
    ProcessMolecule, HarvestEnergy, ExtractInformation,
    UpdateMembraneState,
    
    // Literals
    Integer(i64), Float(f64), String(String), Boolean(bool),
    
    // Identifiers and operators
    Identifier(String), Plus, Minus, Multiply, Divide,
    Equal, NotEqual, Less, Greater, And, Or, Not,
    
    // Delimiters
    LeftParen, RightParen, LeftBrace, RightBrace,
    LeftBracket, RightBracket, Comma, Colon, Newline, EOF,
}
```

### Parser

Constructs Abstract Syntax Trees from token streams.

#### TurbulanceParser

```rust
impl TurbulanceParser {
    pub fn new(tokens: Vec<Token>) -> Self
    pub fn parse(&mut self) -> Result<Program, String>
    pub fn parse_statement(&mut self) -> Result<Statement, String>
    pub fn parse_expression(&mut self) -> Result<Expression, String>
}
```

##### parse

```rust
pub fn parse(&mut self) -> Result<Program, String>
```

Parses a complete Turbulance program.

**Returns**: `Result<Program, String>` containing AST or parse error

#### AST Node Types

```rust
pub enum Statement {
    VariableDeclaration(VariableDeclaration),
    FunctionDeclaration(Function),
    PropositionDeclaration(Proposition),
    EvidenceDeclaration(EvidenceCollector),
    MetacognitiveDeclaration(MetacognitiveMonitor),
    GoalDeclaration(GoalSystem),
    ExpressionStatement(Expression),
    IfStatement(IfStatement),
    WhileStatement(WhileStatement),
    ForStatement(ForStatement),
    WithinStatement(WithinStatement),
    ReturnStatement(ReturnStatement),
    ImportStatement(ImportStatement),
}
```

### Compiler

Converts AST to executable bytecode.

#### TurbulanceCompiler

```rust
impl TurbulanceCompiler {
    pub fn new() -> Self
    pub fn compile(&mut self, program: Program) -> Result<TurbulanceExecutable, String>
}
```

##### compile

```rust
pub fn compile(&mut self, program: Program) -> Result<TurbulanceExecutable, String>
```

Compiles AST to executable bytecode.

**Parameters**:
- `program: Program` - AST representation of the program

**Returns**: `Result<TurbulanceExecutable, String>` containing bytecode or compilation error

#### TurbulanceExecutable

```rust
pub struct TurbulanceExecutable {
    pub instructions: Vec<Instruction>,
    pub constants: Vec<TurbulanceValue>,
    pub symbol_table: HashMap<String, usize>,
    pub metadata: CompilationMetadata,
}
```

### Executor

Executes compiled Turbulance bytecode.

#### TurbulanceExecutor

```rust
impl TurbulanceExecutor {
    pub fn new() -> Self
    pub fn execute(
        &mut self, 
        executable: TurbulanceExecutable,
        global_variables: &mut HashMap<String, TurbulanceValue>
    ) -> Result<TurbulanceExecutionResult, String>
}
```

---

## Biological Integration API

### AutobahnTurbulanceIntegration

Main integration interface between Turbulance and biological systems.

#### Constructor

```rust
impl AutobahnTurbulanceIntegration {
    pub fn new() -> Self
}
```

#### Core Integration Methods

##### execute_with_integration

```rust
pub fn execute_with_integration(&mut self, turbulance_code: &str) -> Result<IntegrationResult, String>
```

Executes Turbulance code with full biological system integration.

**Parameters**:
- `turbulance_code: &str` - Source code to execute

**Returns**: `Result<IntegrationResult, String>` containing integration results

##### register_bmd_integration

```rust
pub fn register_bmd_integration(&mut self, bmd_config: BMDIntegration) -> Result<(), String>
```

Registers a Biological Maxwell's Demon for Turbulance control.

**Parameters**:
- `bmd_config: BMDIntegration` - BMD configuration

**Returns**: `Result<(), String>` indicating success or error

#### BMDIntegration Configuration

```rust
pub struct BMDIntegration {
    pub demon_id: String,
    pub turbulance_proposition: String,
    pub energy_threshold: f64,
    pub information_extraction_rate: f64,
    pub active_goals: Vec<String>,
}
```

**Fields**:
- `demon_id: String` - Unique identifier for the demon
- `turbulance_proposition: String` - Associated scientific proposition
- `energy_threshold: f64` - Energy threshold for molecule processing (kJ/mol)
- `information_extraction_rate: f64` - Information processing rate (bits/second)
- `active_goals: Vec<String>` - Associated optimization goals

##### register_quantum_integration

```rust
pub fn register_quantum_integration(&mut self, quantum_config: QuantumIntegration) -> Result<(), String>
```

Registers quantum system integration.

**Parameters**:
- `quantum_config: QuantumIntegration` - Quantum system configuration

#### QuantumIntegration Configuration

```rust
pub struct QuantumIntegration {
    pub quantum_id: String,
    pub coherence_monitoring: bool,
    pub entanglement_tracking: bool,
    pub quantum_error_correction: bool,
}
```

##### create_biological_proposition

```rust
pub fn create_biological_proposition(&mut self, name: &str, biological_context: &str) -> Result<(), String>
```

Creates a Turbulance proposition for biological system analysis.

**Parameters**:
- `name: &str` - Proposition name
- `biological_context: &str` - Biological context for the proposition

##### create_optimization_goal

```rust
pub fn create_optimization_goal(&mut self, goal_id: &str, target_efficiency: f64) -> Result<(), String>
```

Creates an optimization goal for system performance.

**Parameters**:
- `goal_id: &str` - Unique goal identifier
- `target_efficiency: f64` - Target efficiency threshold (0.0-1.0)

---

## Scientific Method APIs

### Evidence Collection

#### Evidence Structure

```rust
pub struct Evidence {
    pub id: String,
    pub source: String,
    pub evidence_type: String,
    pub value: f64,
    pub confidence: f64,
    pub timestamp: f64,
    pub metadata: HashMap<String, String>,
}
```

#### Evidence Collection Functions

##### collect_evidence

```turbulance
funxn collect_evidence(source: String) -> Evidence
```

Collects evidence from specified source.

**Parameters**:
- `source: String` - Evidence source identifier

**Returns**: `Evidence` structure containing collected data

##### validate_evidence

```turbulance
funxn validate_evidence(evidence: Evidence) -> Bool
```

Validates evidence quality and consistency.

**Parameters**:
- `evidence: Evidence` - Evidence to validate

**Returns**: `Bool` indicating validation result

### Proposition Management

#### Proposition Structure

```rust
pub struct Proposition {
    pub name: String,
    pub motions: Vec<Motion>,
    pub evidence_requirements: Vec<String>,
    pub support_conditions: Vec<SupportCondition>,
}
```

#### Proposition Functions

##### create_proposition

```turbulance
funxn create_proposition(name: String, motions: Array<Motion>) -> Proposition
```

Creates a new scientific proposition.

**Parameters**:
- `name: String` - Proposition identifier
- `motions: Array<Motion>` - Associated motions (hypotheses)

**Returns**: `Proposition` structure

##### evaluate_proposition

```turbulance
funxn evaluate_proposition(proposition: String) -> Float
```

Evaluates proposition support based on evidence.

**Parameters**:
- `proposition: String` - Proposition name to evaluate

**Returns**: `Float` support level (0.0-1.0)

##### support_motion

```turbulance
funxn support_motion(motion_name: String, weight: Float) -> Bool
```

Adds support to a specific motion.

**Parameters**:
- `motion_name: String` - Motion to support
- `weight: Float` - Support weight (0.0-1.0)

**Returns**: `Bool` indicating success

### Goal Management

#### Goal Structure

```rust
pub struct Goal {
    pub id: String,
    pub description: String,
    pub success_threshold: f64,
    pub current_progress: f64,
    pub metrics: HashMap<String, f64>,
    pub constraints: Vec<String>,
}
```

#### Goal Functions

##### create_goal

```turbulance
funxn create_goal(goal_id: String, threshold: Float) -> Goal
```

Creates a new optimization goal.

**Parameters**:
- `goal_id: String` - Unique goal identifier
- `threshold: Float` - Success threshold (0.0-1.0)

**Returns**: `Goal` structure

##### update_goal_progress

```turbulance
funxn update_goal_progress(goal_id: String, progress: Float) -> Bool
```

Updates goal progress.

**Parameters**:
- `goal_id: String` - Goal to update
- `progress: Float` - New progress value (0.0-1.0)

**Returns**: `Bool` indicating success

##### evaluate_goal

```turbulance
funxn evaluate_goal(goal_id: String) -> Bool
```

Evaluates whether goal has been achieved.

**Parameters**:
- `goal_id: String` - Goal to evaluate

**Returns**: `Bool` indicating achievement status

---

## Quantum System Interface

### Quantum State Management

#### QuantumState Structure

```rust
pub struct QuantumState {
    pub amplitude: f64,
    pub phase: f64,
    pub coherence: f64,
}
```

#### Quantum Functions

##### create_quantum_state

```turbulance
funxn create_quantum_state(amplitude: Float, phase: Float) -> QuantumState
```

Creates a new quantum state.

**Parameters**:
- `amplitude: Float` - State amplitude
- `phase: Float` - State phase (radians)

**Returns**: `QuantumState` structure

##### measure_quantum_state

```turbulance
funxn measure_quantum_state(state: QuantumState) -> Float
```

Performs quantum measurement.

**Parameters**:
- `state: QuantumState` - State to measure

**Returns**: `Float` measurement result

##### apply_quantum_gate

```turbulance
funxn apply_quantum_gate(gate_type: String, state: QuantumState) -> QuantumState
```

Applies quantum gate operation.

**Parameters**:
- `gate_type: String` - Gate type ("hadamard", "pauli_x", "pauli_y", "pauli_z")
- `state: QuantumState` - Input state

**Returns**: `QuantumState` transformed state

##### create_entanglement

```turbulance
funxn create_entanglement(state1: QuantumState, state2: QuantumState) -> Bool
```

Creates quantum entanglement between states.

**Parameters**:
- `state1: QuantumState` - First quantum state
- `state2: QuantumState` - Second quantum state

**Returns**: `Bool` indicating success

##### measure_entanglement

```turbulance
funxn measure_entanglement(state1: QuantumState, state2: QuantumState) -> Float
```

Measures entanglement degree between states.

**Parameters**:
- `state1: QuantumState` - First entangled state
- `state2: QuantumState` - Second entangled state

**Returns**: `Float` entanglement measure (0.0-1.0)

---

## Energy Management API

### Energy Operations

#### Biological Energy Functions

##### process_molecule

```turbulance
funxn process_molecule(molecule_id: String) -> Float
```

Processes a biological molecule for energy extraction.

**Parameters**:
- `molecule_id: String` - Molecule identifier ("glucose", "atp", "nadh", etc.)

**Returns**: `Float` energy extracted (kJ/mol)

##### harvest_energy

```turbulance
funxn harvest_energy(source: String) -> Float
```

Harvests energy from biological processes.

**Parameters**:
- `source: String` - Energy source ("glycolysis", "krebs_cycle", "photosynthesis", etc.)

**Returns**: `Float` energy harvested (kJ/mol)

##### extract_information

```turbulance
funxn extract_information(source: String) -> Float
```

Extracts information content from biological processes.

**Parameters**:
- `source: String` - Information source identifier

**Returns**: `Float` information content (bits)

#### Energy System Management

##### EnergyInterface

```rust
pub struct EnergyInterface {
    pub energy_reservoirs: HashMap<String, f64>,
    pub flow_rates: HashMap<String, f64>,
    pub efficiency_metrics: HashMap<String, f64>,
}
```

##### get_energy_level

```turbulance
funxn get_energy_level(reservoir: String) -> Float
```

Gets current energy level in reservoir.

**Parameters**:
- `reservoir: String` - Energy reservoir identifier

**Returns**: `Float` current energy level

##### set_energy_flow_rate

```turbulance
funxn set_energy_flow_rate(source: String, destination: String, rate: Float) -> Bool
```

Sets energy flow rate between reservoirs.

**Parameters**:
- `source: String` - Source reservoir
- `destination: String` - Destination reservoir  
- `rate: Float` - Flow rate (energy units/time)

**Returns**: `Bool` indicating success

---

## Monitoring and Diagnostics

### System Monitoring

#### Performance Metrics

##### monitor_system_performance

```turbulance
funxn monitor_system_performance() -> Dictionary
```

Monitors overall system performance metrics.

**Returns**: `Dictionary` containing performance data:
- `energy_efficiency: Float` - Overall energy efficiency (0.0-1.0)
- `processing_speed: Float` - Processing throughput metric
- `stability_index: Float` - System stability measure
- `error_rate: Float` - Current error rate

##### monitor_demon_performance

```turbulance
funxn monitor_demon_performance(demon_id: String) -> Dictionary
```

Monitors specific Maxwell's demon performance.

**Parameters**:
- `demon_id: String` - Demon identifier to monitor

**Returns**: `Dictionary` containing demon metrics:
- `energy_extraction_rate: Float` - Energy extraction rate
- `information_processing_rate: Float` - Information processing rate
- `efficiency: Float` - Overall demon efficiency
- `state: String` - Current demon state

#### Diagnostic Functions

##### diagnose_system_health

```turbulance
funxn diagnose_system_health() -> Dictionary
```

Performs comprehensive system health diagnosis.

**Returns**: `Dictionary` containing health metrics:
- `overall_health: Float` - Overall system health score
- `component_status: Dictionary` - Individual component status
- `warnings: Array<String>` - System warnings
- `recommendations: Array<String>` - Optimization recommendations

##### detect_anomalies

```turbulance
funxn detect_anomalies(data: Array, threshold: Float) -> Array
```

Detects anomalies in system data.

**Parameters**:
- `data: Array` - Data to analyze for anomalies
- `threshold: Float` - Anomaly detection threshold

**Returns**: `Array` of detected anomalies

---

## Utility Functions

### Mathematical Functions

##### calculate_entropy

```turbulance
funxn calculate_entropy(probability_distribution: Array<Float>) -> Float
```

Calculates Shannon entropy of probability distribution.

##### calculate_free_energy

```turbulance
funxn calculate_free_energy(enthalpy: Float, entropy: Float, temperature: Float) -> Float
```

Calculates Gibbs free energy.

##### boltzmann_factor

```turbulance
funxn boltzmann_factor(energy: Float, temperature: Float) -> Float
```

Calculates Boltzmann factor for energy state.

### String Processing

##### format_scientific

```turbulance
funxn format_scientific(value: Float, precision: Integer) -> String
```

Formats number in scientific notation.

##### parse_molecule_formula

```turbulance
funxn parse_molecule_formula(formula: String) -> Dictionary
```

Parses molecular formula into atomic composition.

### Data Analysis

##### statistical_summary

```turbulance
funxn statistical_summary(data: Array<Float>) -> Dictionary
```

Calculates statistical summary of data.

**Returns**: Dictionary containing:
- `mean: Float` - Arithmetic mean
- `median: Float` - Median value
- `std_dev: Float` - Standard deviation
- `min: Float` - Minimum value
- `max: Float` - Maximum value

##### correlation_analysis

```turbulance
funxn correlation_analysis(x_data: Array<Float>, y_data: Array<Float>) -> Float
```

Calculates Pearson correlation coefficient.

---

## Error Handling

### Error Types

```rust
pub enum TurbulanceError {
    SyntaxError(String),
    RuntimeError(String),
    TypeMismatch(String, String),
    BiologicalSystemError(String),
    QuantumError(String),
    EnergySystemError(String),
    ScientificMethodError(String),
}
```

### Error Handling Functions

##### handle_error

```turbulance
funxn handle_error(error: TurbulanceError) -> String
```

Handles and formats error messages.

##### try_catch

```turbulance
try:
    # Potentially failing operations
catch ErrorType as e:
    # Error handling
finally:
    # Cleanup operations
```

Error handling construct for robust code execution.

---

This API reference provides comprehensive documentation for all Turbulance system interfaces. For practical usage examples, see the [Examples](/examples) section.

---

**© 2024 Autobahn Biological Computing Project. All rights reserved.** 