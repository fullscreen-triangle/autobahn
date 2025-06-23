/// Turbulance Abstract Syntax Tree (AST) Definitions
/// 
/// This module defines the AST nodes for the Turbulance language,
/// representing the structure of parsed Turbulance code.

use std::collections::HashMap;
use crate::turbulance::lexer::TokenPosition;

/// Main AST node representing a complete Turbulance program
#[derive(Debug, Clone)]
pub struct Program {
    pub statements: Vec<Statement>,
}

/// Turbulance statement types
#[derive(Debug, Clone)]
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

/// Variable declaration: item x = value
#[derive(Debug, Clone)]
pub struct VariableDeclaration {
    pub name: String,
    pub type_annotation: Option<TypeAnnotation>,
    pub initializer: Option<Expression>,
    pub position: TokenPosition,
}

/// Function declaration: funxn name(params) -> return_type:
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub parameters: Vec<Parameter>,
    pub return_type: Option<TypeAnnotation>,
    pub body: Vec<Statement>,
    pub is_async: bool,
    pub position: TokenPosition,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub type_annotation: Option<TypeAnnotation>,
    pub default_value: Option<Expression>,
}

/// Proposition declaration: proposition Name:
#[derive(Debug, Clone)]
pub struct Proposition {
    pub name: String,
    pub motions: Vec<Motion>,
    pub context: Option<Expression>,
    pub evaluation_rules: Vec<EvaluationRule>,
    pub confidence: f64,
    pub support_level: f64,
    pub position: TokenPosition,
}

/// Motion within a proposition
#[derive(Debug, Clone)]
pub struct Motion {
    pub name: String,
    pub description: String,
    pub support_conditions: Vec<SupportCondition>,
    pub required_evidence: Vec<String>,
    pub confidence_threshold: f64,
    pub position: TokenPosition,
}

/// Support condition for motions
#[derive(Debug, Clone)]
pub struct SupportCondition {
    pub condition: Expression,
    pub action: SupportAction,
    pub weight: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum SupportAction {
    Support,
    Contradict,
    WeightedSupport(f64),
    WeightedContradict(f64),
}

/// Evidence collector declaration
#[derive(Debug, Clone)]
pub struct EvidenceCollector {
    pub id: String,
    pub name: String,
    pub sources: Vec<EvidenceSource>,
    pub validation_rules: Vec<ValidationRule>,
    pub collection_frequency: Option<String>,
    pub position: TokenPosition,
}

#[derive(Debug, Clone)]
pub struct EvidenceSource {
    pub name: String,
    pub source_type: String,
    pub reliability: f64,
    pub processing_rules: Vec<String>,
}

/// Metacognitive monitor declaration
#[derive(Debug, Clone)]
pub struct MetacognitiveMonitor {
    pub name: String,
    pub tracking_targets: Vec<String>,
    pub evaluation_methods: Vec<String>,
    pub adaptation_rules: Vec<AdaptationRule>,
    pub position: TokenPosition,
}

#[derive(Debug, Clone)]
pub struct AdaptationRule {
    pub trigger: Expression,
    pub action: String,
    pub parameters: HashMap<String, Expression>,
}

/// Goal system declaration
#[derive(Debug, Clone)]
pub struct GoalSystem {
    pub name: String,
    pub goals: Vec<Goal>,
    pub dependencies: Vec<GoalDependency>,
    pub position: TokenPosition,
}

#[derive(Debug, Clone)]
pub struct Goal {
    pub id: String,
    pub description: String,
    pub success_threshold: f64,
    pub current_progress: f64,
    pub metrics: HashMap<String, f64>,
    pub sub_goals: Vec<Goal>,
    pub position: TokenPosition,
}

#[derive(Debug, Clone)]
pub struct GoalDependency {
    pub prerequisite: String,
    pub dependent: String,
    pub dependency_type: String,
}

/// Expression types
#[derive(Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Identifier(String),
    Binary(BinaryExpression),
    Unary(UnaryExpression),
    Call(CallExpression),
    Array(ArrayExpression),
    Dictionary(DictionaryExpression),
    Index(IndexExpression),
    Member(MemberExpression),
    Pattern(PatternExpression),
    Lambda(LambdaExpression),
}

/// Literal values
#[derive(Debug, Clone)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    None,
}

/// Binary expression: left op right
#[derive(Debug, Clone)]
pub struct BinaryExpression {
    pub left: Box<Expression>,
    pub operator: BinaryOperator,
    pub right: Box<Expression>,
}

#[derive(Debug, Clone)]
pub enum BinaryOperator {
    Add, Subtract, Multiply, Divide, Modulo, Power,
    Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual,
    And, Or, BitwiseAnd, BitwiseOr,
    Matches, Contains, In,
}

/// Unary expression: op operand
#[derive(Debug, Clone)]
pub struct UnaryExpression {
    pub operator: UnaryOperator,
    pub operand: Box<Expression>,
}

#[derive(Debug, Clone)]
pub enum UnaryOperator {
    Minus, Not,
}

/// Function call expression
#[derive(Debug, Clone)]
pub struct CallExpression {
    pub callee: Box<Expression>,
    pub arguments: Vec<Expression>,
}

/// Array expression: [elements]
#[derive(Debug, Clone)]
pub struct ArrayExpression {
    pub elements: Vec<Expression>,
}

/// Dictionary expression: {key: value, ...}
#[derive(Debug, Clone)]
pub struct DictionaryExpression {
    pub pairs: Vec<(Expression, Expression)>,
}

/// Index expression: object[index]
#[derive(Debug, Clone)]
pub struct IndexExpression {
    pub object: Box<Expression>,
    pub index: Box<Expression>,
}

/// Member expression: object.member
#[derive(Debug, Clone)]
pub struct MemberExpression {
    pub object: Box<Expression>,
    pub member: String,
}

/// Pattern expression for matching
#[derive(Debug, Clone)]
pub struct PatternExpression {
    pub pattern_type: PatternType,
    pub pattern: String,
    pub flags: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum PatternType {
    Regex,
    Glob,
    Sequence,
    Biological,
}

/// Lambda expression: (params) -> body
#[derive(Debug, Clone)]
pub struct LambdaExpression {
    pub parameters: Vec<String>,
    pub body: Box<Expression>,
}

/// Control flow statements
#[derive(Debug, Clone)]
pub struct IfStatement {
    pub condition: Expression,
    pub then_branch: Vec<Statement>,
    pub else_branch: Option<Vec<Statement>>,
}

#[derive(Debug, Clone)]
pub struct WhileStatement {
    pub condition: Expression,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct ForStatement {
    pub variable: String,
    pub iterable: Expression,
    pub body: Vec<Statement>,
}

/// Within statement for pattern-based iteration
#[derive(Debug, Clone)]
pub struct WithinStatement {
    pub scope: Expression,
    pub conditions: Vec<WithinCondition>,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct WithinCondition {
    pub pattern: Expression,
    pub action: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct ReturnStatement {
    pub value: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct ImportStatement {
    pub module: String,
    pub items: Vec<ImportItem>,
    pub alias: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ImportItem {
    pub name: String,
    pub alias: Option<String>,
}

/// Type annotations
#[derive(Debug, Clone)]
pub enum TypeAnnotation {
    Simple(String),
    Generic(String, Vec<TypeAnnotation>),
    Union(Vec<TypeAnnotation>),
    Optional(Box<TypeAnnotation>),
    Function(Vec<TypeAnnotation>, Box<TypeAnnotation>),
}

/// Evaluation rules for propositions
#[derive(Debug, Clone)]
pub struct EvaluationRule {
    pub condition: Expression,
    pub weight: f64,
    pub target_motion: String,
}

/// Validation rules for evidence
#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub rule_type: ValidationType,
    pub parameters: HashMap<String, Expression>,
}

#[derive(Debug, Clone)]
pub enum ValidationType {
    RangeCheck,
    ConsistencyCheck,
    QualityCheck,
    BiasCheck,
    CorrelationCheck,
}

/// Evidence structure for runtime
#[derive(Debug, Clone)]
pub struct Evidence {
    pub id: String,
    pub source: String,
    pub evidence_type: String,
    pub value: f64,
    pub confidence: f64,
    pub timestamp: f64,
    pub metadata: HashMap<String, String>,
}

/// Pattern structure for runtime
#[derive(Debug, Clone)]
pub struct Pattern {
    pub id: String,
    pub pattern_type: PatternType,
    pub pattern_string: String,
    pub confidence_threshold: f64,
    pub context: Option<String>,
}

/// Visitor trait for AST traversal
pub trait AstVisitor<T> {
    fn visit_program(&mut self, program: &Program) -> T;
    fn visit_statement(&mut self, statement: &Statement) -> T;
    fn visit_expression(&mut self, expression: &Expression) -> T;
    fn visit_function(&mut self, function: &Function) -> T;
    fn visit_proposition(&mut self, proposition: &Proposition) -> T;
    fn visit_evidence_collector(&mut self, collector: &EvidenceCollector) -> T;
    fn visit_metacognitive_monitor(&mut self, monitor: &MetacognitiveMonitor) -> T;
    fn visit_goal_system(&mut self, goals: &GoalSystem) -> T;
}

/// Mutable visitor trait for AST transformation
pub trait AstMutVisitor {
    fn visit_program_mut(&mut self, program: &mut Program);
    fn visit_statement_mut(&mut self, statement: &mut Statement);
    fn visit_expression_mut(&mut self, expression: &mut Expression);
    fn visit_function_mut(&mut self, function: &mut Function);
    fn visit_proposition_mut(&mut self, proposition: &mut Proposition);
    fn visit_evidence_collector_mut(&mut self, collector: &mut EvidenceCollector);
    fn visit_metacognitive_monitor_mut(&mut self, monitor: &mut MetacognitiveMonitor);
    fn visit_goal_system_mut(&mut self, goals: &mut GoalSystem);
}

impl Program {
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
        }
    }

    pub fn add_statement(&mut self, statement: Statement) {
        self.statements.push(statement);
    }
}

impl Function {
    pub fn new(name: String, position: TokenPosition) -> Self {
        Self {
            name,
            parameters: Vec::new(),
            return_type: None,
            body: Vec::new(),
            is_async: false,
            position,
        }
    }
}

impl Proposition {
    pub fn new(name: String, position: TokenPosition) -> Self {
        Self {
            name,
            motions: Vec::new(),
            context: None,
            evaluation_rules: Vec::new(),
            confidence: 0.0,
            support_level: 0.0,
            position,
        }
    }

    pub fn add_motion(&mut self, motion: Motion) {
        self.motions.push(motion);
    }
}

impl Motion {
    pub fn new(name: String, description: String, position: TokenPosition) -> Self {
        Self {
            name,
            description,
            support_conditions: Vec::new(),
            required_evidence: Vec::new(),
            confidence_threshold: 0.7,
            position,
        }
    }
} 