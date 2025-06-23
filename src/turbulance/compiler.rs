/// Turbulance Compiler - Converts AST into executable operations
/// 
/// This compiler transforms the parsed Turbulance AST into executable
/// operations that can be run within the autobahn biological computing system.

use crate::turbulance::ast::*;
use crate::turbulance::{TurbulanceValue, TurbulanceError};
use std::collections::HashMap;

/// Executable representation of compiled Turbulance code
#[derive(Debug, Clone)]
pub struct TurbulanceExecutable {
    pub instructions: Vec<Instruction>,
    pub constants: Vec<TurbulanceValue>,
    pub symbol_table: HashMap<String, usize>, // name -> instruction index
    pub metadata: CompilationMetadata,
}

/// Individual executable instruction
#[derive(Debug, Clone)]
pub enum Instruction {
    // Basic operations
    LoadConstant(usize),              // Load constant by index
    LoadVariable(String),             // Load variable by name
    StoreVariable(String),            // Store to variable
    
    // Arithmetic operations
    Add, Subtract, Multiply, Divide, Modulo, Power,
    
    // Comparison operations
    Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual,
    
    // Logical operations
    And, Or, Not,
    
    // Control flow
    Jump(usize),                      // Unconditional jump
    JumpIfFalse(usize),              // Jump if top of stack is false
    JumpIfTrue(usize),               // Jump if top of stack is true
    
    // Function operations
    Call(String, usize),             // Call function with arg count
    Return,                          // Return from function
    
    // Scientific operations
    CreateProposition(String),        // Create proposition
    CreateMotion(String, String),     // Create motion with description
    EvaluateProposition(String),      // Evaluate proposition
    CollectEvidence(String),          // Collect evidence
    SupportMotion(String, f64),       // Support motion with weight
    ContradictMotion(String, f64),    // Contradict motion with weight
    
    // Pattern operations
    MatchPattern(String),             // Pattern matching
    WithinScope(String),              // Enter scope
    ExitScope,                        // Exit scope
    
    // Goal operations
    CreateGoal(String, f64),          // Create goal with threshold
    UpdateGoalProgress(String, f64),  // Update goal progress
    EvaluateGoal(String),            // Evaluate goal achievement
    
    // Metacognitive operations
    TrackReasoning(String),          // Track reasoning step
    EvaluateConfidence,              // Evaluate confidence level
    DetectBias(String),              // Detect specific bias type
    AdaptBehavior(String),           // Adapt behavior based on metacognition
    
    // Biological operations
    ProcessMolecule(String),         // Process biological molecule
    HarvestEnergy(String),           // Harvest energy from process
    ExtractInformation(String),      // Extract information content
    UpdateMembraneState(String),     // Update membrane interface
    
    // Stack operations
    Pop,                             // Remove top stack item
    Duplicate,                       // Duplicate top stack item
    Swap,                           // Swap top two stack items
    
    // Special operations  
    Print,                          // Print top of stack
    Halt,                           // Stop execution
}

/// Compilation metadata
#[derive(Debug, Clone)]
pub struct CompilationMetadata {
    pub source_file: String,
    pub compilation_time: f64,
    pub optimization_level: usize,
    pub warnings: Vec<String>,
}

/// Turbulance compiler
pub struct TurbulanceCompiler {
    instructions: Vec<Instruction>,
    constants: Vec<TurbulanceValue>,
    symbol_table: HashMap<String, usize>,
    label_stack: Vec<usize>,
    current_function: Option<String>,
    optimization_enabled: bool,
}

impl TurbulanceCompiler {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            constants: Vec::new(),
            symbol_table: HashMap::new(),
            label_stack: Vec::new(),
            current_function: None,
            optimization_enabled: true,
        }
    }

    pub fn compile(&mut self, program: Program) -> Result<TurbulanceExecutable, String> {
        self.reset();
        
        // Compile all statements
        for statement in program.statements {
            self.compile_statement(statement)?;
        }
        
        // Add halt instruction
        self.emit(Instruction::Halt);
        
        // Apply optimizations if enabled
        if self.optimization_enabled {
            self.optimize()?;
        }
        
        Ok(TurbulanceExecutable {
            instructions: self.instructions.clone(),
            constants: self.constants.clone(),
            symbol_table: self.symbol_table.clone(),
            metadata: CompilationMetadata {
                source_file: "turbulance_source".to_string(),
                compilation_time: 0.0, // TODO: add actual timestamp
                optimization_level: if self.optimization_enabled { 1 } else { 0 },
                warnings: Vec::new(),
            },
        })
    }

    fn reset(&mut self) {
        self.instructions.clear();
        self.constants.clear();
        self.symbol_table.clear();
        self.label_stack.clear();
        self.current_function = None;
    }

    fn compile_statement(&mut self, statement: Statement) -> Result<(), String> {
        match statement {
            Statement::VariableDeclaration(var_decl) => {
                if let Some(initializer) = var_decl.initializer {
                    self.compile_expression(initializer)?;
                    self.emit(Instruction::StoreVariable(var_decl.name));
                }
                Ok(())
            }
            
            Statement::FunctionDeclaration(func) => {
                self.compile_function(func)
            }
            
            Statement::PropositionDeclaration(prop) => {
                self.compile_proposition(prop)
            }
            
            Statement::EvidenceDeclaration(evidence) => {
                self.compile_evidence_collector(evidence)
            }
            
            Statement::MetacognitiveDeclaration(meta) => {
                self.compile_metacognitive_monitor(meta)
            }
            
            Statement::GoalDeclaration(goal_sys) => {
                self.compile_goal_system(goal_sys)
            }
            
            Statement::ExpressionStatement(expr) => {
                self.compile_expression(expr)?;
                self.emit(Instruction::Pop); // Remove unused result
                Ok(())
            }
            
            Statement::IfStatement(if_stmt) => {
                self.compile_if_statement(if_stmt)
            }
            
            Statement::WhileStatement(while_stmt) => {
                self.compile_while_statement(while_stmt)
            }
            
            Statement::ForStatement(for_stmt) => {
                self.compile_for_statement(for_stmt)
            }
            
            Statement::WithinStatement(within_stmt) => {
                self.compile_within_statement(within_stmt)
            }
            
            Statement::ReturnStatement(ret_stmt) => {
                if let Some(value) = ret_stmt.value {
                    self.compile_expression(value)?;
                } else {
                    // Return None/null
                    let none_idx = self.add_constant(TurbulanceValue::String("None".to_string()));
                    self.emit(Instruction::LoadConstant(none_idx));
                }
                self.emit(Instruction::Return);
                Ok(())
            }
            
            Statement::ImportStatement(_) => {
                // Import handling would be more complex in a real implementation
                Ok(())
            }
        }
    }

    fn compile_expression(&mut self, expression: Expression) -> Result<(), String> {
        match expression {
            Expression::Literal(literal) => {
                let value = match literal {
                    Literal::Integer(i) => TurbulanceValue::Integer(i),
                    Literal::Float(f) => TurbulanceValue::Float(f),
                    Literal::String(s) => TurbulanceValue::String(s),
                    Literal::Boolean(b) => TurbulanceValue::Boolean(b),
                    Literal::None => TurbulanceValue::String("None".to_string()),
                };
                let const_idx = self.add_constant(value);
                self.emit(Instruction::LoadConstant(const_idx));
                Ok(())
            }
            
            Expression::Identifier(name) => {
                self.emit(Instruction::LoadVariable(name));
                Ok(())
            }
            
            Expression::Binary(binary) => {
                self.compile_expression(*binary.left)?;
                self.compile_expression(*binary.right)?;
                
                let instruction = match binary.operator {
                    BinaryOperator::Add => Instruction::Add,
                    BinaryOperator::Subtract => Instruction::Subtract,
                    BinaryOperator::Multiply => Instruction::Multiply,
                    BinaryOperator::Divide => Instruction::Divide,
                    BinaryOperator::Modulo => Instruction::Modulo,
                    BinaryOperator::Power => Instruction::Power,
                    BinaryOperator::Equal => Instruction::Equal,
                    BinaryOperator::NotEqual => Instruction::NotEqual,
                    BinaryOperator::Less => Instruction::Less,
                    BinaryOperator::Greater => Instruction::Greater,
                    BinaryOperator::LessEqual => Instruction::LessEqual,
                    BinaryOperator::GreaterEqual => Instruction::GreaterEqual,
                    BinaryOperator::And => Instruction::And,
                    BinaryOperator::Or => Instruction::Or,
                    BinaryOperator::Matches => Instruction::MatchPattern("default".to_string()),
                    _ => return Err(format!("Unsupported binary operator: {:?}", binary.operator)),
                };
                
                self.emit(instruction);
                Ok(())
            }
            
            Expression::Unary(unary) => {
                self.compile_expression(*unary.operand)?;
                
                let instruction = match unary.operator {
                    UnaryOperator::Minus => {
                        // Load -1 and multiply
                        let neg_one = self.add_constant(TurbulanceValue::Integer(-1));
                        self.emit(Instruction::LoadConstant(neg_one));
                        Instruction::Multiply
                    }
                    UnaryOperator::Not => Instruction::Not,
                };
                
                self.emit(instruction);
                Ok(())
            }
            
            Expression::Call(call) => {
                // Compile arguments in reverse order (stack-based)
                for arg in call.arguments.iter().rev() {
                    self.compile_expression(arg.clone())?;
                }
                
                if let Expression::Identifier(func_name) = *call.callee {
                    self.emit(Instruction::Call(func_name, call.arguments.len()));
                } else {
                    return Err("Complex function calls not yet supported".to_string());
                }
                
                Ok(())
            }
            
            Expression::Array(array) => {
                // Create array by compiling all elements
                for element in array.elements {
                    self.compile_expression(element)?;
                }
                
                // TODO: Emit array creation instruction
                Ok(())
            }
            
            Expression::Dictionary(dict) => {
                // Create dictionary by compiling all key-value pairs
                for (key, value) in dict.pairs {
                    self.compile_expression(key)?;
                    self.compile_expression(value)?;
                }
                
                // TODO: Emit dictionary creation instruction
                Ok(())
            }
            
            Expression::Index(index) => {
                self.compile_expression(*index.object)?;
                self.compile_expression(*index.index)?;
                
                // TODO: Emit index access instruction
                Ok(())
            }
            
            Expression::Member(member) => {
                self.compile_expression(*member.object)?;
                
                // Load member name as constant
                let member_const = self.add_constant(TurbulanceValue::String(member.member));
                self.emit(Instruction::LoadConstant(member_const));
                
                // TODO: Emit member access instruction
                Ok(())
            }
            
            Expression::Pattern(pattern) => {
                let pattern_const = self.add_constant(TurbulanceValue::String(pattern.pattern));
                self.emit(Instruction::LoadConstant(pattern_const));
                self.emit(Instruction::MatchPattern(format!("{:?}", pattern.pattern_type)));
                Ok(())
            }
            
            Expression::Lambda(_) => {
                // Lambda compilation would be more complex
                Err("Lambda expressions not yet implemented".to_string())
            }
        }
    }

    fn compile_function(&mut self, function: Function) -> Result<(), String> {
        let func_start = self.instructions.len();
        self.symbol_table.insert(function.name.clone(), func_start);
        self.current_function = Some(function.name.clone());
        
        // Compile function body
        for statement in function.body {
            self.compile_statement(statement)?;
        }
        
        // Ensure function returns
        if !matches!(self.instructions.last(), Some(Instruction::Return)) {
            // Return None if no explicit return
            let none_const = self.add_constant(TurbulanceValue::String("None".to_string()));
            self.emit(Instruction::LoadConstant(none_const));
            self.emit(Instruction::Return);
        }
        
        self.current_function = None;
        Ok(())
    }

    fn compile_proposition(&mut self, proposition: Proposition) -> Result<(), String> {
        // Create the proposition
        self.emit(Instruction::CreateProposition(proposition.name.clone()));
        
        // Compile all motions
        for motion in proposition.motions {
            self.emit(Instruction::CreateMotion(motion.name, motion.description));
            
            // Compile support conditions
            for condition in motion.support_conditions {
                self.compile_expression(condition.condition)?;
                
                match condition.action {
                    SupportAction::Support => {
                        self.emit(Instruction::SupportMotion(motion.name.clone(), 1.0));
                    }
                    SupportAction::Contradict => {
                        self.emit(Instruction::ContradictMotion(motion.name.clone(), 1.0));
                    }
                    SupportAction::WeightedSupport(weight) => {
                        self.emit(Instruction::SupportMotion(motion.name.clone(), weight));
                    }
                    SupportAction::WeightedContradict(weight) => {
                        self.emit(Instruction::ContradictMotion(motion.name.clone(), weight));
                    }
                }
            }
        }
        
        // Evaluate the proposition
        self.emit(Instruction::EvaluateProposition(proposition.name));
        Ok(())
    }

    fn compile_evidence_collector(&mut self, evidence: EvidenceCollector) -> Result<(), String> {
        // Set up evidence collection
        self.emit(Instruction::CollectEvidence(evidence.name));
        Ok(())
    }

    fn compile_metacognitive_monitor(&mut self, monitor: MetacognitiveMonitor) -> Result<(), String> {
        // Set up metacognitive monitoring
        for target in monitor.tracking_targets {
            self.emit(Instruction::TrackReasoning(target));
        }
        
        for method in monitor.evaluation_methods {
            match method.as_str() {
                "confidence" => self.emit(Instruction::EvaluateConfidence),
                "bias_detection" => self.emit(Instruction::DetectBias("general".to_string())),
                _ => {} // Unknown evaluation method
            }
        }
        
        Ok(())
    }

    fn compile_goal_system(&mut self, goal_system: GoalSystem) -> Result<(), String> {
        // Create all goals
        for goal in goal_system.goals {
            self.emit(Instruction::CreateGoal(goal.id.clone(), goal.success_threshold));
            self.emit(Instruction::UpdateGoalProgress(goal.id, goal.current_progress));
        }
        
        Ok(())
    }

    fn compile_if_statement(&mut self, if_stmt: IfStatement) -> Result<(), String> {
        // Compile condition
        self.compile_expression(if_stmt.condition)?;
        
        // Jump if false
        let jump_to_else = self.emit_jump_placeholder(Instruction::JumpIfFalse(0));
        
        // Compile then branch
        for statement in if_stmt.then_branch {
            self.compile_statement(statement)?;
        }
        
        let jump_to_end = if if_stmt.else_branch.is_some() {
            Some(self.emit_jump_placeholder(Instruction::Jump(0)))
        } else {
            None
        };
        
        // Patch jump to else
        self.patch_jump(jump_to_else);
        
        // Compile else branch if present
        if let Some(else_branch) = if_stmt.else_branch {
            for statement in else_branch {
                self.compile_statement(statement)?;
            }
        }
        
        // Patch jump to end
        if let Some(jump_idx) = jump_to_end {
            self.patch_jump(jump_idx);
        }
        
        Ok(())
    }

    fn compile_while_statement(&mut self, while_stmt: WhileStatement) -> Result<(), String> {
        let loop_start = self.instructions.len();
        
        // Compile condition
        self.compile_expression(while_stmt.condition)?;
        
        // Jump if false (exit loop)
        let exit_jump = self.emit_jump_placeholder(Instruction::JumpIfFalse(0));
        
        // Compile loop body
        for statement in while_stmt.body {
            self.compile_statement(statement)?;
        }
        
        // Jump back to condition
        self.emit(Instruction::Jump(loop_start));
        
        // Patch exit jump
        self.patch_jump(exit_jump);
        
        Ok(())
    }

    fn compile_for_statement(&mut self, for_stmt: ForStatement) -> Result<(), String> {
        // This is a simplified for loop compilation
        // In practice, we'd need iterator protocol
        
        self.compile_expression(for_stmt.iterable)?;
        
        // TODO: Implement proper iteration logic
        
        for statement in for_stmt.body {
            self.compile_statement(statement)?;
        }
        
        Ok(())
    }

    fn compile_within_statement(&mut self, within_stmt: WithinStatement) -> Result<(), String> {
        self.compile_expression(within_stmt.scope)?;
        
        // Enter scope
        self.emit(Instruction::WithinScope("scope".to_string()));
        
        // Compile body
        for statement in within_stmt.body {
            self.compile_statement(statement)?;
        }
        
        // Exit scope
        self.emit(Instruction::ExitScope);
        
        Ok(())
    }

    // Helper methods
    fn emit(&mut self, instruction: Instruction) -> usize {
        let index = self.instructions.len();
        self.instructions.push(instruction);
        index
    }

    fn emit_jump_placeholder(&mut self, instruction: Instruction) -> usize {
        self.emit(instruction)
    }

    fn patch_jump(&mut self, jump_index: usize) {
        let target = self.instructions.len();
        match &mut self.instructions[jump_index] {
            Instruction::Jump(ref mut addr) => *addr = target,
            Instruction::JumpIfFalse(ref mut addr) => *addr = target,
            Instruction::JumpIfTrue(ref mut addr) => *addr = target,
            _ => panic!("Attempted to patch non-jump instruction"),
        }
    }

    fn add_constant(&mut self, value: TurbulanceValue) -> usize {
        let index = self.constants.len();
        self.constants.push(value);
        index
    }

    fn optimize(&mut self) -> Result<(), String> {
        // Basic peephole optimizations
        let mut optimized = Vec::new();
        let mut i = 0;
        
        while i < self.instructions.len() {
            match (&self.instructions.get(i), &self.instructions.get(i + 1)) {
                // Remove unnecessary pop after print
                (Some(Instruction::Print), Some(Instruction::Pop)) => {
                    optimized.push(Instruction::Print);
                    i += 2; // Skip both instructions
                }
                
                // Remove double negation
                (Some(Instruction::Not), Some(Instruction::Not)) => {
                    i += 2; // Skip both, effectively removing them
                }
                
                // Default case - keep instruction
                _ => {
                    if let Some(instruction) = self.instructions.get(i) {
                        optimized.push(instruction.clone());
                    }
                    i += 1;
                }
            }
        }
        
        self.instructions = optimized;
        Ok(())
    }
}
