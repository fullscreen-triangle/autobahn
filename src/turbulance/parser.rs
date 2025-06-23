/// Turbulance Parser - Converts tokens into Abstract Syntax Tree
/// 
/// This parser handles the conversion of tokenized Turbulance code into
/// a structured AST that can be compiled and executed.

use crate::turbulance::lexer::{Token, TokenWithPosition};
use crate::turbulance::ast::*;

pub struct TurbulanceParser {
    tokens: Vec<TokenWithPosition>,
    current: usize,
}

impl TurbulanceParser {
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            current: 0,
        }
    }

    pub fn parse(&mut self, tokens: Vec<TokenWithPosition>) -> Result<Program, String> {
        self.tokens = tokens;
        self.current = 0;
        
        let mut program = Program::new();
        
        while !self.is_at_end() {
            if self.match_token(&Token::Newline) {
                continue; // Skip newlines
            }
            
            let statement = self.parse_statement()?;
            program.add_statement(statement);
        }
        
        Ok(program)
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        match &self.peek().token {
            Token::Item => self.parse_variable_declaration(),
            Token::Funxn => self.parse_function_declaration(),
            Token::Proposition => self.parse_proposition_declaration(),
            Token::Evidence => self.parse_evidence_declaration(),
            Token::Metacognitive => self.parse_metacognitive_declaration(),
            Token::Goal => self.parse_goal_declaration(),
            Token::Given => self.parse_if_statement(),
            Token::While => self.parse_while_statement(),
            Token::For => self.parse_for_statement(),
            Token::Within => self.parse_within_statement(),
            Token::Return => self.parse_return_statement(),
            Token::Import => self.parse_import_statement(),
            _ => self.parse_expression_statement(),
        }
    }

    fn parse_variable_declaration(&mut self) -> Result<Statement, String> {
        let position = self.advance().position.clone(); // consume 'item'
        
        let name = if let Token::Identifier(name) = &self.advance().token {
            name.clone()
        } else {
            return Err("Expected identifier after 'item'".to_string());
        };

        let type_annotation = if self.match_token(&Token::Colon) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };

        let initializer = if self.match_token(&Token::Assign) {
            Some(self.parse_expression()?)
        } else {
            None
        };

        Ok(Statement::VariableDeclaration(VariableDeclaration {
            name,
            type_annotation,
            initializer,
            position,
        }))
    }

    fn parse_function_declaration(&mut self) -> Result<Statement, String> {
        let position = self.advance().position.clone(); // consume 'funxn'
        
        let name = if let Token::Identifier(name) = &self.advance().token {
            name.clone()
        } else {
            return Err("Expected function name".to_string());
        };

        self.consume(&Token::LeftParen, "Expected '(' after function name")?;
        
        let mut parameters = Vec::new();
        if !self.check(&Token::RightParen) {
            loop {
                let param_name = if let Token::Identifier(name) = &self.advance().token {
                    name.clone()
                } else {
                    return Err("Expected parameter name".to_string());
                };

                let type_annotation = if self.match_token(&Token::Colon) {
                    Some(self.parse_type_annotation()?)
                } else {
                    None
                };

                let default_value = if self.match_token(&Token::Assign) {
                    Some(self.parse_expression()?)
                } else {
                    None
                };

                parameters.push(Parameter {
                    name: param_name,
                    type_annotation,
                    default_value,
                });

                if !self.match_token(&Token::Comma) {
                    break;
                }
            }
        }

        self.consume(&Token::RightParen, "Expected ')' after parameters")?;
        
        let return_type = if self.match_token(&Token::Arrow) {
            Some(self.parse_type_annotation()?)
        } else {
            None
        };

        self.consume(&Token::Colon, "Expected ':' after function signature")?;
        
        let body = self.parse_block()?;

        Ok(Statement::FunctionDeclaration(Function {
            name,
            parameters,
            return_type,
            body,
            is_async: false,
            position,
        }))
    }

    fn parse_proposition_declaration(&mut self) -> Result<Statement, String> {
        let position = self.advance().position.clone(); // consume 'proposition'
        
        let name = if let Token::Identifier(name) = &self.advance().token {
            name.clone()
        } else {
            return Err("Expected proposition name".to_string());
        };

        self.consume(&Token::Colon, "Expected ':' after proposition name")?;
        
        let mut proposition = Proposition::new(name, position);
        
        // Parse proposition body
        while !self.is_at_end() && !self.check_statement_start() {
            if self.match_token(&Token::Motion) {
                let motion = self.parse_motion()?;
                proposition.add_motion(motion);
            } else if self.match_token(&Token::Newline) {
                continue;
            } else {
                break;
            }
        }

        Ok(Statement::PropositionDeclaration(proposition))
    }

    fn parse_motion(&mut self) -> Result<Motion, String> {
        let name = if let Token::Identifier(name) = &self.advance().token {
            name.clone()
        } else {
            return Err("Expected motion name".to_string());
        };

        self.consume(&Token::LeftParen, "Expected '(' after motion name")?;
        
        let description = if let Token::String(desc) = &self.advance().token {
            desc.clone()
        } else {
            return Err("Expected motion description string".to_string());
        };

        self.consume(&Token::RightParen, "Expected ')' after motion description")?;
        
        Ok(Motion::new(name, description, self.previous().position.clone()))
    }

    fn parse_evidence_declaration(&mut self) -> Result<Statement, String> {
        let position = self.advance().position.clone(); // consume 'evidence'
        
        let name = if let Token::Identifier(name) = &self.advance().token {
            name.clone()
        } else {
            return Err("Expected evidence collector name".to_string());
        };

        self.consume(&Token::Colon, "Expected ':' after evidence name")?;
        
        // Simplified evidence parsing
        Ok(Statement::EvidenceDeclaration(EvidenceCollector {
            id: name.clone(),
            name,
            sources: Vec::new(),
            validation_rules: Vec::new(),
            collection_frequency: None,
            position,
        }))
    }

    fn parse_metacognitive_declaration(&mut self) -> Result<Statement, String> {
        let position = self.advance().position.clone(); // consume 'metacognitive'
        
        let name = if let Token::Identifier(name) = &self.advance().token {
            name.clone()
        } else {
            return Err("Expected metacognitive monitor name".to_string());
        };

        self.consume(&Token::Colon, "Expected ':' after metacognitive name")?;
        
        Ok(Statement::MetacognitiveDeclaration(MetacognitiveMonitor {
            name,
            tracking_targets: Vec::new(),
            evaluation_methods: Vec::new(),
            adaptation_rules: Vec::new(),
            position,
        }))
    }

    fn parse_goal_declaration(&mut self) -> Result<Statement, String> {
        let position = self.advance().position.clone(); // consume 'goal'
        
        let name = if let Token::Identifier(name) = &self.advance().token {
            name.clone()
        } else {
            return Err("Expected goal system name".to_string());
        };

        self.consume(&Token::Colon, "Expected ':' after goal name")?;
        
        Ok(Statement::GoalDeclaration(GoalSystem {
            name,
            goals: Vec::new(),
            dependencies: Vec::new(),
            position,
        }))
    }

    fn parse_if_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'given'
        
        let condition = self.parse_expression()?;
        self.consume(&Token::Colon, "Expected ':' after if condition")?;
        
        let then_branch = self.parse_block()?;
        
        let else_branch = if self.match_token(&Token::Otherwise) {
            self.consume(&Token::Colon, "Expected ':' after 'otherwise'")?;
            Some(self.parse_block()?)
        } else {
            None
        };

        Ok(Statement::IfStatement(IfStatement {
            condition,
            then_branch,
            else_branch,
        }))
    }

    fn parse_while_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'while'
        
        let condition = self.parse_expression()?;
        self.consume(&Token::Colon, "Expected ':' after while condition")?;
        
        let body = self.parse_block()?;

        Ok(Statement::WhileStatement(WhileStatement {
            condition,
            body,
        }))
    }

    fn parse_for_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'for'
        
        if self.match_token(&Token::Each) {
            let variable = if let Token::Identifier(name) = &self.advance().token {
                name.clone()
            } else {
                return Err("Expected variable name in for loop".to_string());
            };

            self.consume(&Token::In, "Expected 'in' after for variable")?;
            
            let iterable = self.parse_expression()?;
            self.consume(&Token::Colon, "Expected ':' after for expression")?;
            
            let body = self.parse_block()?;

            Ok(Statement::ForStatement(ForStatement {
                variable,
                iterable,
                body,
            }))
        } else {
            Err("Expected 'each' after 'for'".to_string())
        }
    }

    fn parse_within_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'within'
        
        let scope = self.parse_expression()?;
        self.consume(&Token::Colon, "Expected ':' after within scope")?;
        
        let body = self.parse_block()?;

        Ok(Statement::WithinStatement(WithinStatement {
            scope,
            conditions: Vec::new(),
            body,
        }))
    }

    fn parse_return_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'return'
        
        let value = if self.check(&Token::Newline) || self.is_at_end() {
            None
        } else {
            Some(self.parse_expression()?)
        };

        Ok(Statement::ReturnStatement(ReturnStatement { value }))
    }

    fn parse_import_statement(&mut self) -> Result<Statement, String> {
        self.advance(); // consume 'import'
        
        let module = if let Token::Identifier(name) = &self.advance().token {
            name.clone()
        } else {
            return Err("Expected module name after 'import'".to_string());
        };

        Ok(Statement::ImportStatement(ImportStatement {
            module,
            items: Vec::new(),
            alias: None,
        }))
    }

    fn parse_expression_statement(&mut self) -> Result<Statement, String> {
        let expr = self.parse_expression()?;
        Ok(Statement::ExpressionStatement(expr))
    }

    fn parse_expression(&mut self) -> Result<Expression, String> {
        self.parse_logical_or()
    }

    fn parse_logical_or(&mut self) -> Result<Expression, String> {
        let mut expr = self.parse_logical_and()?;

        while self.match_token(&Token::Or) {
            let right = self.parse_logical_and()?;
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: BinaryOperator::Or,
                right: Box::new(right),
            });
        }

        Ok(expr)
    }

    fn parse_logical_and(&mut self) -> Result<Expression, String> {
        let mut expr = self.parse_equality()?;

        while self.match_token(&Token::And) {
            let right = self.parse_equality()?;
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: BinaryOperator::And,
                right: Box::new(right),
            });
        }

        Ok(expr)
    }

    fn parse_equality(&mut self) -> Result<Expression, String> {
        let mut expr = self.parse_comparison()?;

        while let Some(op) = self.match_equality_operator() {
            let right = self.parse_comparison()?;
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            });
        }

        Ok(expr)
    }

    fn parse_comparison(&mut self) -> Result<Expression, String> {
        let mut expr = self.parse_term()?;

        while let Some(op) = self.match_comparison_operator() {
            let right = self.parse_term()?;
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            });
        }

        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expression, String> {
        let mut expr = self.parse_factor()?;

        while let Some(op) = self.match_term_operator() {
            let right = self.parse_factor()?;
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            });
        }

        Ok(expr)
    }

    fn parse_factor(&mut self) -> Result<Expression, String> {
        let mut expr = self.parse_unary()?;

        while let Some(op) = self.match_factor_operator() {
            let right = self.parse_unary()?;
            expr = Expression::Binary(BinaryExpression {
                left: Box::new(expr),
                operator: op,
                right: Box::new(right),
            });
        }

        Ok(expr)
    }

    fn parse_unary(&mut self) -> Result<Expression, String> {
        if let Some(op) = self.match_unary_operator() {
            let expr = self.parse_unary()?;
            Ok(Expression::Unary(UnaryExpression {
                operator: op,
                operand: Box::new(expr),
            }))
        } else {
            self.parse_call()
        }
    }

    fn parse_call(&mut self) -> Result<Expression, String> {
        let mut expr = self.parse_primary()?;

        loop {
            if self.match_token(&Token::LeftParen) {
                let mut arguments = Vec::new();
                if !self.check(&Token::RightParen) {
                    loop {
                        arguments.push(self.parse_expression()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&Token::RightParen, "Expected ')' after arguments")?;
                
                expr = Expression::Call(CallExpression {
                    callee: Box::new(expr),
                    arguments,
                });
            } else if self.match_token(&Token::Dot) {
                let member = if let Token::Identifier(name) = &self.advance().token {
                    name.clone()
                } else {
                    return Err("Expected property name after '.'".to_string());
                };
                
                expr = Expression::Member(MemberExpression {
                    object: Box::new(expr),
                    member,
                });
            } else if self.match_token(&Token::LeftBracket) {
                let index = self.parse_expression()?;
                self.consume(&Token::RightBracket, "Expected ']' after index")?;
                
                expr = Expression::Index(IndexExpression {
                    object: Box::new(expr),
                    index: Box::new(index),
                });
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expression, String> {
        match &self.peek().token {
            Token::Integer(n) => {
                let value = *n;
                self.advance();
                Ok(Expression::Literal(Literal::Integer(value)))
            }
            Token::Float(f) => {
                let value = *f;
                self.advance();
                Ok(Expression::Literal(Literal::Float(value)))
            }
            Token::String(s) => {
                let value = s.clone();
                self.advance();
                Ok(Expression::Literal(Literal::String(value)))
            }
            Token::Boolean(b) => {
                let value = *b;
                self.advance();
                Ok(Expression::Literal(Literal::Boolean(value)))
            }
            Token::Identifier(name) => {
                let name = name.clone();
                self.advance();
                Ok(Expression::Identifier(name))
            }
            Token::LeftParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.consume(&Token::RightParen, "Expected ')' after expression")?;
                Ok(expr)
            }
            Token::LeftBracket => {
                self.advance();
                let mut elements = Vec::new();
                if !self.check(&Token::RightBracket) {
                    loop {
                        elements.push(self.parse_expression()?);
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&Token::RightBracket, "Expected ']' after array elements")?;
                Ok(Expression::Array(ArrayExpression { elements }))
            }
            Token::LeftBrace => {
                self.advance();
                let mut pairs = Vec::new();
                if !self.check(&Token::RightBrace) {
                    loop {
                        let key = self.parse_expression()?;
                        self.consume(&Token::Colon, "Expected ':' after dictionary key")?;
                        let value = self.parse_expression()?;
                        pairs.push((key, value));
                        if !self.match_token(&Token::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&Token::RightBrace, "Expected '}' after dictionary")?;
                Ok(Expression::Dictionary(DictionaryExpression { pairs }))
            }
            _ => Err(format!("Unexpected token: {:?}", self.peek().token)),
        }
    }

    fn parse_type_annotation(&mut self) -> Result<TypeAnnotation, String> {
        if let Token::Identifier(name) = &self.advance().token {
            Ok(TypeAnnotation::Simple(name.clone()))
        } else {
            Err("Expected type annotation".to_string())
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Statement>, String> {
        let mut statements = Vec::new();
        
        while !self.is_at_end() && !self.check_statement_start() {
            if self.match_token(&Token::Newline) {
                continue;
            }
            statements.push(self.parse_statement()?);
        }
        
        Ok(statements)
    }

    // Helper methods
    fn match_token(&mut self, token: &Token) -> bool {
        if self.check(token) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check(&self, token: &Token) -> bool {
        if self.is_at_end() {
            false
        } else {
            std::mem::discriminant(&self.peek().token) == std::mem::discriminant(token)
        }
    }

    fn advance(&mut self) -> &TokenWithPosition {
        if !self.is_at_end() {
            self.current += 1;
        }
        self.previous()
    }

    fn is_at_end(&self) -> bool {
        matches!(self.peek().token, Token::Eof)
    }

    fn peek(&self) -> &TokenWithPosition {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &TokenWithPosition {
        &self.tokens[self.current - 1]
    }

    fn consume(&mut self, token: &Token, message: &str) -> Result<&TokenWithPosition, String> {
        if self.check(token) {
            Ok(self.advance())
        } else {
            Err(format!("{} at line {}", message, self.peek().position.line))
        }
    }

    fn check_statement_start(&self) -> bool {
        matches!(self.peek().token, 
            Token::Item | Token::Funxn | Token::Proposition | Token::Evidence |
            Token::Metacognitive | Token::Goal | Token::Given | Token::While |
            Token::For | Token::Within | Token::Return | Token::Import | Token::Eof
        )
    }

    fn match_equality_operator(&mut self) -> Option<BinaryOperator> {
        match &self.peek().token {
            Token::Equal => { self.advance(); Some(BinaryOperator::Equal) }
            Token::NotEqual => { self.advance(); Some(BinaryOperator::NotEqual) }
            _ => None,
        }
    }

    fn match_comparison_operator(&mut self) -> Option<BinaryOperator> {
        match &self.peek().token {
            Token::Greater => { self.advance(); Some(BinaryOperator::Greater) }
            Token::GreaterEqual => { self.advance(); Some(BinaryOperator::GreaterEqual) }
            Token::Less => { self.advance(); Some(BinaryOperator::Less) }
            Token::LessEqual => { self.advance(); Some(BinaryOperator::LessEqual) }
            _ => None,
        }
    }

    fn match_term_operator(&mut self) -> Option<BinaryOperator> {
        match &self.peek().token {
            Token::Minus => { self.advance(); Some(BinaryOperator::Subtract) }
            Token::Plus => { self.advance(); Some(BinaryOperator::Add) }
            _ => None,
        }
    }

    fn match_factor_operator(&mut self) -> Option<BinaryOperator> {
        match &self.peek().token {
            Token::Divide => { self.advance(); Some(BinaryOperator::Divide) }
            Token::Multiply => { self.advance(); Some(BinaryOperator::Multiply) }
            Token::Modulo => { self.advance(); Some(BinaryOperator::Modulo) }
            Token::Power => { self.advance(); Some(BinaryOperator::Power) }
            _ => None,
        }
    }

    fn match_unary_operator(&mut self) -> Option<UnaryOperator> {
        match &self.peek().token {
            Token::Not => { self.advance(); Some(UnaryOperator::Not) }
            Token::Minus => { self.advance(); Some(UnaryOperator::Minus) }
            _ => None,
        }
    }
} 