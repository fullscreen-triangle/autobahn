/// Turbulance Lexer - Tokenizes Turbulance source code
/// 
/// This module handles lexical analysis of Turbulance code, breaking it down
/// into tokens that can be parsed into an Abstract Syntax Tree.

use std::fmt;

/// Turbulance token types
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Literals
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Identifier(String),

    // Keywords
    Item,           // item
    Funxn,          // funxn
    Given,          // given
    Otherwise,      // otherwise
    Within,         // within
    Considering,    // considering
    For,            // for
    Each,           // each
    In,             // in
    While,          // while
    Return,         // return
    Try,            // try
    Catch,          // catch
    Finally,        // finally
    Import,         // import
    From,           // from
    As,             // as
    Parallel,       // parallel
    Async,          // async
    Await,          // await
    
    // Scientific constructs
    Proposition,    // proposition
    Motion,         // motion
    Evidence,       // evidence
    Metacognitive,  // metacognitive
    Pattern,        // pattern
    Goal,           // goal (from goal system)
    Track,          // track
    Evaluate,       // evaluate
    Support,        // support
    Contradict,     // contradict
    Matches,        // matches
    Contains,       // contains
    WithWeight,     // with_weight
    
    // Operators
    Plus,           // +
    Minus,          // -
    Multiply,       // *
    Divide,         // /
    Modulo,         // %
    Power,          // **
    Assign,         // =
    PlusAssign,     // +=
    MinusAssign,    // -=
    MultiplyAssign, // *=
    DivideAssign,   // /=
    Equal,          // ==
    NotEqual,       // !=
    Less,           // <
    Greater,        // >
    LessEqual,      // <=
    GreaterEqual,   // >=
    And,            // and
    Or,             // or
    Not,            // not
    BitwiseAnd,     // &
    BitwiseOr,      // |
    Arrow,          // ->
    
    // Delimiters
    LeftParen,      // (
    RightParen,     // )
    LeftBracket,    // [
    RightBracket,   // ]
    LeftBrace,      // {
    RightBrace,     // }
    Comma,          // ,
    Dot,            // .
    Colon,          // :
    Semicolon,      // ;
    Question,       // ?
    
    // Special
    Newline,
    Indent,
    Dedent,
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Integer(n) => write!(f, "{}", n),
            Token::Float(n) => write!(f, "{}", n),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Boolean(b) => write!(f, "{}", b),
            Token::Identifier(s) => write!(f, "{}", s),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Position information for tokens
#[derive(Debug, Clone, PartialEq)]
pub struct TokenPosition {
    pub line: usize,
    pub column: usize,
    pub offset: usize,
}

/// Token with position information
#[derive(Debug, Clone)]
pub struct TokenWithPosition {
    pub token: Token,
    pub position: TokenPosition,
}

/// Turbulance lexer
#[derive(Debug)]
pub struct TurbulanceLexer {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
}

impl TurbulanceLexer {
    /// Create a new lexer
    pub fn new() -> Self {
        Self {
            input: Vec::new(),
            position: 0,
            line: 1,
            column: 1,
        }
    }

    /// Tokenize input source code
    pub fn tokenize(&mut self, input: &str) -> Result<Vec<TokenWithPosition>, String> {
        self.input = input.chars().collect();
        self.position = 0;
        self.line = 1;
        self.column = 1;

        let mut tokens = Vec::new();
        
        while !self.is_at_end() {
            let token_pos = TokenPosition {
                line: self.line,
                column: self.column,
                offset: self.position,
            };

            match self.scan_token()? {
                Some(token) => {
                    tokens.push(TokenWithPosition { token, position: token_pos });
                }
                None => {} // Skip whitespace and comments
            }
        }

        tokens.push(TokenWithPosition {
            token: Token::Eof,
            position: TokenPosition {
                line: self.line,
                column: self.column,
                offset: self.position,
            },
        });

        Ok(tokens)
    }

    /// Scan a single token
    fn scan_token(&mut self) -> Result<Option<Token>, String> {
        let c = self.advance();

        match c {
            ' ' | '\r' | '\t' => Ok(None), // Skip whitespace
            '\n' => {
                self.line += 1;
                self.column = 1;
                Ok(Some(Token::Newline))
            }
            '#' => {
                // Skip comments
                while !self.is_at_end() && self.current_char() != '\n' {
                    self.advance();
                }
                Ok(None)
            }
            
            // String literals
            '"' => self.scan_string(),
            '\'' => self.scan_string_single(),
            
            // Numbers
            c if c.is_ascii_digit() => self.scan_number(),
            
            // Identifiers and keywords
            c if c.is_alphabetic() || c == '_' => self.scan_identifier(),
            
            // Two-character operators
            '=' => {
                if self.match_char('=') {
                    Ok(Some(Token::Equal))
                } else {
                    Ok(Some(Token::Assign))
                }
            }
            '!' => {
                if self.match_char('=') {
                    Ok(Some(Token::NotEqual))
                } else {
                    Err(format!("Unexpected character '!' at line {}", self.line))
                }
            }
            '<' => {
                if self.match_char('=') {
                    Ok(Some(Token::LessEqual))
                } else {
                    Ok(Some(Token::Less))
                }
            }
            '>' => {
                if self.match_char('=') {
                    Ok(Some(Token::GreaterEqual))
                } else {
                    Ok(Some(Token::Greater))
                }
            }
            '*' => {
                if self.match_char('*') {
                    Ok(Some(Token::Power))
                } else if self.match_char('=') {
                    Ok(Some(Token::MultiplyAssign))
                } else {
                    Ok(Some(Token::Multiply))
                }
            }
            '+' => {
                if self.match_char('=') {
                    Ok(Some(Token::PlusAssign))
                } else {
                    Ok(Some(Token::Plus))
                }
            }
            '-' => {
                if self.match_char('=') {
                    Ok(Some(Token::MinusAssign))
                } else if self.match_char('>') {
                    Ok(Some(Token::Arrow))
                } else {
                    Ok(Some(Token::Minus))
                }
            }
            '/' => {
                if self.match_char('=') {
                    Ok(Some(Token::DivideAssign))
                } else {
                    Ok(Some(Token::Divide))
                }
            }
            
            // Single-character tokens
            '(' => Ok(Some(Token::LeftParen)),
            ')' => Ok(Some(Token::RightParen)),
            '[' => Ok(Some(Token::LeftBracket)),
            ']' => Ok(Some(Token::RightBracket)),
            '{' => Ok(Some(Token::LeftBrace)),
            '}' => Ok(Some(Token::RightBrace)),
            ',' => Ok(Some(Token::Comma)),
            '.' => Ok(Some(Token::Dot)),
            ':' => Ok(Some(Token::Colon)),
            ';' => Ok(Some(Token::Semicolon)),
            '?' => Ok(Some(Token::Question)),
            '%' => Ok(Some(Token::Modulo)),
            '&' => Ok(Some(Token::BitwiseAnd)),
            '|' => Ok(Some(Token::BitwiseOr)),
            
            _ => Err(format!("Unexpected character '{}' at line {}", c, self.line)),
        }
    }

    /// Scan a string literal with double quotes
    fn scan_string(&mut self) -> Result<Option<Token>, String> {
        let mut value = String::new();

        while !self.is_at_end() && self.current_char() != '"' {
            if self.current_char() == '\\' {
                self.advance();
                if self.is_at_end() {
                    return Err(format!("Unterminated string escape at line {}", self.line));
                }
                
                match self.current_char() {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    '\'' => value.push('\''),
                    c => {
                        value.push('\\');
                        value.push(c);
                    }
                }
            } else {
                value.push(self.current_char());
            }
            self.advance();
        }

        if self.is_at_end() {
            return Err(format!("Unterminated string at line {}", self.line));
        }

        self.advance(); // Closing quote
        Ok(Some(Token::String(value)))
    }

    /// Scan a string literal with single quotes
    fn scan_string_single(&mut self) -> Result<Option<Token>, String> {
        let mut value = String::new();

        while !self.is_at_end() && self.current_char() != '\'' {
            if self.current_char() == '\\' {
                self.advance();
                if self.is_at_end() {
                    return Err(format!("Unterminated string escape at line {}", self.line));
                }
                
                match self.current_char() {
                    'n' => value.push('\n'),
                    't' => value.push('\t'),
                    'r' => value.push('\r'),
                    '\\' => value.push('\\'),
                    '"' => value.push('"'),
                    '\'' => value.push('\''),
                    c => {
                        value.push('\\');
                        value.push(c);
                    }
                }
            } else {
                value.push(self.current_char());
            }
            self.advance();
        }

        if self.is_at_end() {
            return Err(format!("Unterminated string at line {}", self.line));
        }

        self.advance(); // Closing quote
        Ok(Some(Token::String(value)))
    }

    /// Scan a number (integer or float)
    fn scan_number(&mut self) -> Result<Option<Token>, String> {
        let start = self.position - 1;
        
        while !self.is_at_end() && self.current_char().is_ascii_digit() {
            self.advance();
        }

        let mut is_float = false;
        if !self.is_at_end() && self.current_char() == '.' && 
           self.peek_next().map_or(false, |c| c.is_ascii_digit()) {
            is_float = true;
            self.advance(); // Decimal point
            
            while !self.is_at_end() && self.current_char().is_ascii_digit() {
                self.advance();
            }
        }

        let number_str: String = self.input[start..self.position].iter().collect();
        
        if is_float {
            match number_str.parse::<f64>() {
                Ok(f) => Ok(Some(Token::Float(f))),
                Err(_) => Err(format!("Invalid float '{}' at line {}", number_str, self.line)),
            }
        } else {
            match number_str.parse::<i64>() {
                Ok(i) => Ok(Some(Token::Integer(i))),
                Err(_) => Err(format!("Invalid integer '{}' at line {}", number_str, self.line)),
            }
        }
    }

    /// Scan an identifier or keyword
    fn scan_identifier(&mut self) -> Result<Option<Token>, String> {
        let start = self.position - 1;
        
        while !self.is_at_end() && (self.current_char().is_alphanumeric() || self.current_char() == '_') {
            self.advance();
        }

        let text: String = self.input[start..self.position].iter().collect();
        
        let token = match text.as_str() {
            // Keywords
            "item" => Token::Item,
            "funxn" => Token::Funxn,
            "given" => Token::Given,
            "otherwise" => Token::Otherwise,
            "within" => Token::Within,
            "considering" => Token::Considering,
            "for" => Token::For,
            "each" => Token::Each,
            "in" => Token::In,
            "while" => Token::While,
            "return" => Token::Return,
            "try" => Token::Try,
            "catch" => Token::Catch,
            "finally" => Token::Finally,
            "import" => Token::Import,
            "from" => Token::From,
            "as" => Token::As,
            "parallel" => Token::Parallel,
            "async" => Token::Async,
            "await" => Token::Await,
            
            // Scientific constructs
            "proposition" => Token::Proposition,
            "motion" => Token::Motion,
            "evidence" => Token::Evidence,
            "metacognitive" => Token::Metacognitive,
            "pattern" => Token::Pattern,
            "goal" => Token::Goal,
            "track" => Token::Track,
            "evaluate" => Token::Evaluate,
            "support" => Token::Support,
            "contradict" => Token::Contradict,
            "matches" => Token::Matches,
            "contains" => Token::Contains,
            "with_weight" => Token::WithWeight,
            
            // Logical operators
            "and" => Token::And,
            "or" => Token::Or,
            "not" => Token::Not,
            
            // Boolean literals
            "true" => Token::Boolean(true),
            "false" => Token::Boolean(false),
            
            // Identifier
            _ => Token::Identifier(text),
        };

        Ok(Some(token))
    }

    /// Helper methods
    fn is_at_end(&self) -> bool {
        self.position >= self.input.len()
    }

    fn current_char(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.input[self.position]
        }
    }

    fn advance(&mut self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            let c = self.input[self.position];
            self.position += 1;
            self.column += 1;
            c
        }
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() || self.current_char() != expected {
            false
        } else {
            self.advance();
            true
        }
    }

    fn peek_next(&self) -> Option<char> {
        if self.position + 1 >= self.input.len() {
            None
        } else {
            Some(self.input[self.position + 1])
        }
    }
} 