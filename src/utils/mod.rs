//! Utility functions for the Autobahn biological metabolism computer

/// Basic utility functions
pub mod helpers;

/// Mathematical utilities
pub mod math;

/// Text processing utilities  
pub mod text;

use crate::error::AutobahnResult;

/// Calculate text complexity score
pub fn calculate_text_complexity(text: &str) -> f64 {
    let word_count = text.split_whitespace().count() as f64;
    let sentence_count = text.matches(&['.', '!', '?'][..]).count() as f64;
    
    if sentence_count == 0.0 {
        return word_count / 100.0; // Simple fallback
    }
    
    let avg_sentence_length = word_count / sentence_count;
    let complexity_score = (avg_sentence_length / 20.0) + (word_count / 1000.0);
    
    complexity_score.max(0.1).min(10.0)
}

/// Validate input content
pub fn validate_content(content: &str) -> AutobahnResult<()> {
    if content.is_empty() {
        return Err(crate::error::AutobahnError::InvalidInputError {
            expected: "Non-empty content".to_string(),
            actual: "Empty content".to_string(),
        });
    }
    
    if content.len() > 1_000_000 {
        return Err(crate::error::AutobahnError::InvalidInputError {
            expected: "Content under 1MB".to_string(),
            actual: format!("Content size: {} bytes", content.len()),
        });
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_complexity_calculation() {
        let simple = "Hello world.";
        let complex = "However, the implications of this discovery are quite significant. Therefore, we must consider the ramifications carefully.";
        
        let simple_score = calculate_text_complexity(simple);
        let complex_score = calculate_text_complexity(complex);
        
        assert!(complex_score > simple_score);
    }

    #[test]
    fn test_content_validation() {
        assert!(validate_content("Valid content").is_ok());
        assert!(validate_content("").is_err());
    }
} 