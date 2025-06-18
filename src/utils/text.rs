//! Text processing utilities for biological metabolism

use std::collections::HashMap;

/// Extract key terms from text
pub fn extract_key_terms(text: &str) -> Vec<String> {
    let words: Vec<String> = text
        .split_whitespace()
        .map(|w| w.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() > 3)
        .collect();
    
    let mut frequency = HashMap::new();
    for word in &words {
        *frequency.entry(word.clone()).or_insert(0) += 1;
    }
    
    let mut sorted_words: Vec<_> = frequency.into_iter().collect();
    sorted_words.sort_by(|a, b| b.1.cmp(&a.1));
    
    sorted_words.into_iter().take(10).map(|(word, _)| word).collect()
}

/// Calculate semantic density
pub fn calculate_semantic_density(text: &str) -> f64 {
    let words = text.split_whitespace().count();
    let unique_words = text
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect::<std::collections::HashSet<_>>()
        .len();
    
    if words == 0 {
        return 0.0;
    }
    
    unique_words as f64 / words as f64
}

/// Clean text for processing
pub fn clean_text(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?;:".contains(*c))
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract sentences from text
pub fn extract_sentences(text: &str) -> Vec<String> {
    text.split(&['.', '!', '?'][..])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_term_extraction() {
        let text = "The biological processor metabolizes information through cellular pathways";
        let terms = extract_key_terms(text);
        assert!(!terms.is_empty());
        assert!(terms.contains(&"biological".to_string()));
    }

    #[test]
    fn test_semantic_density() {
        let text = "the the the the";
        let density = calculate_semantic_density(text);
        assert_eq!(density, 0.25); // 1 unique word out of 4 total
    }

    #[test]
    fn test_text_cleaning() {
        let dirty_text = "Hello!!! @#$%^&* World???   ";
        let clean = clean_text(dirty_text);
        assert_eq!(clean, "Hello!!! World???");
    }

    #[test]
    fn test_sentence_extraction() {
        let text = "First sentence. Second sentence! Third sentence?";
        let sentences = extract_sentences(text);
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First sentence");
    }
} 