//! Mathematical utilities for biological processing

/// Calculate confidence interval
pub fn confidence_interval(mean: f64, std_dev: f64, confidence_level: f64) -> (f64, f64) {
    let z_score = match confidence_level {
        0.95 => 1.96,
        0.99 => 2.576,
        0.90 => 1.645,
        _ => 1.96, // Default to 95%
    };
    
    let margin = z_score * std_dev;
    (mean - margin, mean + margin)
}

/// Calculate exponential decay
pub fn exponential_decay(initial_value: f64, decay_rate: f64, time: f64) -> f64 {
    initial_value * (-decay_rate * time).exp()
}

/// Sigmoid function for probabilistic calculations
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Normalize vector to sum to 1.0
pub fn normalize_probabilities(values: &mut [f64]) {
    let sum: f64 = values.iter().sum();
    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_interval() {
        let (lower, upper) = confidence_interval(100.0, 10.0, 0.95);
        assert!(lower < 100.0);
        assert!(upper > 100.0);
    }

    #[test]
    fn test_exponential_decay() {
        let result = exponential_decay(100.0, 0.1, 1.0);
        assert!(result < 100.0);
        assert!(result > 0.0);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 0.001);
        assert!(sigmoid(10.0) > 0.9);
        assert!(sigmoid(-10.0) < 0.1);
    }
} 