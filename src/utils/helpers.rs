//! Helper functions for Autobahn

/// Generate a random seed for probabilistic operations
pub fn generate_seed() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .hash(&mut hasher);
    
    hasher.finish()
}

/// Format ATP amount for display
pub fn format_atp(amount: f64) -> String {
    if amount >= 1000.0 {
        format!("{:.1}k ATP", amount / 1000.0)
    } else {
        format!("{:.1} ATP", amount)
    }
}

/// Calculate processing efficiency
pub fn calculate_efficiency(atp_consumed: f64, atp_produced: f64) -> f64 {
    if atp_consumed == 0.0 {
        return 0.0;
    }
    
    (atp_produced / atp_consumed).max(0.0).min(10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atp_formatting() {
        assert_eq!(format_atp(100.0), "100.0 ATP");
        assert_eq!(format_atp(1500.0), "1.5k ATP");
    }

    #[test]
    fn test_efficiency_calculation() {
        assert_eq!(calculate_efficiency(100.0, 150.0), 1.5);
        assert_eq!(calculate_efficiency(0.0, 100.0), 0.0);
    }
} 