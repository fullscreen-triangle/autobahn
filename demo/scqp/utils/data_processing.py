"""Data processing utilities for SCQP"""

import numpy as np
from typing import Dict, Any, Tuple, List


def normalize_coordinates(coordinates: Tuple[float, float, float], 
                         bounds: Tuple[float, float] = (0.0, 1.0)) -> Tuple[float, float, float]:
    """Normalize coordinates to specified bounds."""
    min_bound, max_bound = bounds
    return tuple(max(min_bound, min(max_bound, coord)) for coord in coordinates)


def validate_environmental_context(context: Dict[str, Any]) -> bool:
    """Validate environmental context structure."""
    required_modalities = ['visual', 'audio', 'semantic']
    return all(modality in context for modality in required_modalities)


def extract_pattern_features(data: Dict[str, Any]) -> List[float]:
    """Extract numerical features from pattern data."""
    features = []
    
    def extract_recursive(obj):
        if isinstance(obj, (int, float)):
            features.append(float(obj))
        elif isinstance(obj, str):
            features.append(len(obj) * 0.1)  # Simple string feature
        elif isinstance(obj, dict):
            for value in obj.values():
                extract_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_recursive(item)
    
    extract_recursive(data)
    return features
