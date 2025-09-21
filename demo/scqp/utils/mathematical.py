"""Mathematical utility functions for SCQP"""

import numpy as np
from typing import Tuple, List


def s_entropy_distance(coord1: Tuple[float, float, float], 
                      coord2: Tuple[float, float, float]) -> float:
    """Calculate S-entropy distance between two coordinates."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))


def gibbs_energy_calculation(energy: float, temperature: float, 
                           entropy: float, pressure: float, volume: float) -> float:
    """Calculate Gibbs free energy: G = H - TS = E + PV - TS"""
    enthalpy = energy + pressure * volume
    return enthalpy - temperature * entropy


def precision_by_difference(local_time: float, reference_time: float) -> float:
    """Calculate precision-by-difference metric."""
    return reference_time - local_time
