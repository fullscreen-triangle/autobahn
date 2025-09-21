"""
Counterfactual Reasoning Unit (CRU)

Implements exponential "what could have happened?" exploration that characterizes
human consciousness, providing causation understanding through counterfactual analysis.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
import time
import itertools
from collections import defaultdict
import random


@dataclass
class CounterfactualScenario:
    """Represents a single counterfactual scenario."""
    scenario_id: str
    description: str
    causal_factors: List[str]
    probability: float
    deviation_from_actual: float
    required_changes: Dict[str, Any]
    

@dataclass
class CausationHypothesis:
    """Hypothesis about causal relationships."""
    factor: str
    causal_strength: float
    evidence: List[str]
    confidence: float


@dataclass
class CounterfactualAnalysis:
    """Complete counterfactual analysis result."""
    depth: int
    total_scenarios_explored: int
    primary_causes: List[CausationHypothesis]
    counterfactual_scenarios: List[CounterfactualScenario]
    causation_understanding: Dict[str, Any]
    exploration_time_ms: float


class CounterfactualUnit:
    """
    Counterfactual Reasoning Unit
    
    Implements human-level counterfactual reasoning through exponential exploration
    of "what could have happened?" scenarios to understand causation.
    """
    
    def __init__(self, max_depth: int = 5, branching_factor: int = 3):
        """
        Initialize Counterfactual Reasoning Unit.
        
        Args:
            max_depth: Maximum depth for counterfactual exploration
            branching_factor: Average number of alternatives per level
        """
        self.logger = logging.getLogger(__name__)
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.counterfactual_count = 0
        
        # Counterfactual exploration parameters
        self.causation_threshold = 0.7
        self.scenario_generation_limit = 1000  # Prevent exponential explosion
        self.confidence_threshold = 0.8
        
        self.logger.info(f"Counterfactual Unit initialized with depth={max_depth}")
    
    def generate_counterfactual_space(self, observed_configuration: Dict[str, Any],
                                    s_coordinates: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Generate complete counterfactual space for observed configuration.
        
        This implements the core consciousness mechanism: given what we observe,
        what is the complete space of scenarios that could have produced this?
        
        Args:
            observed_configuration: The actual observed configuration
            s_coordinates: S-entropy coordinates for guidance
            
        Returns:
            Dictionary containing counterfactual analysis results
        """
        start_time = time.time()
        
        # Step 1: Generate Level 1 counterfactuals (direct alternatives)
        level1_counterfactuals = self._generate_direct_counterfactuals(observed_configuration)
        
        # Step 2: Generate deeper counterfactual levels
        all_counterfactuals = [level1_counterfactuals]
        current_level = level1_counterfactuals
        
        for depth in range(2, self.max_depth + 1):
            next_level = self._generate_meta_counterfactuals(current_level, observed_configuration, depth)
            all_counterfactuals.append(next_level)
            current_level = next_level
            
            # Prevent exponential explosion
            if sum(len(level) for level in all_counterfactuals) > self.scenario_generation_limit:
                break
        
        # Step 3: Analyze causation through counterfactual comparison
        causation_analysis = self._analyze_causation_through_counterfactuals(
            observed_configuration, all_counterfactuals
        )
        
        # Step 4: Extract primary causal factors
        primary_causes = self._extract_primary_causal_factors(causation_analysis)
        
        # Step 5: Synthesize counterfactual understanding
        counterfactual_understanding = self._synthesize_counterfactual_understanding(
            all_counterfactuals, causation_analysis, primary_causes
        )
        
        exploration_time = (time.time() - start_time) * 1000
        self.counterfactual_count += 1
        
        return {
            'depth': len(all_counterfactuals),
            'total_scenarios': sum(len(level) for level in all_counterfactuals),
            'primary_causes': [cause.factor for cause in primary_causes],
            'causation_confidence': np.mean([cause.confidence for cause in primary_causes]) if primary_causes else 0.0,
            'counterfactual_scenarios': all_counterfactuals,
            'causation_analysis': causation_analysis,
            'understanding': counterfactual_understanding,
            'exploration_time_ms': exploration_time
        }
    
    def _generate_direct_counterfactuals(self, observed_config: Dict[str, Any]) -> List[CounterfactualScenario]:
        """
        Generate Level 1 counterfactuals: "What if X hadn't happened?"
        
        Args:
            observed_config: Observed configuration
            
        Returns:
            List of direct counterfactual scenarios
        """
        scenarios = []
        
        # Extract key factors from observed configuration
        key_factors = self._extract_key_factors(observed_config)
        
        for factor in key_factors:
            # Generate scenarios where this factor is different
            factor_alternatives = self._generate_factor_alternatives(factor, observed_config)
            
            for alt_idx, alternative in enumerate(factor_alternatives):
                scenario = CounterfactualScenario(
                    scenario_id=f"L1_{factor}_{alt_idx}",
                    description=f"What if {factor} had been {alternative} instead?",
                    causal_factors=[factor],
                    probability=self._estimate_scenario_probability(alternative, observed_config),
                    deviation_from_actual=self._calculate_deviation(alternative, observed_config[factor] if factor in observed_config else None),
                    required_changes={factor: alternative}
                )
                scenarios.append(scenario)
        
        return scenarios
    
    def _generate_meta_counterfactuals(self, previous_level: List[CounterfactualScenario],
                                     observed_config: Dict[str, Any],
                                     depth: int) -> List[CounterfactualScenario]:
        """
        Generate meta-counterfactuals: "What if they thought X?" or "What about Y causing Z?"
        
        Args:
            previous_level: Counterfactual scenarios from previous level
            observed_config: Original observed configuration
            depth: Current counterfactual depth
            
        Returns:
            List of meta-counterfactual scenarios
        """
        scenarios = []
        scenario_count = 0
        
        for prev_scenario in previous_level:
            if scenario_count >= self.scenario_generation_limit // depth:  # Limit per level
                break
                
            # Generate meta-counterfactuals based on previous scenario
            meta_alternatives = self._generate_meta_alternatives(prev_scenario, observed_config)
            
            for alt_idx, meta_alt in enumerate(meta_alternatives):
                scenario = CounterfactualScenario(
                    scenario_id=f"L{depth}_{prev_scenario.scenario_id}_{alt_idx}",
                    description=f"What if the cause of '{prev_scenario.description}' was {meta_alt}?",
                    causal_factors=prev_scenario.causal_factors + [meta_alt],
                    probability=prev_scenario.probability * self._estimate_meta_probability(meta_alt),
                    deviation_from_actual=prev_scenario.deviation_from_actual + 0.1,  # Increases with depth
                    required_changes={**prev_scenario.required_changes, f"meta_factor_{depth}": meta_alt}
                )
                scenarios.append(scenario)
                scenario_count += 1
        
        return scenarios
    
    def _analyze_causation_through_counterfactuals(self, observed_config: Dict[str, Any],
                                                  counterfactual_levels: List[List[CounterfactualScenario]]) -> Dict[str, Any]:
        """
        Analyze causation by comparing counterfactual scenarios with actual outcome.
        
        This implements the key insight: true causation is identified when
        removing a factor would change the outcome significantly.
        
        Args:
            observed_config: Actual observed configuration
            counterfactual_levels: All generated counterfactual scenarios
            
        Returns:
            Causation analysis results
        """
        causation_evidence = defaultdict(list)
        factor_impact_scores = defaultdict(float)
        
        # Flatten all counterfactual scenarios
        all_scenarios = [scenario for level in counterfactual_levels for scenario in level]
        
        for scenario in all_scenarios:
            for factor in scenario.causal_factors:
                # Calculate impact: how much would outcome change if this factor was different?
                impact_score = self._calculate_causal_impact(
                    factor, scenario.required_changes, observed_config
                )
                
                factor_impact_scores[factor] += impact_score
                causation_evidence[factor].append({
                    'scenario_id': scenario.scenario_id,
                    'impact_score': impact_score,
                    'description': scenario.description
                })
        
        # Normalize impact scores
        total_factors = len(factor_impact_scores)
        if total_factors > 0:
            for factor in factor_impact_scores:
                factor_impact_scores[factor] /= total_factors
        
        return {
            'factor_impact_scores': dict(factor_impact_scores),
            'causation_evidence': dict(causation_evidence),
            'total_scenarios_analyzed': len(all_scenarios)
        }
    
    def _extract_primary_causal_factors(self, causation_analysis: Dict[str, Any]) -> List[CausationHypothesis]:
        """
        Extract primary causal factors from counterfactual analysis.
        
        Args:
            causation_analysis: Results from causation analysis
            
        Returns:
            List of primary causal hypotheses
        """
        hypotheses = []
        factor_scores = causation_analysis['factor_impact_scores']
        evidence = causation_analysis['causation_evidence']
        
        # Sort factors by impact score
        sorted_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        
        for factor, impact_score in sorted_factors:
            if impact_score > self.causation_threshold:
                # Calculate confidence based on evidence consistency
                factor_evidence = evidence[factor]
                confidence = self._calculate_causal_confidence(factor_evidence, impact_score)
                
                hypothesis = CausationHypothesis(
                    factor=factor,
                    causal_strength=impact_score,
                    evidence=[ev['description'] for ev in factor_evidence],
                    confidence=confidence
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _synthesize_counterfactual_understanding(self, counterfactual_levels: List[List[CounterfactualScenario]],
                                               causation_analysis: Dict[str, Any],
                                               primary_causes: List[CausationHypothesis]) -> Dict[str, Any]:
        """
        Synthesize overall understanding from counterfactual analysis.
        
        Args:
            counterfactual_levels: All counterfactual scenarios
            causation_analysis: Causation analysis results
            primary_causes: Primary causal factors
            
        Returns:
            Synthesized counterfactual understanding
        """
        total_scenarios = sum(len(level) for level in counterfactual_levels)
        
        # Analyze counterfactual complexity (measure of consciousness-level processing)
        counterfactual_complexity = len(counterfactual_levels) * np.mean([len(level) for level in counterfactual_levels])
        
        # Extract key insights
        causal_certainty = np.mean([cause.confidence for cause in primary_causes]) if primary_causes else 0.0
        alternative_possibility_space = total_scenarios
        
        # Determine understanding quality
        if causal_certainty > 0.8 and len(primary_causes) > 0:
            understanding_quality = "High - Clear causal understanding"
        elif causal_certainty > 0.6:
            understanding_quality = "Medium - Some causal clarity"
        else:
            understanding_quality = "Low - Uncertain causation"
        
        return {
            'counterfactual_complexity': counterfactual_complexity,
            'causal_certainty': causal_certainty,
            'alternative_possibility_space': alternative_possibility_space,
            'understanding_quality': understanding_quality,
            'primary_causal_narrative': self._generate_causal_narrative(primary_causes),
            'exploration_depth': len(counterfactual_levels),
            'consciousness_level_indicator': min(counterfactual_complexity / 100.0, 1.0)  # Normalized
        }
    
    # Helper methods for counterfactual generation
    def _extract_key_factors(self, config: Dict[str, Any]) -> List[str]:
        """Extract key factors from configuration for counterfactual analysis."""
        factors = []
        
        for key, value in config.items():
            if isinstance(value, dict):
                # Extract sub-factors from nested dictionaries
                for sub_key in value.keys():
                    factors.append(f"{key}.{sub_key}")
            else:
                factors.append(key)
        
        # Limit number of factors to prevent explosion
        return factors[:10]
    
    def _generate_factor_alternatives(self, factor: str, config: Dict[str, Any]) -> List[Any]:
        """Generate alternative values for a factor."""
        current_value = config.get(factor)
        alternatives = []
        
        if isinstance(current_value, bool):
            alternatives = [not current_value]
        elif isinstance(current_value, (int, float)):
            alternatives = [current_value * 0.5, current_value * 1.5, 0]
        elif isinstance(current_value, str):
            alternatives = ["alternative_" + current_value, "opposite_" + current_value, ""]
        else:
            alternatives = ["alternative_state", "absent", "modified"]
        
        return alternatives[:3]  # Limit alternatives
    
    def _generate_meta_alternatives(self, scenario: CounterfactualScenario, config: Dict[str, Any]) -> List[str]:
        """Generate meta-level alternatives for deeper counterfactual exploration."""
        meta_alternatives = [
            "environmental_influence",
            "temporal_factor", 
            "hidden_variable",
            "interaction_effect",
            "observer_bias",
            "measurement_error"
        ]
        
        # Add scenario-specific meta-alternatives
        for factor in scenario.causal_factors:
            meta_alternatives.extend([
                f"precondition_for_{factor}",
                f"consequence_of_{factor}",
                f"interaction_with_{factor}"
            ])
        
        return meta_alternatives[:self.branching_factor]
    
    # Probability and impact calculation methods
    def _estimate_scenario_probability(self, alternative: Any, config: Dict[str, Any]) -> float:
        """Estimate probability of counterfactual scenario."""
        # Simple heuristic based on deviation from observed
        if alternative is None or alternative == "absent":
            return 0.1
        elif str(alternative).startswith("opposite") or str(alternative).startswith("alternative"):
            return 0.3
        else:
            return 0.2
    
    def _estimate_meta_probability(self, meta_factor: str) -> float:
        """Estimate probability of meta-level factor."""
        if "environmental" in meta_factor or "temporal" in meta_factor:
            return 0.4
        elif "hidden" in meta_factor or "interaction" in meta_factor:
            return 0.2
        else:
            return 0.1
    
    def _calculate_deviation(self, alternative: Any, actual: Any) -> float:
        """Calculate deviation between alternative and actual values."""
        if actual is None:
            return 0.5
        
        if type(alternative) != type(actual):
            return 1.0
        
        if isinstance(actual, (int, float)) and isinstance(alternative, (int, float)):
            return abs(alternative - actual) / max(abs(actual), 1.0)
        elif isinstance(actual, str) and isinstance(alternative, str):
            return 1.0 if alternative != actual else 0.0
        else:
            return 0.5
    
    def _calculate_causal_impact(self, factor: str, changes: Dict[str, Any], config: Dict[str, Any]) -> float:
        """Calculate how much impact a factor has on the outcome."""
        # Heuristic for causal impact based on factor type and change magnitude
        impact_score = 0.0
        
        if factor in changes:
            change = changes[factor]
            original = config.get(factor)
            
            # Calculate impact based on change magnitude
            deviation = self._calculate_deviation(change, original)
            impact_score = deviation * 0.8  # Scale impact
            
        # Additional impact for meta-factors (they represent deeper causation)
        if factor.startswith("meta_factor") or "interaction" in factor:
            impact_score *= 1.2
        
        return min(impact_score, 1.0)
    
    def _calculate_causal_confidence(self, evidence: List[Dict[str, Any]], impact_score: float) -> float:
        """Calculate confidence in causal hypothesis based on evidence."""
        if not evidence:
            return 0.0
        
        # Confidence based on evidence consistency and impact strength
        evidence_consistency = len(evidence) / 10.0  # More evidence = higher confidence
        impact_factor = impact_score
        
        confidence = (evidence_consistency + impact_factor) / 2.0
        return min(confidence, 1.0)
    
    def _generate_causal_narrative(self, primary_causes: List[CausationHypothesis]) -> str:
        """Generate natural language narrative of causal understanding."""
        if not primary_causes:
            return "No clear causal factors identified through counterfactual analysis."
        
        narrative_parts = []
        for cause in primary_causes[:3]:  # Top 3 causes
            narrative_parts.append(f"{cause.factor} (strength: {cause.causal_strength:.2f}, confidence: {cause.confidence:.2f})")
        
        return f"Primary causal factors: {', '.join(narrative_parts)}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current unit status."""
        return {
            'max_depth': self.max_depth,
            'branching_factor': self.branching_factor,
            'counterfactual_count': self.counterfactual_count,
            'causation_threshold': self.causation_threshold,
            'status': 'operational'
        }
