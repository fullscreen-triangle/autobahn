"""
Gas Molecular Processing Engine (GMPE)

Implements thermodynamic information processing where information elements operate
as gas molecules seeking equilibrium states for optimal meaning extraction.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import time
from scipy.integrate import odeint
from scipy.optimize import minimize
import random


@dataclass
class InformationGasMolecule:
    """Information Gas Molecule (IGM) with thermodynamic properties."""
    molecule_id: str
    semantic_energy: float      # E_i - internal semantic energy
    information_entropy: float  # S_i - entropy content
    processing_temperature: float  # T_i - processing temperature
    semantic_pressure: float    # P_i - semantic pressure
    information_volume: float   # V_i - information volume
    chemical_potential: float   # μ_i - chemical potential
    velocity_vector: np.ndarray # v_i - velocity in semantic space
    formal_proof_validity: bool # Φ_i - proof of validity
    
    def gibbs_free_energy(self) -> float:
        """Calculate Gibbs free energy of the IGM."""
        return self.semantic_energy - self.processing_temperature * self.information_entropy + \
               self.semantic_pressure * self.information_volume
    
    def update_state(self, dt: float, external_forces: np.ndarray = None):
        """Update IGM state according to thermodynamic equations."""
        if external_forces is None:
            external_forces = np.zeros(3)
        
        # Update velocity based on forces
        self.velocity_vector += external_forces * dt
        
        # Update thermodynamic properties based on interactions
        self.processing_temperature += np.random.normal(0, 0.01) * dt
        self.information_entropy += np.random.normal(0, 0.005) * dt
        
        # Ensure physical bounds
        self.processing_temperature = max(0.1, self.processing_temperature)
        self.information_entropy = max(0.0, self.information_entropy)


@dataclass
class ThermodynamicState:
    """Complete thermodynamic state of the gas molecular system."""
    total_semantic_energy: float
    total_entropy: float
    system_temperature: float
    system_pressure: float
    system_volume: float
    information_density: float
    equilibrium_reached: bool
    convergence_iterations: int


class GasMolecularEngine:
    """
    Gas Molecular Processing Engine
    
    Processes information through thermodynamic gas dynamics where optimal meaning
    corresponds to thermodynamic equilibrium states.
    """
    
    def __init__(self, molecular_count: int = 100, system_volume: float = 1.0):
        """
        Initialize Gas Molecular Processing Engine.
        
        Args:
            molecular_count: Number of Information Gas Molecules
            system_volume: Volume of the processing system
        """
        self.logger = logging.getLogger(__name__)
        self.molecular_count = molecular_count
        self.system_volume = system_volume
        self.processing_count = 0
        
        # Thermodynamic parameters
        self.boltzmann_constant = 1.0  # Normalized
        self.equilibrium_threshold = 1e-6
        self.max_equilibrium_iterations = 1000
        self.dt = 0.01  # Time step for evolution
        
        # Initialize molecular system
        self.molecules: List[InformationGasMolecule] = []
        self.current_state = None
        
        self.logger.info(f"Gas Molecular Engine initialized with {molecular_count} IGMs")
    
    def process_to_equilibrium(self, information_input: Dict[str, Any],
                             environmental_perturbation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process information through gas molecular dynamics to thermodynamic equilibrium.
        
        This implements the core theoretical claim that optimal meaning extraction
        corresponds to thermodynamic equilibrium states.
        
        Args:
            information_input: Input information to process
            environmental_perturbation: Environmental perturbation data
            
        Returns:
            Dictionary containing equilibrium state and extracted meaning
        """
        start_time = time.time()
        
        # Step 1: Convert information input to IGMs
        self.molecules = self._convert_information_to_igms(information_input)
        
        # Step 2: Apply environmental perturbation
        self._apply_environmental_perturbation(environmental_perturbation)
        
        # Step 3: Evolve system to thermodynamic equilibrium
        equilibrium_state = self._evolve_to_equilibrium()
        
        # Step 4: Extract meaning from equilibrium state
        extracted_meaning = self._extract_meaning_from_equilibrium(equilibrium_state)
        
        processing_time = (time.time() - start_time) * 1000
        self.processing_count += 1
        
        return {
            'equilibrium_state': equilibrium_state.__dict__,
            'extracted_meaning': extracted_meaning,
            'molecular_configurations': [mol.__dict__ for mol in self.molecules],
            'processing_time_ms': processing_time,
            'convergence_achieved': equilibrium_state.equilibrium_reached
        }
    
    def _convert_information_to_igms(self, information_input: Dict[str, Any]) -> List[InformationGasMolecule]:
        """
        Convert input information into Information Gas Molecules.
        
        Args:
            information_input: Information to convert
            
        Returns:
            List of Information Gas Molecules
        """
        molecules = []
        
        # Extract information elements from input
        info_elements = self._extract_information_elements(information_input)
        
        for i, element in enumerate(info_elements[:self.molecular_count]):
            # Calculate thermodynamic properties from information content
            semantic_energy = self._calculate_semantic_energy(element)
            info_entropy = self._calculate_information_entropy(element)
            proc_temperature = self._calculate_processing_temperature(element)
            semantic_pressure = self._calculate_semantic_pressure(element, info_entropy)
            info_volume = self._calculate_information_volume(element)
            chemical_potential = semantic_energy + semantic_pressure * info_volume
            
            # Initialize velocity with random thermal motion
            velocity = np.random.normal(0, np.sqrt(proc_temperature), 3)
            
            # Create IGM
            molecule = InformationGasMolecule(
                molecule_id=f"IGM_{i}",
                semantic_energy=semantic_energy,
                information_entropy=info_entropy,
                processing_temperature=proc_temperature,
                semantic_pressure=semantic_pressure,
                information_volume=info_volume,
                chemical_potential=chemical_potential,
                velocity_vector=velocity,
                formal_proof_validity=True  # Simplified - assume valid
            )
            
            molecules.append(molecule)
        
        # Fill remaining slots with background molecules if needed
        while len(molecules) < self.molecular_count:
            molecules.append(self._create_background_molecule(len(molecules)))
        
        return molecules
    
    def _apply_environmental_perturbation(self, perturbation: Dict[str, Any]):
        """
        Apply environmental perturbation to the gas molecular system.
        
        Args:
            perturbation: Environmental perturbation data
        """
        perturbation_strength = self._calculate_perturbation_strength(perturbation)
        
        for molecule in self.molecules:
            # Apply perturbation to thermodynamic properties
            molecule.semantic_energy += perturbation_strength * np.random.normal(0, 0.1)
            molecule.processing_temperature += perturbation_strength * np.random.normal(0, 0.05)
            molecule.information_entropy += perturbation_strength * np.random.normal(0, 0.02)
            
            # Apply force perturbation
            perturbation_force = perturbation_strength * np.random.normal(0, 0.1, 3)
            molecule.velocity_vector += perturbation_force
    
    def _evolve_to_equilibrium(self) -> ThermodynamicState:
        """
        Evolve the gas molecular system to thermodynamic equilibrium.
        
        Returns:
            Final thermodynamic state
        """
        previous_gibbs_energy = float('inf')
        convergence_iterations = 0
        
        for iteration in range(self.max_equilibrium_iterations):
            # Update each molecule's state
            for molecule in self.molecules:
                # Calculate forces from other molecules
                intermolecular_forces = self._calculate_intermolecular_forces(molecule)
                
                # Update molecular state
                molecule.update_state(self.dt, intermolecular_forces)
            
            # Calculate current system Gibbs energy
            current_gibbs_energy = self._calculate_system_gibbs_energy()
            
            # Check for equilibrium convergence
            energy_change = abs(current_gibbs_energy - previous_gibbs_energy)
            if energy_change < self.equilibrium_threshold:
                convergence_iterations = iteration + 1
                break
            
            previous_gibbs_energy = current_gibbs_energy
        
        # Calculate final thermodynamic state
        final_state = self._calculate_thermodynamic_state()
        final_state.equilibrium_reached = convergence_iterations < self.max_equilibrium_iterations
        final_state.convergence_iterations = convergence_iterations
        
        return final_state
    
    def _calculate_intermolecular_forces(self, target_molecule: InformationGasMolecule) -> np.ndarray:
        """
        Calculate forces between molecules for thermodynamic evolution.
        
        Args:
            target_molecule: Molecule to calculate forces for
            
        Returns:
            Force vector acting on the molecule
        """
        total_force = np.zeros(3)
        
        for other_molecule in self.molecules:
            if other_molecule.molecule_id == target_molecule.molecule_id:
                continue
            
            # Simple interaction model based on semantic similarity
            semantic_distance = abs(target_molecule.semantic_energy - other_molecule.semantic_energy)
            entropy_distance = abs(target_molecule.information_entropy - other_molecule.information_entropy)
            
            # Attractive force for similar semantics, repulsive for dissimilar
            if semantic_distance < 0.1:  # Similar semantics
                force_magnitude = 0.01 / max(semantic_distance, 0.01)
            else:  # Dissimilar semantics
                force_magnitude = -0.005 / max(semantic_distance, 0.1)
            
            # Random direction for simplicity (in full implementation, would be position-based)
            force_direction = np.random.normal(0, 1, 3)
            force_direction /= np.linalg.norm(force_direction) + 1e-10
            
            total_force += force_magnitude * force_direction
        
        return total_force
    
    def _calculate_system_gibbs_energy(self) -> float:
        """Calculate total Gibbs free energy of the system."""
        total_gibbs = 0.0
        
        for molecule in self.molecules:
            total_gibbs += molecule.gibbs_free_energy()
        
        # Add interaction energy
        interaction_energy = self._calculate_interaction_energy()
        total_gibbs += interaction_energy
        
        return total_gibbs
    
    def _calculate_interaction_energy(self) -> float:
        """Calculate intermolecular interaction energy."""
        interaction_energy = 0.0
        
        for i, mol1 in enumerate(self.molecules):
            for j, mol2 in enumerate(self.molecules[i+1:], i+1):
                # Simple pairwise interaction
                semantic_diff = abs(mol1.semantic_energy - mol2.semantic_energy)
                entropy_diff = abs(mol1.information_entropy - mol2.information_entropy)
                
                # Interaction strength based on similarity
                if semantic_diff < 0.2:
                    interaction_energy -= 0.01  # Attractive interaction
                else:
                    interaction_energy += 0.005 * semantic_diff  # Repulsive
        
        return interaction_energy
    
    def _calculate_thermodynamic_state(self) -> ThermodynamicState:
        """Calculate current thermodynamic state of the system."""
        total_energy = sum(mol.semantic_energy for mol in self.molecules)
        total_entropy = sum(mol.information_entropy for mol in self.molecules)
        
        # System properties
        avg_temperature = np.mean([mol.processing_temperature for mol in self.molecules])
        avg_pressure = np.mean([mol.semantic_pressure for mol in self.molecules])
        total_volume = sum(mol.information_volume for mol in self.molecules)
        info_density = total_energy / max(total_volume, 0.1)
        
        return ThermodynamicState(
            total_semantic_energy=total_energy,
            total_entropy=total_entropy,
            system_temperature=avg_temperature,
            system_pressure=avg_pressure,
            system_volume=total_volume,
            information_density=info_density,
            equilibrium_reached=False,  # Set by caller
            convergence_iterations=0    # Set by caller
        )
    
    def _extract_meaning_from_equilibrium(self, equilibrium_state: ThermodynamicState) -> Dict[str, Any]:
        """
        Extract meaning from thermodynamic equilibrium state.
        
        This implements the theoretical claim that meaning emerges from
        equilibrium configurations of Information Gas Molecules.
        
        Args:
            equilibrium_state: Thermodynamic equilibrium state
            
        Returns:
            Extracted meaning information
        """
        # Analyze molecular configurations at equilibrium
        high_energy_molecules = [mol for mol in self.molecules if mol.semantic_energy > 0.7]
        low_entropy_molecules = [mol for mol in self.molecules if mol.information_entropy < 0.3]
        stable_molecules = [mol for mol in self.molecules if abs(mol.gibbs_free_energy()) < 0.1]
        
        # Extract meaning indicators
        meaning_complexity = len(high_energy_molecules) / len(self.molecules)
        meaning_certainty = len(low_entropy_molecules) / len(self.molecules) 
        meaning_stability = len(stable_molecules) / len(self.molecules)
        
        # Classify meaning type based on equilibrium characteristics
        if equilibrium_state.information_density > 0.8:
            meaning_type = "High-density information processing"
        elif equilibrium_state.system_temperature > 0.7:
            meaning_type = "Dynamic information exploration"
        elif meaning_stability > 0.6:
            meaning_type = "Stable meaning configuration"
        else:
            meaning_type = "Uncertain information state"
        
        # Generate semantic content from molecular patterns
        semantic_patterns = self._analyze_molecular_patterns()
        
        return {
            'meaning_type': meaning_type,
            'complexity_level': meaning_complexity,
            'certainty_level': meaning_certainty,
            'stability_level': meaning_stability,
            'semantic_patterns': semantic_patterns,
            'thermodynamic_indicators': {
                'energy_density': equilibrium_state.information_density,
                'entropy_level': equilibrium_state.total_entropy / len(self.molecules),
                'temperature': equilibrium_state.system_temperature,
                'pressure': equilibrium_state.system_pressure
            }
        }
    
    # Information processing helper methods
    def _extract_information_elements(self, information: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract information elements for IGM conversion."""
        elements = []
        
        def extract_recursive(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        extract_recursive(value, f"{prefix}.{key}" if prefix else key)
                    else:
                        elements.append({
                            'key': f"{prefix}.{key}" if prefix else key,
                            'value': value,
                            'type': type(value).__name__
                        })
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    extract_recursive(item, f"{prefix}[{i}]")
        
        extract_recursive(information)
        return elements
    
    def _calculate_semantic_energy(self, element: Dict[str, Any]) -> float:
        """Calculate semantic energy of an information element."""
        if 'value' not in element:
            return 0.1
        
        value = element['value']
        if isinstance(value, (int, float)):
            return min(abs(float(value)) * 0.1, 1.0)
        elif isinstance(value, str):
            return min(len(value) * 0.05, 1.0)
        else:
            return 0.3
    
    def _calculate_information_entropy(self, element: Dict[str, Any]) -> float:
        """Calculate information entropy of an element."""
        value = element.get('value', '')
        
        if isinstance(value, str) and value:
            # Simple entropy based on character distribution
            char_counts = {}
            for char in value:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            entropy = 0.0
            total_chars = len(value)
            for count in char_counts.values():
                prob = count / total_chars
                entropy -= prob * np.log2(prob + 1e-10)
            
            return min(entropy / 10.0, 1.0)  # Normalize
        else:
            return 0.5  # Default entropy
    
    def _calculate_processing_temperature(self, element: Dict[str, Any]) -> float:
        """Calculate processing temperature for an element."""
        # Temperature based on complexity and uncertainty
        complexity = len(str(element.get('value', '')))
        return min(0.1 + complexity * 0.01, 1.0)
    
    def _calculate_semantic_pressure(self, element: Dict[str, Any], entropy: float) -> float:
        """Calculate semantic pressure based on element properties."""
        return max(0.1, entropy * 0.8 + np.random.normal(0, 0.1))
    
    def _calculate_information_volume(self, element: Dict[str, Any]) -> float:
        """Calculate information volume of an element."""
        value_size = len(str(element.get('value', '')))
        return min(0.1 + value_size * 0.01, 1.0)
    
    def _create_background_molecule(self, idx: int) -> InformationGasMolecule:
        """Create a background molecule to fill the system."""
        return InformationGasMolecule(
            molecule_id=f"BG_{idx}",
            semantic_energy=np.random.uniform(0.1, 0.3),
            information_entropy=np.random.uniform(0.3, 0.7),
            processing_temperature=np.random.uniform(0.2, 0.5),
            semantic_pressure=np.random.uniform(0.1, 0.4),
            information_volume=np.random.uniform(0.1, 0.3),
            chemical_potential=np.random.uniform(0.1, 0.5),
            velocity_vector=np.random.normal(0, 0.1, 3),
            formal_proof_validity=True
        )
    
    def _calculate_perturbation_strength(self, perturbation: Dict[str, Any]) -> float:
        """Calculate strength of environmental perturbation."""
        if not perturbation:
            return 0.1
        
        # Simple heuristic based on perturbation complexity
        total_elements = 0
        for value in perturbation.values():
            if isinstance(value, (list, tuple)):
                total_elements += len(value)
            elif isinstance(value, dict):
                total_elements += len(value)
            else:
                total_elements += 1
        
        return min(total_elements * 0.05, 1.0)
    
    def _analyze_molecular_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in molecular configurations."""
        patterns = []
        
        # Energy clustering patterns
        energy_levels = [mol.semantic_energy for mol in self.molecules]
        high_energy_count = sum(1 for e in energy_levels if e > 0.7)
        if high_energy_count > len(self.molecules) * 0.3:
            patterns.append({
                'type': 'high_energy_cluster',
                'description': f'{high_energy_count} molecules in high-energy state',
                'significance': 'Indicates complex information processing'
            })
        
        # Entropy distribution patterns
        entropy_levels = [mol.information_entropy for mol in self.molecules]
        low_entropy_count = sum(1 for e in entropy_levels if e < 0.3)
        if low_entropy_count > len(self.molecules) * 0.4:
            patterns.append({
                'type': 'low_entropy_cluster',
                'description': f'{low_entropy_count} molecules in low-entropy state',
                'significance': 'Indicates organized information structure'
            })
        
        return patterns
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            'molecular_count': self.molecular_count,
            'system_volume': self.system_volume,
            'processing_count': self.processing_count,
            'equilibrium_threshold': self.equilibrium_threshold,
            'current_molecules': len(self.molecules),
            'status': 'operational'
        }
