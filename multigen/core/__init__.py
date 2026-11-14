"""Core data contracts and population lineage management."""

from multigen.core.schema import CapabilityProfile, ModelNode, TrainingRecipe
from multigen.core.population_dag import PopulationDAG

__all__ = [
    "TrainingRecipe",
    "CapabilityProfile",
    "ModelNode",
    "PopulationDAG",
]
