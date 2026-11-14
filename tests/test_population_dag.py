"""Unit tests for Population DAG management (multigen/core/population_dag.py)."""

from __future__ import annotations

import tempfile
import pytest
from multigen.core.population_dag import PopulationDAG
from multigen.core.schema import CapabilityProfile, ModelNode, TrainingRecipe


def test_population_dag_root_initialization():
    dag = PopulationDAG(root_model_name="Qwen/Qwen3-0.6B")
    assert dag.has_node("node_0")
    root = dag.get_node("node_0")
    assert root is not None
    assert root.node_id == "node_0"
    assert root.recipe.base_model_name_or_path == "Qwen/Qwen3-0.6B"
    assert root.depth == 0
    assert root.cumulative_flops == 0.0


def test_population_dag_add_child_and_edges():
    dag = PopulationDAG()
    child_recipe = TrainingRecipe(seed=42, target_flop_budget=1.0e16)
    child_node = ModelNode(
        node_id="exp1_1_seed42",
        parent_ids=["node_0"],
        depth=1,
        recipe=child_recipe,
        cumulative_flops=1.0e16,
    )
    dag.add_node(child_node)

    assert dag.has_node("exp1_1_seed42")
    assert dag.get_parents("exp1_1_seed42") == ["node_0"]
    assert dag.get_children("node_0") == ["exp1_1_seed42"]
    assert "node_0" in dag.get_ancestors("exp1_1_seed42")
    assert "exp1_1_seed42" in dag.get_descendants("node_0")


def test_population_dag_lineage_path_and_flops():
    dag = PopulationDAG()
    # node_0 -> child_1 -> grandchild_1
    child_node = ModelNode(
        node_id="child_1",
        parent_ids=["node_0"],
        depth=1,
        recipe=TrainingRecipe(),
        cumulative_flops=1.0e16,
    )
    grandchild_node = ModelNode(
        node_id="grandchild_1",
        parent_ids=["child_1"],
        depth=2,
        recipe=TrainingRecipe(),
        cumulative_flops=5.0e15,
    )
    dag.add_node(child_node)
    dag.add_node(grandchild_node)

    path = dag.get_lineage_path("grandchild_1", "node_0")
    assert path == ["node_0", "child_1", "grandchild_1"]

    total_flops = dag.get_total_lineage_flops("grandchild_1")
    # node_0 (0.0) + child_1 (1.0e16) + grandchild_1 (5.0e15) = 1.5e16
    assert total_flops == 1.5e16


def test_population_dag_update_profile():
    dag = PopulationDAG()
    profile = CapabilityProfile(
        benchmark="gsm8k",
        split="test",
        accuracy=0.65,
        permissive_accuracy=0.70,
        format_compliance_rate=0.96,
        mean_completion_tokens=210.0,
    )
    dag.update_capability_profile("node_0", profile)
    root = dag.get_node("node_0")
    assert root is not None
    assert root.capability_profile is not None
    assert root.capability_profile.accuracy == 0.65


def test_population_dag_json_persistence():
    dag = PopulationDAG()
    child = ModelNode(
        node_id="exp1_1_grpo_seed42",
        parent_ids=["node_0"],
        depth=1,
        recipe=TrainingRecipe(seed=42),
        cumulative_flops=1.0e16,
    )
    dag.add_node(child)

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        dag.save_to_json(f.name)
        loaded_dag = PopulationDAG.load_from_json(f.name)

    assert loaded_dag.has_node("node_0")
    assert loaded_dag.has_node("exp1_1_grpo_seed42")
    assert loaded_dag.get_parents("exp1_1_grpo_seed42") == ["node_0"]
    loaded_child = loaded_dag.get_node("exp1_1_grpo_seed42")
    assert loaded_child is not None
    assert loaded_child.cumulative_flops == 1.0e16
