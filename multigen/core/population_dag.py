"""Directed Acyclic Graph (DAG) population manager.

Maintains population lineage, tracks parent-child mutations and compute ancestry,
and persists tree states to structured JSON files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import networkx as nx

from multigen.core.schema import CapabilityProfile, ModelNode, TrainingRecipe

logger = logging.getLogger(__name__)


class PopulationDAG:
    """NetworkX DiGraph wrapper managing model lineage and compute ancestry."""

    def __init__(self, root_model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.root_model_name = root_model_name
        self.graph = nx.DiGraph()
        self._ensure_root_node()

    def _ensure_root_node(self) -> None:
        """Register the frozen base model as node_0 if DAG is empty."""
        if "node_0" not in self.graph:
            root_recipe = TrainingRecipe(
                algorithm="sft",
                base_model_name_or_path=self.root_model_name,
                target_flop_budget=0.0,
            )
            root_node = ModelNode(
                node_id="node_0",
                parent_ids=[],
                depth=0,
                recipe=root_recipe,
                adapter_path=None,
                cumulative_flops=0.0,
                metadata={"description": f"Root base model ({self.root_model_name})"},
            )
            self.add_node(root_node)

    def add_edge(
        self, source_id: str, target_id: str, edge_type: str = "mutation", **attrs: Any
    ) -> None:
        """Add a directed edge between two nodes with semantic attributes."""
        self.graph.add_edge(source_id, target_id, edge_type=edge_type, **attrs)

    def register_teacher_node(
        self,
        teacher_id: str = "teacher_qwen25_math_7b",
        model_name_or_path: str = "Qwen/Qwen2.5-Math-7B-Instruct",
        params: float = 7.61e9,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelNode:
        """Register an external reference teacher node in the population DAG."""
        if teacher_id in self.graph:
            return self.get_node(teacher_id)  # type: ignore[return-value]

        meta = {"is_external_teacher": True, "params": params}
        if metadata:
            meta.update(metadata)
        recipe = TrainingRecipe(
            algorithm="distill",
            base_model_name_or_path=model_name_or_path,
            target_flop_budget=0.0,
        )
        teacher_node = ModelNode(
            node_id=teacher_id,
            parent_ids=[],
            depth=0,
            recipe=recipe,
            adapter_path=None,
            cumulative_flops=0.0,
            metadata=meta,
        )
        self.add_node(teacher_node)
        return teacher_node

    def add_node(self, node: ModelNode, edge_type: str = "mutation") -> None:
        """Add a model checkpoint node to the graph and connect to parent nodes."""
        self.graph.add_node(node.node_id, data=node.model_dump())
        for parent_id in node.parent_ids:
            if parent_id in self.graph:
                parent_data = self.graph.nodes[parent_id].get("data", {})
                is_teacher = (
                    parent_data.get("metadata", {}).get("is_external_teacher", False)
                    or "teacher" in parent_id.lower()
                )
                etype = "distillation" if is_teacher else edge_type
                self.graph.add_edge(parent_id, node.node_id, edge_type=etype)
            else:
                logger.warning(
                    "Parent node %s not found in graph when adding %s.",
                    parent_id,
                    node.node_id,
                )

    def get_node(self, node_id: str) -> Optional[ModelNode]:
        """Retrieve ModelNode by its identifier."""
        if node_id in self.graph:
            raw_data = self.graph.nodes[node_id]["data"]
            return ModelNode.model_validate(raw_data)
        return None

    def has_node(self, node_id: str) -> bool:
        """Check if node exists in the DAG."""
        return node_id in self.graph

    def update_capability_profile(
        self, node_id: str, profile: CapabilityProfile
    ) -> None:
        """Attach or update evaluation results for an existing node."""
        node = self.get_node(node_id)
        if node is None:
            raise KeyError(f"Node {node_id} does not exist in population DAG.")
        node.capability_profile = profile
        self.graph.nodes[node_id]["data"] = node.model_dump()

    def get_parents(self, node_id: str) -> List[str]:
        """Return immediate parent node IDs."""
        if node_id not in self.graph:
            return []
        return list(self.graph.predecessors(node_id))

    def get_children(self, node_id: str) -> List[str]:
        """Return immediate child node IDs."""
        if node_id not in self.graph:
            return []
        return list(self.graph.successors(node_id))

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Return all ancestor node IDs in the compute tree."""
        if node_id not in self.graph:
            return set()
        return nx.ancestors(self.graph, node_id)

    def get_descendants(self, node_id: str) -> Set[str]:
        """Return all descendant node IDs."""
        if node_id not in self.graph:
            return set()
        return nx.descendants(self.graph, node_id)

    def get_lineage_path(
        self, target_node_id: str, source_node_id: str = "node_0"
    ) -> List[str]:
        """Return shortest lineage path from root to target node."""
        if source_node_id not in self.graph or target_node_id not in self.graph:
            return []
        try:
            return nx.shortest_path(self.graph, source=source_node_id, target=target_node_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_total_lineage_flops(self, node_id: str) -> float:
        """Calculate cumulative FLOPs consumed across all ancestors leading to this node."""
        node = self.get_node(node_id)
        if node is None:
            return 0.0
        ancestors = self.get_ancestors(node_id)
        total_flops = node.cumulative_flops
        for anc_id in ancestors:
            anc_node = self.get_node(anc_id)
            if anc_node is not None:
                total_flops += anc_node.cumulative_flops
        return total_flops

    def get_all_nodes(self) -> List[ModelNode]:
        """Return list of all registered ModelNodes."""
        nodes: List[ModelNode] = []
        for n in self.graph.nodes:
            raw = self.graph.nodes[n]["data"]
            nodes.append(ModelNode.model_validate(raw))
        return nodes

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to a serializable dictionary."""
        edges_data: List[Any] = []
        for u, v, data in self.graph.edges(data=True):
            if data:
                edges_data.append({"source": u, "target": v, **data})
            else:
                edges_data.append([u, v])
        return {
            "root_model_name": self.root_model_name,
            "nodes": [self.graph.nodes[n]["data"] for n in self.graph.nodes],
            "edges": edges_data,
        }

    def save_to_json(self, filepath: str) -> None:
        """Save graph to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def save(self, filepath: str) -> None:
        """Alias for save_to_json."""
        self.save_to_json(filepath)

    @classmethod
    def load_from_json(cls, filepath: str) -> "PopulationDAG":
        """Load graph from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        root_model = data.get("root_model_name", "Qwen/Qwen3-0.6B")
        dag = cls(root_model_name=root_model)
        dag.graph.clear()

        for node_dict in data.get("nodes", []):
            node = ModelNode.model_validate(node_dict)
            dag.graph.add_node(node.node_id, data=node.model_dump())

        for edge in data.get("edges", []):
            if isinstance(edge, dict):
                src = edge.get("source")
                tgt = edge.get("target")
                attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
                if src and tgt:
                    dag.graph.add_edge(src, tgt, **attrs)
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                dag.graph.add_edge(edge[0], edge[1])

        return dag

    @classmethod
    def load_or_create(cls, filepath: str, root_model_name: str = "Qwen/Qwen3-0.6B") -> "PopulationDAG":
        """Load DAG from JSON file if it exists, otherwise create a new DAG."""
        if Path(filepath).exists():
            return cls.load_from_json(filepath)
        dag = cls(root_model_name=root_model_name)
        dag.save_to_json(filepath)
        return dag
