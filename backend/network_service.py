from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


ENTITY_COLUMNS = {
    "Declarant ID": ("declarant", 1.0),
    "Importer ID": ("importer", 1.0),
    "Seller ID": ("seller", 1.2),
    "Courier ID": ("courier", 0.6),
}


class NetworkService:
    """Structural network detector adapted from the Salma graph pipeline.

    It deliberately does not use Fraud labels. Shared declarants, importers,
    sellers and couriers form a transaction graph; small dense components and
    multi-role bridges receive a higher network-risk score.
    """

    def __init__(self, max_group_size: int = 30, min_network_size: int = 3) -> None:
        self.max_group_size = max_group_size
        self.min_network_size = min_network_size

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
        graph = nx.Graph()
        graph.add_nodes_from(data.index.tolist())
        for column, (relation, weight) in ENTITY_COLUMNS.items():
            if column not in data.columns:
                continue
            groups = data.groupby(column, dropna=True).groups
            for entity, indices in groups.items():
                members = list(indices)
                if len(members) <= 1 or len(members) > self.max_group_size:
                    continue
                for left_index, left in enumerate(members):
                    for right in members[left_index + 1 :]:
                        if graph.has_edge(left, right):
                            graph[left][right]["weight"] += weight
                            graph[left][right]["relations"].add(relation)
                        else:
                            graph.add_edge(left, right, weight=weight, relations={relation})

        degrees = dict(graph.degree(weight="weight"))
        clustering = nx.clustering(graph, weight="weight") if graph.number_of_edges() else {}
        relation_types: dict[int, set[str]] = defaultdict(set)
        for left, right, attributes in graph.edges(data=True):
            relation_types[left].update(attributes.get("relations", set()))
            relation_types[right].update(attributes.get("relations", set()))

        components = self._communities(graph)
        scored_components = [
            (component, self._component_score(graph, component, clustering, relation_types))
            for component in components
        ]
        score_values = [score for _, score in scored_components]
        baseline = float(np.percentile(score_values, 60)) if score_values else 0.0
        networks = []
        row_network: dict[int, dict[str, Any]] = {}
        for network_index, (component, score) in enumerate(sorted(scored_components, key=lambda item: item[1], reverse=True), start=1):
            network_id = f"NET-{network_index:03d}"
            members = []
            for row_index in sorted(component):
                row = data.loc[row_index]
                member = {
                    "row_index": int(row_index),
                    "declaration_id": self._safe(row.get("Declaration ID", row_index)),
                    "declarant_id": self._safe(row.get("Declarant ID")),
                    "network_risk": round(self._member_score(row_index, score, degrees, clustering, relation_types, baseline), 6),
                }
                members.append(member)
                row_network[row_index] = {"network_id": network_id, "network_risk": member["network_risk"]}
            networks.append({
                "network_id": network_id,
                "network_risk": round(score, 6),
                "member_count": len(members),
                "members": members,
                "relations": sorted({relation for row_index in component for relation in relation_types[row_index]}),
                "cohesion": round(self._community_cohesion(graph, component), 4),
                "detection_method": "weighted_louvain",
                "edges": [
                    {"source": int(left), "target": int(right), "weight": round(float(attributes.get("weight", 1)), 2), "relations": sorted(attributes.get("relations", set()))}
                    for left, right, attributes in graph.subgraph(component).edges(data=True)
                ],
                "estimated_value": round(float(data.loc[list(component), "Item Price"].fillna(0).sum()), 2) if "Item Price" in data else 0.0,
            })

        edges = [
            {"source": int(left), "target": int(right), "weight": round(float(attributes.get("weight", 1)), 2), "relations": sorted(attributes.get("relations", set()))}
            for left, right, attributes in graph.edges(data=True)
            if left in row_network and right in row_network
        ]
        nodes = [
            {
                "row_index": int(row_index),
                "declaration_id": self._safe(row.get("Declaration ID", row_index)),
                "label": self._safe(row.get("Declarant ID", row.get("Importer ID", row_index))),
                "network_risk": row_network.get(row_index, {}).get("network_risk", 0.0),
                "network_id": row_network.get(row_index, {}).get("network_id"),
            }
            for row_index, row in data.iterrows()
            if row_index in row_network
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "networks": networks,
            "row_network": row_network,
            "summary": {
                "networks_detected": len(networks),
                "transactions_linked": len(row_network),
                "graph_nodes": len(data),
                "graph_edges": graph.number_of_edges(),
                "network_status": "community_graph_ready",
                "detection_method": "weighted_louvain_multi_entity",
            },
        }

    def _communities(self, graph: nx.Graph) -> list[set[int]]:
        if graph.number_of_edges() == 0:
            return []
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=1.15,
            seed=42,
        )
        return [community for community in communities if len(community) >= self.min_network_size]

    @staticmethod
    def _community_cohesion(graph: nx.Graph, community: set[int]) -> float:
        if len(community) < 2:
            return 0.0
        subgraph = graph.subgraph(community)
        possible_edges = len(community) * (len(community) - 1) / 2
        density = subgraph.number_of_edges() / possible_edges
        weighted_strength = sum(attributes.get("weight", 1.0) for _, _, attributes in subgraph.edges(data=True))
        max_strength = possible_edges * max(weight for _, weight in ENTITY_COLUMNS.values())
        return float(min(1.0, 0.6 * density + 0.4 * (weighted_strength / max_strength if max_strength else 0.0)))

    @classmethod
    def _component_score(cls, graph: nx.Graph, component: set[int], clustering: dict[int, float], relation_types: dict[int, set[str]]) -> float:
        size = len(component)
        density = cls._community_cohesion(graph, component)
        bridge_score = np.mean([len(relation_types[node]) / len(ENTITY_COLUMNS) for node in component])
        cluster_score = np.mean([clustering.get(node, 0.0) for node in component])
        return float(min(1.0, 0.45 * density + 0.30 * bridge_score + 0.25 * cluster_score))

    @staticmethod
    def _member_score(row_index: int, component_score: float, degrees: dict[int, float], clustering: dict[int, float], relation_types: dict[int, set[str]], baseline: float) -> float:
        degree_signal = min(1.0, degrees.get(row_index, 0.0) / 8.0)
        role_signal = len(relation_types[row_index]) / len(ENTITY_COLUMNS)
        local_score = 0.55 * component_score + 0.25 * degree_signal + 0.20 * (0.5 * clustering.get(row_index, 0.0) + 0.5 * role_signal)
        return min(1.0, max(0.0, local_score + (0.08 if component_score >= baseline else 0.0)))

    @staticmethod
    def _safe(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value


network_service = NetworkService()
