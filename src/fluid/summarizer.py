from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from rdflib import Graph, Literal, URIRef

from src.data.rdf_loader import iter_entities, term_to_str


@dataclass
class FluidSummary:
    supernode_map: dict[str, int]
    supernode_members: dict[int, list[str]]
    summary_graph: dict[str, Any]
    stats: dict[str, Any]


def _entity_signature(graph: Graph, entity: str, k_hop: int = 1) -> tuple:
    entity_ref = URIRef(entity)
    types = sorted({term_to_str(o) for _, _, o in graph.triples((entity_ref, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), None))})
    out_edges = Counter()
    in_edges = Counter()
    for _, predicate, obj in graph.triples((entity_ref, None, None)):
        if isinstance(obj, Literal):
            continue
        out_edges[term_to_str(predicate)] += 1
    for subject, predicate, _ in graph.triples((None, None, entity_ref)):
        if isinstance(subject, Literal):
            continue
        in_edges[term_to_str(predicate)] += 1
    out_sig = tuple(sorted(out_edges.items()))
    in_sig = tuple(sorted(in_edges.items()))
    if k_hop <= 1:
        return (tuple(types), out_sig, in_sig)
    return (tuple(types), out_sig, in_sig, k_hop)


def partition_supernodes(graph: Graph, k_hop: int = 1) -> tuple[dict[str, int], dict[int, list[str]]]:
    signature_to_id: dict[tuple, int] = {}
    supernode_map: dict[str, int] = {}
    supernode_members: dict[int, list[str]] = defaultdict(list)

    for entity in iter_entities(graph):
        signature = _entity_signature(graph, entity, k_hop=k_hop)
        if signature not in signature_to_id:
            signature_to_id[signature] = len(signature_to_id)
        sid = signature_to_id[signature]
        supernode_map[entity] = sid
        supernode_members[sid].append(entity)
    return supernode_map, dict(supernode_members)


def _collect_vocab(graph: Graph) -> tuple[list[str], list[str]]:
    type_vocab: set[str] = set()
    relation_vocab: set[str] = set()
    rdf_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    for subject, predicate, obj in graph:
        predicate_str = term_to_str(predicate)
        relation_vocab.add(predicate_str)
        if predicate == rdf_type and not isinstance(obj, Literal):
            type_vocab.add(term_to_str(obj))
    return sorted(type_vocab), sorted(relation_vocab)


def build_summary_graph(
    graph: Graph,
    supernode_map: dict[str, int],
    supernode_members: dict[int, list[str]],
) -> dict[str, Any]:
    type_vocab, relation_vocab = _collect_vocab(graph)
    type_index = {name: idx for idx, name in enumerate(type_vocab)}
    relation_index = {name: idx for idx, name in enumerate(relation_vocab)}

    num_supernodes = len(supernode_members)
    feature_dim = 3 + len(type_vocab)
    x = torch.zeros((num_supernodes, feature_dim), dtype=torch.float32)

    edge_counter: Counter[tuple[int, int, int]] = Counter()
    internal_edges = 0
    external_edges = 0

    for sid, members in supernode_members.items():
        member_count = len(members)
        out_degrees: list[int] = []
        in_degrees: list[int] = []
        type_counts = torch.zeros(len(type_vocab), dtype=torch.float32)
        for entity in members:
            entity_ref = URIRef(entity)
            out_deg = len(list(graph.triples((entity_ref, None, None))))
            in_deg = len(list(graph.triples((None, None, entity_ref))))
            out_degrees.append(out_deg)
            in_degrees.append(in_deg)
            for _, predicate, obj in graph.triples((entity_ref, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), None)):
                if isinstance(obj, Literal):
                    continue
                type_name = term_to_str(obj)
                if type_name in type_index:
                    type_counts[type_index[type_name]] += 1.0
        x[sid, 0] = float(member_count)
        x[sid, 1] = float(sum(out_degrees) / max(len(out_degrees), 1))
        x[sid, 2] = float(sum(in_degrees) / max(len(in_degrees), 1))
        if type_counts.sum() > 0:
            x[sid, 3:] = type_counts / type_counts.sum()

    numeric = torch.log1p(x[:, :3])
    max_vals = numeric.max(dim=0).values.clamp(min=1.0)
    x[:, :3] = numeric / max_vals

    for subject, predicate, obj in graph:
        if isinstance(subject, Literal) or isinstance(obj, Literal):
            continue
        subj = term_to_str(subject)
        obj_str = term_to_str(obj)
        if subj not in supernode_map or obj_str not in supernode_map:
            continue
        src = supernode_map[subj]
        dst = supernode_map[obj_str]
        rel_idx = relation_index[term_to_str(predicate)]
        if src == dst:
            internal_edges += 1
            continue
        external_edges += 1
        edge_counter[(src, dst, rel_idx)] += 1

    edge_index = []
    edge_attr = []
    for (src, dst, rel_idx), weight in edge_counter.items():
        edge_index.append([src, dst])
        edge_attr.append([rel_idx, float(weight)])

    if edge_index:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.zeros((0, 2), dtype=torch.float32)

    return {
        "x": x,
        "edge_index": edge_index_tensor,
        "edge_attr": edge_attr_tensor,
        "num_nodes": num_supernodes,
        "type_vocab": type_vocab,
        "relation_vocab": relation_vocab,
        "feature_dim": feature_dim,
        "num_relations": len(relation_vocab),
    }


def summarize_graph(graph: Graph, k_hop: int = 1) -> FluidSummary:
    supernode_map, supernode_members = partition_supernodes(graph, k_hop=k_hop)
    summary_graph = build_summary_graph(graph, supernode_map, supernode_members)
    stats = {
        "num_entities": len(supernode_map),
        "num_supernodes": len(supernode_members),
        "compression_ratio": len(supernode_map) / max(len(supernode_members), 1),
        "num_summary_edges": summary_graph["edge_index"].shape[1],
    }
    return FluidSummary(
        supernode_map=supernode_map,
        supernode_members=supernode_members,
        summary_graph=summary_graph,
        stats=stats,
    )
