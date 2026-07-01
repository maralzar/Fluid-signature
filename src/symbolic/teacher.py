from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rdflib import Graph


@dataclass
class ReasoningEvent:
    rule_id: str
    rule_family: str
    depth: int
    subject: str
    predicate: str
    object: str
    trigger_entity: str
    branch_width: int = 1
    propagation_strength: float = 1.0
    support_count: int = 1
    semantic_constraint: float = 0.0


@dataclass
class TraceRecorder:
    events: list[ReasoningEvent] = field(default_factory=list)
    skipped_events: int = 0

    def record(
        self,
        *,
        rule_id: str,
        rule_family: str,
        depth: int,
        subject: str,
        predicate: str,
        object: str,
        trigger_entity: str,
        branch_width: int = 1,
        propagation_strength: float = 1.0,
        support_count: int = 1,
        semantic_constraint: float = 0.0,
    ) -> None:
        self.events.append(
            ReasoningEvent(
                rule_id=rule_id,
                rule_family=rule_family,
                depth=depth,
                subject=subject,
                predicate=predicate,
                object=object,
                trigger_entity=trigger_entity,
                branch_width=branch_width,
                propagation_strength=propagation_strength,
                support_count=support_count,
                semantic_constraint=semantic_constraint,
            )
        )


@dataclass(frozen=True)
class _EventMetadata:
    rule_id: str
    rule_family: str
    depth: int
    branch_width: int
    propagation_strength: float
    support_count: int
    semantic_constraint: float


def _infer_rule_id(predicate: str, inferred_type: str | None = None) -> tuple[str, str]:
    from src.data.rdf_loader import rule_to_family

    predicate_lower = predicate.lower()
    if "subclassof" in predicate_lower or "subpropertyof" in predicate_lower:
        rule_id = "cax-sco" if "subclassof" in predicate_lower else "prp-spo1"
    elif "type" in predicate_lower or inferred_type:
        rule_id = "cls-hv1"
    elif "inverseof" in predicate_lower:
        rule_id = "prp-inv1"
    elif "transitive" in predicate_lower:
        rule_id = "prp-trp"
    elif "domain" in predicate_lower or "range" in predicate_lower:
        rule_id = "prp-dom"
    elif "equivalent" in predicate_lower:
        rule_id = "prp-eqp1"
    elif "disjoint" in predicate_lower:
        rule_id = "prp-pdw"
    else:
        rule_id = "prp-ap"
    return rule_id, rule_to_family(rule_id)


def _triple_depth(
    triple: tuple,
    baseline: set[tuple],
    inferred_depths: dict[tuple, int],
) -> int | None:
    if triple in baseline:
        return 0
    return inferred_depths.get(triple)


def _available_triples(
    graph: Graph,
    pattern: tuple,
    baseline: set[tuple],
    inferred_depths: dict[tuple, int],
) -> list[tuple[tuple, int]]:
    triples = []
    for triple in graph.triples(pattern):
        depth = _triple_depth(triple, baseline, inferred_depths)
        if depth is not None:
            triples.append((triple, depth))
    return triples


def _metadata_from_supports(
    rule_id: str,
    support_depths: list[int],
    max_depth: int,
    *,
    semantic_constraint: float = 1.0,
) -> _EventMetadata | None:
    from src.data.rdf_loader import rule_to_family

    if not support_depths:
        return None
    branch_width = max(len(support_depths), 1)
    depth = min(max(support_depths) + 1, max_depth)
    propagation_strength = float(branch_width) / float(max(depth, 1))
    return _EventMetadata(
        rule_id=rule_id,
        rule_family=rule_to_family(rule_id),
        depth=depth,
        branch_width=branch_width,
        propagation_strength=propagation_strength,
        support_count=branch_width,
        semantic_constraint=semantic_constraint,
    )


def _choose_metadata(candidates: list[_EventMetadata]) -> _EventMetadata | None:
    if not candidates:
        return None
    priority = {
        "prp-dom": 0,
        "prp-rng": 1,
        "cax-sco": 2,
        "prp-trp": 3,
        "prp-spo1": 4,
        "prp-inv1": 5,
    }
    return min(candidates, key=lambda item: (item.depth, priority.get(item.rule_id, 99), -item.support_count))


def _classify_supported_event(
    triple: tuple,
    graph: Graph,
    baseline: set[tuple],
    inferred_depths: dict[tuple, int],
    max_depth: int,
) -> _EventMetadata | None:
    from rdflib.namespace import OWL, RDF, RDFS

    subject, predicate, obj = triple
    candidates: list[_EventMetadata] = []

    if predicate == RDF.type:
        domain_depths = [
            depth
            for (support, depth) in _available_triples(graph, (subject, None, None), baseline, inferred_depths)
            if (support[1], RDFS.domain, obj) in graph
        ]
        if domain_depths:
            candidate = _metadata_from_supports("prp-dom", domain_depths, max_depth)
            if candidate:
                candidates.append(candidate)

        range_depths = [
            depth
            for (support, depth) in _available_triples(graph, (None, None, subject), baseline, inferred_depths)
            if (support[1], RDFS.range, obj) in graph
        ]
        if range_depths:
            candidate = _metadata_from_supports("prp-rng", range_depths, max_depth)
            if candidate:
                candidates.append(candidate)

        subclass_depths = [
            depth
            for (support, depth) in _available_triples(graph, (subject, RDF.type, None), baseline, inferred_depths)
            if support[2] != obj and (support[2], RDFS.subClassOf, obj) in graph
        ]
        if subclass_depths:
            candidate = _metadata_from_supports("cax-sco", subclass_depths, max_depth)
            if candidate:
                candidates.append(candidate)

    transitive_depths = []
    if (predicate, RDF.type, OWL.TransitiveProperty) in graph:
        for left, left_depth in _available_triples(graph, (subject, predicate, None), baseline, inferred_depths):
            middle = left[2]
            for right, right_depth in _available_triples(graph, (middle, predicate, obj), baseline, inferred_depths):
                transitive_depths.extend([left_depth, right_depth])
    if transitive_depths:
        candidate = _metadata_from_supports("prp-trp", transitive_depths, max_depth)
        if candidate:
            candidates.append(candidate)

    subproperty_depths = [
        depth
        for (support, depth) in _available_triples(graph, (subject, None, obj), baseline, inferred_depths)
        if support[1] != predicate and (support[1], RDFS.subPropertyOf, predicate) in graph
    ]
    if subproperty_depths:
        candidate = _metadata_from_supports("prp-spo1", subproperty_depths, max_depth)
        if candidate:
            candidates.append(candidate)

    inverse_depths = [
        depth
        for (support, depth) in _available_triples(graph, (obj, None, subject), baseline, inferred_depths)
        if (support[1], OWL.inverseOf, predicate) in graph or (predicate, OWL.inverseOf, support[1]) in graph
    ]
    if inverse_depths:
        candidate = _metadata_from_supports("prp-inv1", inverse_depths, max_depth)
        if candidate:
            candidates.append(candidate)

    return _choose_metadata(candidates)


def _fallback_metadata(triple: tuple, max_depth: int) -> _EventMetadata:
    from src.data.rdf_loader import rule_to_family, term_to_str

    _, predicate, obj = triple
    rule_id, family = _infer_rule_id(term_to_str(predicate), inferred_type=term_to_str(obj))
    return _EventMetadata(
        rule_id=rule_id,
        rule_family=family or rule_to_family(rule_id),
        depth=max_depth,
        branch_width=1,
        propagation_strength=1.0 / float(max(max_depth, 1)),
        support_count=1,
        semantic_constraint=0.0,
    )


def _assign_inference_metadata(
    inferred: list[tuple],
    graph: Graph,
    baseline: set[tuple],
    max_depth: int,
) -> dict[tuple, _EventMetadata]:
    pending = set(inferred)
    inferred_depths: dict[tuple, int] = {}
    metadata: dict[tuple, _EventMetadata] = {}

    while pending:
        progressed = False
        for triple in sorted(pending, key=lambda item: tuple(str(part) for part in item)):
            event = _classify_supported_event(triple, graph, baseline, inferred_depths, max_depth)
            if event is None:
                continue
            metadata[triple] = event
            inferred_depths[triple] = event.depth
            pending.remove(triple)
            progressed = True
        if not progressed:
            break

    for triple in sorted(pending, key=lambda item: tuple(str(part) for part in item)):
        event = _fallback_metadata(triple, max_depth)
        metadata[triple] = event
        inferred_depths[triple] = event.depth

    return metadata


def run_instrumented_reasoning(
    graph: Graph,
    max_depth: int = 5,
    axiomatic_triples: bool = False,
) -> tuple[Graph, TraceRecorder]:
    """Run OWL2 RL materialization with reasoning-event tracing."""
    from owlrl import DeductiveClosure, OWLRL_Semantics
    from rdflib import Literal

    from src.data.rdf_loader import term_to_str

    recorder = TraceRecorder()
    working = Graph()
    for triple in graph:
        working.add(triple)

    baseline = set(working)
    closure = DeductiveClosure(OWLRL_Semantics, axiomatic_triples=axiomatic_triples, datatype_axioms=False)
    closure.expand(working)

    inferred = sorted((triple for triple in working if triple not in baseline), key=lambda item: tuple(str(part) for part in item))
    metadata = _assign_inference_metadata(inferred, working, baseline, max_depth)
    for subject, predicate, obj in inferred:
        event = metadata[(subject, predicate, obj)]
        subj_str = term_to_str(subject)
        pred_str = term_to_str(predicate)
        obj_str = term_to_str(obj)
        trigger = subj_str if not isinstance(subject, Literal) else obj_str
        recorder.record(
            rule_id=event.rule_id,
            rule_family=event.rule_family,
            depth=event.depth,
            subject=subj_str,
            predicate=pred_str,
            object=obj_str,
            trigger_entity=trigger,
            branch_width=event.branch_width,
            propagation_strength=event.propagation_strength,
            support_count=event.support_count,
            semantic_constraint=event.semantic_constraint,
        )
    return working, recorder


def events_to_dataframe(events: list[ReasoningEvent]):
    import pandas as pd

    rows = [
        {
            "rule_id": event.rule_id,
            "rule_family": event.rule_family,
            "depth": event.depth,
            "subject": event.subject,
            "predicate": event.predicate,
            "object": event.object,
            "trigger_entity": event.trigger_entity,
            "branch_width": event.branch_width,
            "propagation_strength": event.propagation_strength,
            "support_count": event.support_count,
            "semantic_constraint": event.semantic_constraint,
        }
        for event in events
    ]
    return pd.DataFrame(rows)


def assign_supernodes(events: list[ReasoningEvent], supernode_map: dict[str, int]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    skipped = 0
    for event in events:
        entity = event.trigger_entity
        if entity not in supernode_map:
            if event.subject in supernode_map:
                entity = event.subject
            elif event.object in supernode_map:
                entity = event.object
            else:
                skipped += 1
                continue
        row = {
            "rule_id": event.rule_id,
            "rule_family": event.rule_family,
            "depth": event.depth,
            "subject": event.subject,
            "predicate": event.predicate,
            "object": event.object,
            "trigger_entity": entity,
            "supernode_id": supernode_map[entity],
            "branch_width": event.branch_width,
            "propagation_strength": event.propagation_strength,
            "support_count": event.support_count,
            "semantic_constraint": event.semantic_constraint,
        }
        enriched.append(row)
    return enriched
