from rdflib import Graph, Namespace
from rdflib.namespace import OWL, RDF, RDFS

from src.symbolic.teacher import run_instrumented_reasoning


def test_teacher_records_supported_owlrl_events():
    ex = Namespace("http://example.org/")
    graph = Graph()
    graph.add((ex.Student, RDF.type, OWL.Class))
    graph.add((ex.Person, RDF.type, OWL.Class))
    graph.add((ex.Professor, RDF.type, OWL.Class))
    graph.add((ex.Course, RDF.type, OWL.Class))
    graph.add((ex.Student, RDFS.subClassOf, ex.Person))
    graph.add((ex.Professor, RDFS.subClassOf, ex.Person))
    graph.add((ex.teaches, RDF.type, OWL.ObjectProperty))
    graph.add((ex.teaches, RDFS.domain, ex.Professor))
    graph.add((ex.teaches, RDFS.range, ex.Course))
    graph.add((ex.alice, RDF.type, ex.Student))
    graph.add((ex.bob, ex.teaches, ex.cs101))

    _, recorder = run_instrumented_reasoning(graph, max_depth=4)
    rule_ids = {event.rule_id for event in recorder.events}

    assert "cax-sco" in rule_ids
    assert "prp-dom" in rule_ids
    assert "prp-rng" in rule_ids
    assert all(1 <= event.depth <= 4 for event in recorder.events)
    assert any(event.semantic_constraint > 0.0 for event in recorder.events)
