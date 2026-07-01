from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

UNIV = Namespace("http://swat.cse.lehigh.edu/onto/univ-bench.owl#")
BASE = Namespace("http://www.lehigh.edu/~mhp22/tesis/Exp4/")

RULE_FAMILIES = [
    "equality",
    "property_assertion",
    "property_characteristics",
    "property_axioms",
    "class_assertion",
    "class_axioms",
    "datatype",
    "schema_vocabulary",
]

RULE_TO_FAMILY: dict[str, str] = {
    "eq-ref": "equality",
    "eq-sym": "equality",
    "eq-trans": "equality",
    "eq-rep-s": "equality",
    "eq-rep-p": "equality",
    "eq-rep-o": "equality",
    "prp-ap": "property_assertion",
    "prp-dom": "property_assertion",
    "prp-rng": "property_assertion",
    "prp-fp": "property_characteristics",
    "prp-ifp": "property_characteristics",
    "prp-irp": "property_characteristics",
    "prp-symp": "property_characteristics",
    "prp-asyp": "property_characteristics",
    "prp-trp": "property_characteristics",
    "prp-spo1": "property_axioms",
    "prp-spo2": "property_axioms",
    "prp-eqp1": "property_axioms",
    "prp-eqp2": "property_axioms",
    "prp-pdw": "property_axioms",
    "prp-inv1": "property_axioms",
    "prp-inv2": "property_axioms",
    "cax-sco": "class_axioms",
    "cax-ec": "class_axioms",
    "cax-owl2rl": "class_axioms",
    "cls-hv1": "class_axioms",
    "cls-hv2": "class_axioms",
    "cls-svf1": "class_axioms",
    "cls-svf2": "class_axioms",
    "cls-svf3": "class_axioms",
    "cls-svf4": "class_axioms",
    "cls-com": "class_axioms",
    "cls-avf": "class_axioms",
    "cls-maxc2": "class_axioms",
    "cls-maxdc2": "class_axioms",
    "cls-nnf1": "class_axioms",
    "cls-nnf2": "class_axioms",
    "dt-type1": "datatype",
    "dt-type2": "datatype",
    "dt-type3": "datatype",
    "dt-eq": "datatype",
    "dt-diff": "datatype",
    "dt-not-type": "datatype",
    "scm-cls": "schema_vocabulary",
    "scm-sch": "schema_vocabulary",
    "scm-sp": "schema_vocabulary",
    "scm-dp": "schema_vocabulary",
    "scm-op": "schema_vocabulary",
    "scm-rp": "schema_vocabulary",
    "scm-ap": "schema_vocabulary",
}


def family_to_index(family: str) -> int:
    return RULE_FAMILIES.index(family)


def rule_to_family(rule_id: str) -> str:
    return RULE_TO_FAMILY.get(rule_id, "schema_vocabulary")


@dataclass
class TripleRecord:
    subject: str
    predicate: str
    object: str


def term_to_str(term) -> str:
    if isinstance(term, Literal):
        return str(term)
    if isinstance(term, (URIRef, BNode)):
        return str(term)
    return str(term)


def load_graph(path: str | Path) -> Graph:
    graph = Graph()
    graph.parse(str(path))
    return graph


def merge_graphs(*graphs: Graph) -> Graph:
    merged = Graph()
    for graph in graphs:
        for triple in graph:
            merged.add(triple)
    return merged


def graph_to_triple_records(graph: Graph) -> list[TripleRecord]:
    records: list[TripleRecord] = []
    for subject, predicate, obj in graph:
        records.append(
            TripleRecord(
                subject=term_to_str(subject),
                predicate=term_to_str(predicate),
                object=term_to_str(obj),
            )
        )
    return records


def iter_entities(graph: Graph) -> Iterable[str]:
    seen: set[str] = set()
    for subject, _, obj in graph:
        for term in (subject, obj):
            if isinstance(term, Literal):
                continue
            key = term_to_str(term)
            if key not in seen:
                seen.add(key)
                yield key


def build_univ_bench_ontology() -> Graph:
    """Minimal OWL2 RL-compatible LUBM ontology (univ-bench subset)."""
    g = Graph()
    g.bind("univ", UNIV)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)

    classes = [
        "University",
        "Department",
        "FullProfessor",
        "AssociateProfessor",
        "AssistantProfessor",
        "Professor",
        "GraduateStudent",
        "UndergraduateStudent",
        "Student",
        "Person",
        "Publication",
        "Research",
        "Course",
    ]
    for cls in classes:
        uri = UNIV[cls]
        g.add((uri, RDF.type, OWL.Class))
        g.add((uri, RDFS.label, Literal(cls)))

    g.add((UNIV.Professor, RDFS.subClassOf, UNIV.Person))
    g.add((UNIV.FullProfessor, RDFS.subClassOf, UNIV.Professor))
    g.add((UNIV.AssociateProfessor, RDFS.subClassOf, UNIV.Professor))
    g.add((UNIV.AssistantProfessor, RDFS.subClassOf, UNIV.Professor))
    g.add((UNIV.Student, RDFS.subClassOf, UNIV.Person))
    g.add((UNIV.GraduateStudent, RDFS.subClassOf, UNIV.Student))
    g.add((UNIV.UndergraduateStudent, RDFS.subClassOf, UNIV.Student))

    properties = [
        "name",
        "emailAddress",
        "telephone",
        "worksFor",
        "headOf",
        "memberOf",
        "member",
        "subOrganizationOf",
        "undergraduateDegreeFrom",
        "mastersDegreeFrom",
        "doctoralDegreeFrom",
        "advisor",
        "teachingAssistantOf",
        "takesCourse",
        "researchInterest",
        "publicationAuthor",
        "publication",
        "leadershipOf",
    ]
    for prop in properties:
        uri = UNIV[prop]
        g.add((uri, RDF.type, OWL.ObjectProperty if prop not in {"name", "emailAddress", "telephone", "researchInterest"} else OWL.DatatypeProperty))
        g.add((uri, RDFS.label, Literal(prop)))

    g.add((UNIV.worksFor, RDFS.domain, UNIV.Person))
    g.add((UNIV.worksFor, RDFS.range, UNIV.Department))
    g.add((UNIV.headOf, RDFS.domain, UNIV.Person))
    g.add((UNIV.headOf, RDFS.range, UNIV.Department))
    g.add((UNIV.memberOf, RDFS.domain, UNIV.Person))
    g.add((UNIV.memberOf, RDFS.range, UNIV.Department))
    g.add((UNIV.subOrganizationOf, RDFS.domain, UNIV.Department))
    g.add((UNIV.subOrganizationOf, RDFS.range, UNIV.University))
    g.add((UNIV.advisor, RDF.type, OWL.TransitiveProperty))
    g.add((UNIV.advisor, RDFS.domain, UNIV.Student))
    g.add((UNIV.advisor, RDFS.range, UNIV.Person))
    g.add((UNIV.teachingAssistantOf, RDFS.domain, UNIV.GraduateStudent))
    g.add((UNIV.teachingAssistantOf, RDFS.range, UNIV.Course))
    g.add((UNIV.takesCourse, RDFS.domain, UNIV.Student))
    g.add((UNIV.takesCourse, RDFS.range, UNIV.Course))
    return g


def generate_lubm_instance(university_id: int, scale: int = 2, seed: int = 42) -> Graph:
    """Generate a synthetic LUBM-like ABox with fixed ontology vocabulary."""
    import random

    rng = random.Random(seed + university_id)
    g = Graph()
    g.bind("univ", UNIV)
    g.bind("base", BASE)

    uni = BASE[f"University{university_id}"]
    g.add((uni, RDF.type, UNIV.University))
    g.add((uni, UNIV.name, Literal(f"University{university_id}")))

    departments: list[URIRef] = []
    for dept_idx in range(scale * 2):
        dept = BASE[f"University{university_id}-Department{dept_idx}"]
        g.add((dept, RDF.type, UNIV.Department))
        g.add((dept, UNIV.subOrganizationOf, uni))
        g.add((dept, UNIV.name, Literal(f"Department{dept_idx}")))
        departments.append(dept)

    professors: list[URIRef] = []
    professor_types = [UNIV.FullProfessor, UNIV.AssociateProfessor, UNIV.AssistantProfessor]
    for dept_idx, dept in enumerate(departments):
        for prof_idx in range(scale):
            prof_type = professor_types[(dept_idx + prof_idx) % len(professor_types)]
            prof = BASE[f"University{university_id}-Prof{dept_idx}-{prof_idx}"]
            g.add((prof, RDF.type, prof_type))
            g.add((prof, RDF.type, UNIV.Professor))
            g.add((prof, RDF.type, UNIV.Person))
            g.add((prof, UNIV.worksFor, dept))
            g.add((prof, UNIV.name, Literal(f"Professor{dept_idx}-{prof_idx}")))
            if prof_idx == 0:
                g.add((prof, UNIV.headOf, dept))
            professors.append(prof)

    grad_students: list[URIRef] = []
    for dept_idx, dept in enumerate(departments):
        for st_idx in range(scale * 2):
            student = BASE[f"University{university_id}-Grad{dept_idx}-{st_idx}"]
            g.add((student, RDF.type, UNIV.GraduateStudent))
            g.add((student, RDF.type, UNIV.Student))
            g.add((student, RDF.type, UNIV.Person))
            g.add((student, UNIV.memberOf, dept))
            g.add((student, UNIV.name, Literal(f"GradStudent{dept_idx}-{st_idx}")))
            advisor = professors[(dept_idx + st_idx) % len(professors)]
            g.add((student, UNIV.advisor, advisor))
            grad_students.append(student)

    undergrads: list[URIRef] = []
    for dept_idx, dept in enumerate(departments):
        for st_idx in range(scale):
            student = BASE[f"University{university_id}-Under{dept_idx}-{st_idx}"]
            g.add((student, RDF.type, UNIV.UndergraduateStudent))
            g.add((student, RDF.type, UNIV.Student))
            g.add((student, RDF.type, UNIV.Person))
            g.add((student, UNIV.memberOf, dept))
            g.add((student, UNIV.name, Literal(f"Undergrad{dept_idx}-{st_idx}")))
            undergrads.append(student)

    courses: list[URIRef] = []
    for dept_idx, dept in enumerate(departments):
        for course_idx in range(scale):
            course = BASE[f"University{university_id}-Course{dept_idx}-{course_idx}"]
            g.add((course, RDF.type, UNIV.Course))
            g.add((course, UNIV.name, Literal(f"Course{dept_idx}-{course_idx}")))
            courses.append(course)
            ta = grad_students[(dept_idx + course_idx) % len(grad_students)]
            g.add((ta, UNIV.teachingAssistantOf, course))

    for idx, student in enumerate(undergrads + grad_students):
        course = courses[idx % len(courses)]
        g.add((student, UNIV.takesCourse, course))

    publications: list[URIRef] = []
    for pub_idx in range(scale * len(departments)):
        pub = BASE[f"University{university_id}-Publication{pub_idx}"]
        g.add((pub, RDF.type, UNIV.Publication))
        author = professors[pub_idx % len(professors)]
        g.add((author, UNIV.publicationAuthor, pub))
        g.add((pub, UNIV.publication, author))
        publications.append(pub)

    researches: list[URIRef] = []
    for res_idx in range(scale * len(departments)):
        research = BASE[f"University{university_id}-Research{res_idx}"]
        g.add((research, RDF.type, UNIV.Research))
        prof = professors[res_idx % len(professors)]
        g.add((prof, UNIV.researchInterest, Literal(f"Topic{res_idx % 5}")))
        researches.append(research)

    # Add graph-specific noise edges to differentiate topology while preserving ontology behavior.
    for idx in range(scale):
        left = professors[rng.randint(0, len(professors) - 1)]
        right = professors[rng.randint(0, len(professors) - 1)]
        if left != right:
            g.add((left, UNIV.leadershipOf, departments[idx % len(departments)]))

    return g


def prepare_lubm_dataset(
    raw_dir: str | Path,
    scale: int = 3,
    graph_scales: dict[str, int] | None = None,
    seed: int = 42,
) -> dict:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    default_scales = {"lubm1": scale, "lubm2": scale}
    scales = {**default_scales, **(graph_scales or {})}

    ontology = build_univ_bench_ontology()
    ontology_path = raw_path / "univ-bench.owl"
    ontology.serialize(destination=str(ontology_path), format="xml")

    manifest = {"ontology": str(ontology_path), "graphs": {}}
    for graph_name, university_id in [("lubm1", 1), ("lubm2", 2)]:
        graph_scale = scales.get(graph_name, scale)
        instance = generate_lubm_instance(university_id, scale=graph_scale, seed=seed)
        nt_path = raw_path / f"{graph_name}.nt"
        instance.serialize(destination=str(nt_path), format="nt")
        merged = merge_graphs(ontology, instance)
        merged_path = raw_path / f"{graph_name}_with_ontology.nt"
        merged.serialize(destination=str(merged_path), format="nt")
        manifest["graphs"][graph_name] = {
            "abox_path": str(nt_path),
            "merged_path": str(merged_path),
            "triple_count": len(instance),
            "merged_triple_count": len(merged),
            "university_id": university_id,
            "scale": graph_scale,
        }
    return manifest
