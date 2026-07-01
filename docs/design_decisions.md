# Design Decisions

## 1. Event-to-supernode mapping

Reasoning events are assigned to the **trigger entity** supernode. If the trigger is not in the entity map (literal/bnode), fall back to subject, then object. Unmapped events are skipped.

## 2. Rule-family bucketing

Eight families aligned with OWL 2 RL spec sections are defined in `src/data/rdf_loader.py` (`RULE_FAMILIES`, `RULE_TO_FAMILY`). Post-hoc rule inference from predicate patterns is used when owlrl does not expose per-rule callbacks.

## 3. Depth definition

Depth is approximated from supporting triples after OWL-RL closure, capped at `K=5`. Direct domain/range/subclass/subproperty/transitive supports produce shallow events; events that depend on previously inferred supports receive deeper depths. Triples without recognizable supports are still recorded with a conservative fallback depth.

## 4. FLUID equivalence (MVP)

Supernodes partition entities by `(rdf:type multiset, outgoing edge signature, incoming edge signature)` at 1-hop. This approximates FLUID type + property signature equivalence without the Java `fluid-framework`.

Numeric summary features use `log1p` plus per-graph max scaling for member count and mean in/out degree. This keeps the student from treating raw graph scale as semantic evidence during cross-graph transfer.

## 5. RBT vs triple materialization

Supervision targets are `[8, 5, 6]` behavioral tensors, not inferred triple sets. The six named channels are:

- `activation_frequency`
- `propagation_strength`
- `branching_factor`
- `rule_interaction`
- `semantic_constraint`
- `rule_centrality`

Transfer eval includes a triple-count proxy baseline to show behavior-based matching differs from materialization compression.

## 6. Transfer diagnostics

The student exports semantic/topological reasoning signatures for both source and target graphs without invoking the symbolic teacher at inference time. The transfer report includes source/target RBT reconstruction MSE plus matched semantic alignment, matched topology alignment, random-pair alignment, and bootstrap significance. `mvp_passed` is intentionally strict and may fail on harder cross-scale runs even when RBT reconstruction is low.
