# Golden Datasets

Golden datasets provide deterministic semantic checks that run after the
deterministic eval report exists. They do not call an LLM and they do not
replace deterministic scoring.

Cases can reference an entry:

```yaml
golden_ref: fun_smoke.yaml#SMOKE01
```

or define checks inline:

```yaml
golden:
  acceptable_winners:
    - Arrival
  expected_status:
    - resolved
```

Supported fields:

```yaml
acceptable_winners:
unacceptable_winners:
required_final_output_terms:
forbidden_final_output_terms:
required_behaviors:
forbidden_behaviors:
expected_status:
allow_unresolved_tie:
```

Matching is case-insensitive and punctuation-tolerant. Exact wording is not
required; short phrase containment and token overlap are both accepted.
