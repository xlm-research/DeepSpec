# Domain Docs

This repo uses a single-context domain documentation layout.

## Before exploring

- Read `GLOSSARY.md` at the repo root.
- Read ADRs in `docs/adr/` that touch the area being explored.

If either is absent, proceed silently.
The `/domain-modeling` skill creates documentation lazily when
terms or decisions are resolved.

## File structure

- `GLOSSARY.md`: shared domain vocabulary.
- `docs/adr/`: architectural decisions.

## Use the glossary's vocabulary

Use the glossary's defined terms in issue titles, refactor proposals,
hypotheses, and test names.

If a needed concept is absent, reconsider whether it matches the
project's language. Record real vocabulary gaps for `/domain-modeling`.

## Flag ADR conflicts

Explicitly identify any existing ADR that a proposal contradicts
and explain why the decision should be reopened.
