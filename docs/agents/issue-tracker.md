# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`.
- The spec is `.scratch/<feature-slug>/spec.md`.
- Implementation issues are one file per ticket:
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`.
  Number tickets from `01`; keep each ticket in its own file.
- Record triage state as a `Status:` line near the top of each issue.
  Use the role strings in `docs/agents/triage-labels.md`.
- Append comments and conversation history under `## Comments`.

## When a skill says "publish to the issue tracker"

Create the spec or ticket file at its conventional path above,
creating directories as needed.

## When a skill says "fetch the relevant ticket"

Read the referenced file. Resolve a ticket number within the relevant
feature's `issues/` directory; ask for the feature if ambiguous.

## Wayfinding operations

Used by `/wayfinder`. The map is a file with one child file per ticket.

- Map: `.scratch/<effort>/map.md`, containing Notes,
  Decisions-so-far, and Fog.
- Child ticket: `.scratch/<effort>/issues/<NN>-<slug>.md`,
  numbered from `01`, with the question in the body.
- Record ticket type in `Type:`:
  `research`, `prototype`, `grilling`, or `task`.
- Wayfinding tickets use `Status: open`, `Status: claimed`,
  or `Status: resolved` for their execution lifecycle.
  Record any triage role separately as `Triage:`.
- Blocking: record `Blocked by: NN, NN` near the top.
  A ticket is unblocked when every listed ticket is resolved.
- Frontier: select the first open, unblocked ticket by number.
- Claim: set `Status: claimed` and save before starting work.
- Resolve: append the answer under `## Answer`, set
  `Status: resolved`, and append a summary plus ticket link
  to Decisions-so-far in the map.
