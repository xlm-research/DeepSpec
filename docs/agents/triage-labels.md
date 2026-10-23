# Triage Labels

Map each canonical triage role to the string recorded in local tickets.

| Canonical role | Local string | Meaning |
| --- | --- | --- |
| needs-triage | needs-triage | Maintainer needs to evaluate this issue |
| needs-info | needs-info | Waiting on reporter for more information |
| ready-for-agent | ready-for-agent | Fully specified, ready for an agent |
| ready-for-human | ready-for-human | Requires human implementation |
| wontfix | wontfix | Will not be actioned |

Use the corresponding local string whenever a skill mentions a role.

Ordinary tickets record it as `Status: <local string>`.
Wayfinding tickets record it as `Triage: <local string>` because
their `Status:` field tracks execution, as defined in
`docs/agents/issue-tracker.md`.

Edit the Local string column to change the vocabulary.
