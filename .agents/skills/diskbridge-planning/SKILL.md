---
name: diskbridge-planning
description: Plan a non-trivial DiskBridge code change before implementation. Use when the user asks to plan, implement, refactor, add physics, change APIs, change config, add tests, add validations, or create a markdown project plan.
---

# DiskBridge Planning Skill

Use this skill before implementing non-trivial DiskBridge changes.

A non-trivial change includes any change that touches multiple files; changes public APIs, configuration, physics behavior, validation behavior, or file outputs; affects performance-sensitive code; or modifies chemistry, shielding, RADMC-3D, dust, gas temperature, masks, or line-transfer workflows.

## Main Rule

Before editing code, create or update a markdown plan in `projects/active/`. Do not put implementation plans inside `.agents/skills/`.

DiskBridge is single-user development code. Plans must replace obsolete APIs,
config keys, output formats, and execution paths directly. Do not plan aliases,
deprecations, migrations, compatibility readers, or fallback branches unless
the user explicitly requests them for a specific interface.

Use this filename form:

```text
projects/active/YYYY-MM-DD-short-descriptive-name.md
```

For a new plan, start from `.agents/skills/diskbridge-planning/assets/plan_template.md` or `projects/templates/implementation_plan_template.md`.

## Planning Process

1. Identify the scientific or code-development goal.
2. Identify affected modules.
3. Identify likely capability keywords and inspect existing tagged code with
   `python tools/diskbridge_agent/capability_map.py list --keyword <keyword>`.
4. Search the generated symbol index for proposed function names and helper
   concepts with `python tools/diskbridge_agent/capability_map.py symbols --query <text>`.
5. Check `.agents/code_index/` and
   `diskbridge_code_map/UTILITY_CONSOLIDATION_GUIDE.md` for known duplicate or
   duplication-prone helpers before proposing new helpers.
6. State whether the change reuses, extends, merges, or adds code.
7. Separate structural changes from physics changes.
8. Identify the canonical public APIs, config keys, file outputs, and array
   shapes after the change. Do not assume superseded interfaces must remain
   accepted.
9. Define implementation steps.
10. Define fast pytest tests, including simple objective physics invariants when appropriate.
11. Classify evidence work before choosing a directory. Use
    `validation/<case>/` only when the plan states (a) the scientific claim or
    code behavior being tested, (b) its reference, benchmark, invariant,
    controlled comparison, or explicit expected behavior, and (c) what result
    supports or contradicts it. Runtime, data volume, plots, reports, and human
    interpretation are not sufficient. Put descriptive measurements,
    inventories, snapshot atlases, and exploratory plots beside the owning
    example or project, normally `examples/<workflow>/plots/<analysis>/`.
    Keep fast objective invariants in pytest.
12. Define documentation updates.
13. Define risks and human decisions.
14. Define completion criteria.

## RADMC-3D I/O Invariant

For any plan that touches RADMC-3D I/O, data loading, data writing, cached
RADMC-3D outputs, UV products, or line-transfer staging, require one supported
binary-only RADMC-3D data representation. Do not plan broad ASCII/text fallback
discovery or parallel data structures for the same RADMC-3D quantity. Plans
must state the canonical binary file names, units, shapes, and axis order after
the change.

## Required Sections

Every plan must include:

- Status
- Goal
- Motivation
- Affected code areas
- Current behavior
- Proposed behavior
- Capability map review
- Implementation steps
- Tests
- Validations
- Documentation updates
- Release-note impact
- Risks and human decisions
- Completion criteria
- Progress log

## Capability Map Review Requirements

The plan's capability map review must state:

- relevant keywords;
- existing tagged code inspected;
- generated symbol-index searches and what they found;
- utility-consolidation guide entries inspected, when relevant;
- reuse, extend, merge, or add decision;
- new or updated source tags planned;
- utility/helper module impact.

## Rules

- Do not use the plan to document stable user-facing behavior. Use `docs/` for that.
- Do not hide subjective physics validation inside pytest.
- Do not move simple objective physics invariants out of pytest only because they are physics-based.
- Do not place descriptive or exploratory analysis in `validation/` without a
  tested claim, an evidentiary expectation, and a stated interpretation rule.
- Do not create `tests/validation/`. Use `validation/<case>/` for validation
  workflows. Do not add pytest files for validation-specific helpers.
- If a plan replaces another plan, mark the old plan as superseded and move it to `projects/superseded/`.
- If the task is completed, move the plan to `projects/completed/` and update `projects/index.md`.
- Keep the plan concise and current. Remove stale details instead of accumulating contradictory history.

## Output

When planning, produce or update the markdown plan and then summarize the plan file path, proposed scope, main risks, tests needed, validations needed, and whether implementation can start.
