---
name: diskbridge-planning
description: Plan a non-trivial DiskBridge code change before implementation. Use when the user asks to plan, implement, refactor, add physics, change APIs, change config, add tests, add validations, or create a markdown project plan.
---

# DiskBridge Planning Skill

Use this skill before implementing non-trivial DiskBridge changes.

A non-trivial change includes any change that touches multiple files; changes public APIs, configuration, physics behavior, validation behavior, or file outputs; affects performance-sensitive code; or modifies chemistry, shielding, RADMC-3D, dust, gas temperature, masks, or line-transfer workflows.

## Main Rule

Before editing code, create or update a markdown plan in `projects/active/`. Do not put implementation plans inside `.agents/skills/`.

Use this filename form:

```text
projects/active/YYYY-MM-DD-short-descriptive-name.md
```

For a new plan, start from `.agents/skills/diskbridge-planning/assets/plan_template.md` or `projects/templates/implementation_plan_template.md`.

## Planning Process

1. Identify the scientific or code-development goal.
2. Identify affected modules.
3. Separate structural changes from physics changes.
4. Identify public APIs, config keys, file outputs, and array shapes that must remain stable.
5. Define implementation steps.
6. Define fast pytest tests, including simple objective physics invariants when appropriate.
7. Define validation work separately from tests when checks are slower, broader, data-heavy, or need human interpretation.
8. Define documentation updates.
9. Define risks and human decisions.
10. Define completion criteria.

## Required Sections

Every plan must include:

- Status
- Goal
- Motivation
- Affected code areas
- Current behavior
- Proposed behavior
- Implementation steps
- Tests
- Validations
- Documentation updates
- Release-note impact
- Risks and human decisions
- Completion criteria
- Progress log

## Rules

- Do not use the plan to document stable user-facing behavior. Use `docs/` for that.
- Do not hide subjective physics validation inside pytest.
- Do not move simple objective physics invariants out of pytest only because they are physics-based.
- If a plan replaces another plan, mark the old plan as superseded and move it to `projects/superseded/`.
- If the task is completed, move the plan to `projects/completed/` and update `projects/index.md`.
- Keep the plan concise and current. Remove stale details instead of accumulating contradictory history.

## Output

When planning, produce or update the markdown plan and then summarize the plan file path, proposed scope, main risks, tests needed, validations needed, and whether implementation can start.
