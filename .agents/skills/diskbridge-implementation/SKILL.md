---
name: diskbridge-implementation
description: Implement a new DiskBridge feature or extend an existing scientific workflow. Use when adding code, changing APIs, wiring config, adding physics options, changing outputs, or implementing an approved project plan.
---

# DiskBridge Implementation Skill

Use this skill when the user asks to add new functionality, implement a physics extension, add a workflow option, expose a new API, or wire together existing DiskBridge components.

## Goal

Implement the requested feature with the smallest clean design that fits DiskBridge's existing architecture.

## Before Coding

1. Identify whether the change affects physics, units, performance, file I/O, configuration, validation, or plotting/diagnostics.
2. Check existing modules before creating new ones.
3. Identify likely capability keywords and inspect existing tagged code with
   `python tools/diskbridge_agent/capability_map.py list --keyword <keyword>`.
4. Search the generated symbol index for proposed helper names and concepts
   with `python tools/diskbridge_agent/capability_map.py symbols --query <text>`.
5. Check `.agents/code_index/` and
   `diskbridge_code_map/UTILITY_CONSOLIDATION_GUIDE.md` for existing duplicate
   or duplication-prone helpers before adding private helpers.
6. Prefer extending tagged canonical/entrypoint code over creating a parallel
   implementation. If new discoverable code is added, add sparse
   `db-keywords` / `db-role` tags using the registry; use module-level tags for
   helper families and direct tags for reusable or duplication-prone helpers.
7. Avoid adding new dependencies unless the user explicitly approves.
8. If the task belongs to an active project, update the relevant file in `projects/` before editing code.

## Design Rules

- Prefer explicit, readable code over clever abstractions.
- Prefer one high-level callable function for a workflow.
- Keep low-level numerical kernels small.
- Keep physical assumptions close to the code that uses them.
- Do not duplicate existing config parsing.
- Do not add broad fallback behavior.
- Do not silently change defaults.
- Document new user-facing options in the relevant docstring or config comments.

## RADMC-3D I/O Invariant

When implementing RADMC-3D readers, writers, UV products, cached outputs, or
line-transfer staging, support only RADMC-3D binary files and one internal
file-backed RADMC-3D data representation. Do not add ASCII/text fallbacks,
broad extension discovery, or parallel data structures for the same quantity.
Keep units, shapes, axis order, and binary filename policy explicit and tested.

## Units And Arrays

- Public APIs may accept Pint quantities.
- Internal heavy numerical routines should use raw NumPy arrays and explicit CGS magnitudes.
- Be explicit about shape expectations.
- Avoid hidden transposes or implicit axis-order assumptions.

## Completion Criteria

A feature is complete when:

1. The implementation is wired into the intended public path.
2. A fast pytest test exists if the behavior is deterministic.
3. A validation case is proposed if the feature changes physical interpretation.
4. Relevant docs or project notes are updated.
5. The final response states what was tested and what still needs scientific validation.
