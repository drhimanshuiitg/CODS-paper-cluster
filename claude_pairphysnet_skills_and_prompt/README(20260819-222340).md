# Claude Code Skills Package for the Paired Device / Physiology-Guided Sleep-Apnea Study

This package contains:

- `CLAUDE_CODE_MASTER_PROMPT.md` — paste this into Claude Code to refactor the current project.
- `.claude/skills/paired-device-experiments/SKILL.md`
- `.claude/skills/gpu-only-4way/SKILL.md`
- `.claude/skills/results-audit/SKILL.md`
- `.claude/skills/publication-figures/SKILL.md`
- `.claude/skills/journal-paper/SKILL.md`

## Install in your project

From the project root, copy the `.claude` directory from this package into the project root so the final structure is:

```text
<project-root>/
└── .claude/
    └── skills/
        ├── paired-device-experiments/
        │   └── SKILL.md
        ├── gpu-only-4way/
        │   └── SKILL.md
        ├── results-audit/
        │   └── SKILL.md
        ├── publication-figures/
        │   └── SKILL.md
        └── journal-paper/
            └── SKILL.md
```

Then start Claude Code from the project root.

## Suggested order

### 1. Start the refactor
Paste the complete contents of `CLAUDE_CODE_MASTER_PROMPT.md`.

### 2. During experiment execution
Use:
```text
/paired-device-experiments
```

For GPU scheduling:
```text
/gpu-only-4way
```

### 3. After all computation is complete
Use:
```text
/results-audit
```

Do not proceed if it returns FAIL.

### 4. Generate final figures
Use:
```text
/publication-figures
```

### 5. Generate the full paper + PDF
Use:
```text
/journal-paper
```

The journal-paper skill is configured for explicit user invocation rather than automatic invocation.

## Important GPU note
The GPU skill forbids silent CPU fallback for numerical/model computation. Operating-system orchestration, process launch, metadata parsing, and disk I/O inevitably involve CPU, so the skill treats those as infrastructure rather than research computation.
