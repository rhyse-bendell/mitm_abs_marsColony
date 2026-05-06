# Brain–Mecha–Environment Architecture

## Brain / Pilot
- Chooses actions.
- Owns pilot-specific decision policy and capability parameters.
- Examples: **Procedural Baseline Pilot**, LLM pilot, future bot brain.

## Agent Mecha
- Provides capacities and execution interface.
- Owns memory containers, DIK interfaces, communication mechanics, action translation, legality checks, and blocker reporting.
- Should not silently solve task strategy unless explicitly documented as scaffold mode.

## Environment
- Owns world truth, task rules, construction state, spatial state, validation truth, and outcomes.
