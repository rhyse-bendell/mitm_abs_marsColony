# Mars Colony Canonical Task Truth

- Runtime canon for Mars Colony lives in `config/tasks/mars_colony/*`.
- External prose docs can drift; update docs to match config (not vice versa).
- Canonical phase populations:
  - `phase1`: `{"civilians": 50, "VIPs": 0}`
  - `phase2`: `{"civilians": 40, "VIPs": 20}`
- Canonical Build/Table C access unlock: `bridge_to_zone_C` (configured in `phase2.unlocks`).
- Rule references inside task config should use canonical `R_*` IDs (legacy aliases remain loader-compatible only).
