# Mars Colony Canonical Task Truth

- Runtime canon for Mars Colony lives in `config/tasks/mars_colony/*`.
- External prose docs can drift; update docs to match config (not vice versa).
- Canonical phase populations:
  - `phase1`: `{"civilians": 50, "VIPs": 0}`
  - `phase2`: `{"civilians": 40, "VIPs": 20}`
- Canonical Site C construction/resource access unlock: runtime bridge state (`bridge_bc.status == "complete"`), with site-gating truth configured in `construction_parameters.json`.
- Rule references inside task config should use canonical `R_*` IDs (legacy aliases remain loader-compatible only).
- Construction sites are capacity-limited containers; they are not one-to-one structure templates.
- Site capacities are authoritative as configured (including `site_a_capacity = 0`).
- Construction projects are dynamically instantiated at runtime; mission startup does not auto-seed projects.
- `construction_templates.csv` defines structure archetypes (`house`, `greenhouse`, `water_generator`) that can be instantiated at any buildable site.
- Canonical build interaction targets are `Build_Site_A|B|C`; legacy `Build_Table_*` identifiers are deprecated compatibility aliases only.
