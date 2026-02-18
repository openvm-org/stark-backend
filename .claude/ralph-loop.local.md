---
active: true
iteration: 1
max_iterations: 0
completion_promise: "DONEDONE"
started_at: "2026-02-18T00:21:01Z"
---

Implement the design in @docs/metal-backend/DESIGN.md  Never convert back to cpu types. Metal implementation must match cuda implementation. Run all tests in openvm-metal-backend directly, you are on a mac with metal gpu. Completion criteria:
- all tests pass
- ./scripts/check_metal_no_cpu_fallback.sh must not error
- use /codex to ask codex to review metal-backend against docs/metal-backend/DESIGN.md 
- codex review must have no unresolved findings

 print DONEDONE when completion criteria are met
