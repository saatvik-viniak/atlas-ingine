# IK2 monitor

The Monitor page from **Project Maya** (`serve/web/monitor.js`, `monitor.css`, `tokens.css`,
`components.css`, `app.css`, `preact.js`, `lib.js`, `state.js`, `server.js`, `ui.js`, `sprite.svg`,
fonts), copied verbatim where it still applies and cut where it does not, wired to Inkling-2-Mini
instead of GLM.

- Upstream: Project Maya, MIT License, Copyright (c) 2026 Peasant Smith and the Project Maya contributors.
- This copy: `dashboard/web/`. The only edits are listed under **Changes** below.

## What it shows

- **Throughput** — decode and prefill rates, the peak, the last minute.
- **Right now** — idle / writing, active slots, queue, reused cache.
- **Context** — prompt, reused, written, free (of `--max-ctx`).
- **Hardware** — GPUs 2 and 3: load, VRAM, power, temperature, PCIe; CPU, RAM, disk.
- **Recent requests** — time, status, prompt/reused/prefill/output/decode, took.

## Data sources (no engine changes)

| Panel | Source |
|---|---|
| live slots, queue | `GET http://127.0.0.1:8090/health` |
| live tokens (generated, max, started, first token) | `GET http://127.0.0.1:8090/stats` (new in this PR) |
| finished requests | `journalctl --user -u ik2-serve.service -f -o cat` (the engine's `prefill=` and `completion=` lines) |
| GPU load / VRAM / power / temp / PCIe | `nvidia-smi` for GPUs 2 and 3 |
| CPU / RAM / disk | `/proc/stat`, `/proc/meminfo`, `/proc/diskstats` |

## Changes from Maya's code

- `app.js` renders `MonitorPage` only (no chat, sidebar, models, settings).
- `chats.js` / `sidebar.js` are cut to the exports the Monitor uses (`newChat`, `setSide`, `status`).
- `tokens.css` gains an `html[data-scheme="ik2"]` block: deep green-black page, teal accent, amber warnings.
- `monitor.js`: the subtitle and the Context note mention IK2; the VRAM-hit stat (not exposed by
  ingine-serve) is removed. With `/stats` the Decode figure is the live aggregate rate while the
  engine writes; without it, the last finished request's rate is shown and labelled **last request**.
- `index.html` is the dashboard's own shell; the theme is dark.

## Run

```bash
dashboard/run.sh                 # 0.0.0.0:8080, engine on 127.0.0.1:8090
```

Systemd user unit: `ik2-dashboard.service` (enabled; `Conflicts=maya-glm.service strata-qwen.service`).
Same URL on loopback and the tailnet: `http://127.0.0.1:8080/` and `http://100.127.90.18:8080/`.
