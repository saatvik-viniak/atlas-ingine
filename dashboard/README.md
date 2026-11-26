# IK2 monitor

The **Strata** dashboard UI — the `serve/web` bundle Project Maya ships (Strata is the app; Project Maya is the
build this box ran) — cut to its Monitor page and wired to Inkling-2-Mini. Same layout, components and
Interactions; IK2's own palette.

- Source: Project Maya's `serve/web` (`monitor.js`, `monitor.css`, `tokens.css`, `components.css`, `app.css`,
  `preact.js`, `lib.js`, `state.js`, `server.js`, `ui.js`, `sprite.svg`, fonts), MIT License,
  Copyright (c) 2026 Peasant Smith and the Project Maya contributors.
- This copy: `dashboard/web/`. The edits are listed under **Changes** below.

## What it shows

- **Throughput** — live decode (aggregate while several answers run), prefill, the peak, the last minute.
- **Right now** — reading / writing, tokens generated of the request's max, active slots, queue, reused cache.
- **Context** — prompt, reused, written, free (of `--max-ctx`).
- **Hardware** — GPUs 2 and 3: load, VRAM, power, temperature, PCIe; CPU, RAM, disk.
- **Electricity** — the machine's cost tracker: watts now (per GPU, CPU, board), live ComEd price, today,
  this month, all time, and the last 14 days.
- **Recent requests** — time, status, prompt/reused/prefill/output/decode, took.

## Data sources

| Panel | Source |
|---|---|
| live slots, queue | `GET http://127.0.0.1:8090/health` |
| live tokens (generated, max, started, first token) | `GET http://127.0.0.1:8090/stats` (new in this PR) |
| finished requests | `journalctl --user -u ik2-serve.service -f -o cat` (the engine's `prefill=` / `completion=` lines) |
| GPU load / VRAM / power / temp / PCIe | `nvidia-smi` for GPUs 2 and 3 |
| CPU / RAM / disk | `/proc/stat`, `/proc/meminfo`, `/proc/diskstats` |
| electricity | `~/Maya-data/tools/energy-cost.py json` (the same tracker and SQLite DB Maya used; `ik2-energy.service` runs its daemon) |

## Changes from the Strata/Maya code

- `app.js` renders `MonitorPage` only (no chat, sidebar, models, settings).
- `chats.js` / `sidebar.js` are cut to the exports the Monitor uses (`newChat`, `setSide`, `status`).
- `tokens.css` gains an `html[data-scheme="ik2"]` block: deep green-black page, teal accent, amber warnings.
- `monitor.js`: the subtitle and the Context note mention IK2; the VRAM-hit stat (not exposed by
  ingine-serve) is removed; the **Electricity** card (Strata's Settings › Usage card) is moved onto the page.
- `energy.js` is that card, with the "saved minus electricity" stat dropped (IK2 has no API-savings ledger).
- `index.html` is the dashboard's own shell; the theme is dark.
- Engine: the read-only `GET /stats` live counters (see `native/src/serve/chat_server.*`).

## Run

```bash
dashboard/run.sh                 # 0.0.0.0:8080, engine on 127.0.0.1:8090
```

Systemd user units: `ik2-dashboard.service`, `ik2-energy.service`
(`Conflicts=maya-glm.service strata-qwen.service`; the energy tracker `Conflicts=maya-energy.service`).
Same URL on loopback and the tailnet: `http://127.0.0.1:8080/` and `http://100.127.90.18:8080/`.
