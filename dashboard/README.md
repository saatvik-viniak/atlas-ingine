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
- **Hardware** — every GPU nvidia-smi reports (or the `IK2_GPUS` filter): load, VRAM, power, temperature, PCIe; CPU, RAM, disk.
- **Electricity** — the machine's cost tracker: watts now (per GPU, CPU, board), live ComEd price, today,
  this month, all time, and the last 14 days.
- **Recent requests** — time, status, prompt/reused/prefill/output/decode, took.

## Data sources

| Panel | Source |
|---|---|
| live slots, queue | `GET http://127.0.0.1:8090/health` |
| live tokens (generated, max, started, first token) | `GET http://127.0.0.1:8090/stats` (new in this PR) |
| finished requests | `journalctl --user -u ik2-serve.service -f -o cat` (the engine's `prefill=` / `completion=` lines) |
| GPU load / VRAM / power / temp / PCIe | `nvidia-smi` (all GPUs, or the `IK2_GPUS` filter) |
| CPU / RAM / disk | `/proc/stat`, `/proc/meminfo`, `/proc/diskstats` |
| electricity | `IK2_ENERGY_SCRIPT json` (opt-in; no external script is run unless it is set) |

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
dashboard/run.sh                 # 127.0.0.1:8080, engine on 127.0.0.1:8090
```

Environment:

| Variable | Default | Meaning |
|---|---|---|
| `IK2_URL` | `http://127.0.0.1:8090` | the ingine-serve base URL |
| `IK2_UNIT` | `ik2-serve.service` | the journal unit to follow |
| `IK2_DASH_HOST` | `127.0.0.1` | bind address; `0.0.0.0` exposes it on other interfaces |
| `IK2_DASH_PORT` | `8080` | listen port |
| `IK2_GPUS` | all | comma-separated GPU indexes to show (e.g. `2,3`) |
| `IK2_MAX_CONTEXT` | engine's `/health` | fallback context size |
| `IK2_ENERGY_SCRIPT` | unset | opt-in: path to `energy-cost.py`; the card is unavailable without it |

The systemd units in this deployment add `IK2_DASH_HOST=0.0.0.0`, `IK2_GPUS=2,3` and
`IK2_ENERGY_SCRIPT=...` as environment overrides; the code defaults stay host-neutral.
