#!/usr/bin/env python3
"""dashboard/server.py - the IK2 monitor's server.

Maya's Monitor page (serve/web/monitor.js) is unchanged; this process gives it the same JSON shape Maya's
/metrics does, built from what the IK2 stack actually exposes:

  * GET  http://127.0.0.1:8090/health   the live slots (running, active, waiting)
  * journalctl -u ik2-serve.service     one line per finished request (prompt/cached/ttft/decode)
  * nvidia-smi                          GPUs 2 and 3: load, VRAM, power, temperature, PCIe
  * /proc/stat, /proc/meminfo, /proc/diskstats

No engine changes. The page is served from this process on 0.0.0.0:8080, so the same URL answers on loopback
and on the tailnet.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = os.environ.get("IK2_URL", "http://127.0.0.1:8090").rstrip("/")
UNIT = os.environ.get("IK2_UNIT", "ik2-serve.service")
GPU_IDS = {int(x) for x in os.environ.get("IK2_GPUS", "2,3").split(",") if x.strip()}
HOST = os.environ.get("IK2_DASH_HOST", "0.0.0.0")
PORT = int(os.environ.get("IK2_DASH_PORT", "8080"))
MAX_CONTEXT = 262144
HISTORY = 60

PREFILL_RE = re.compile(
    r"(chatcmpl-[0-9a-f]+): slot=.*?reused=(\d+) recomputed=(\d+).*?prefill=([\d.]+)s")
DONE_RE = re.compile(
    r"(chatcmpl-[0-9a-f]+): kv_cache_mode=\S+ prompt=(\d+) cached=(\d+) recomputed=(\d+) .*?"
    r"completion=(\d+) finish=(\w+) queued=([\d.]+)s ttft=([\d.]+)s decode=([\d.]+) tok/s")
CANCEL_RE = re.compile(r"(chatcmpl-[0-9a-f]+): client disconnected after (\d+) tokens; cancelled")
STAMP_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)")


def log(msg: str) -> None:
    print(f"[ik2-dash] {msg}", flush=True)


def http_json(path: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(ENGINE + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def stamp_epoch(line: str) -> float:
    m = STAMP_RE.match(line)
    if not m:
        return time.time()
    try:
        return time.mktime(time.strptime(m.group(1)[:19], "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return time.time()


class Collector:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.since = time.time()
        self.health: dict = {}
        self.gpus: list[dict] = []
        self.cpu = 0.0
        self.ram_used = 0
        self.ram_total = 0
        self.disk_read_mb = 0.0
        self.disk_write_mb = 0.0
        self.requests: deque[dict] = deque(maxlen=200)
        self.current: dict[str, dict] = {}
        self.totals = {"requests": 0, "prompt_tokens": 0, "reused": 0, "output_tokens": 0,
                       "prompt_ms": 0.0, "decode_ms": 0.0, "peak_decode": None}
        self.history: dict[str, deque] = {k: deque(maxlen=HISTORY) for k in
                                          ("tok_s", "prefill_tok_s_mean", "gpu_util", "gpu_mem_used",
                                           "gpu_power", "gpu_temp", "gpu_pcie_rx_mb", "cpu",
                                           "ram_used", "disk_read_mb")}
        self._prev_cpu: tuple[int, int] | None = None
        self._prev_disk: tuple[float, float] | None = None

    # ---------------------------------------------------------------- journal: one line per request
    def follow_journal(self) -> None:
        while True:
            cmd = ["journalctl", "--user", "-u", UNIT, "-f", "-o", "cat", "--since=-2hours"]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                        bufsize=1)
            except Exception as e:  # journalctl missing: the page still shows live slots and hardware
                log(f"journalctl: {e}")
                time.sleep(10)
                continue
            log("following " + " ".join(cmd))
            for line in proc.stdout or []:
                self._parse(line.rstrip("\n"))
            proc.wait()
            time.sleep(2)

    def _parse(self, line: str) -> None:
        m = PREFILL_RE.search(line)
        if m:
            cid, reused, recomputed, prefill_s = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4))
            with self.lock:
                self.current[cid] = {"time": stamp_epoch(line), "reused": reused,
                                     "recomputed": recomputed, "prefill_s": prefill_s,
                                     "prompt": reused + recomputed}
            return
        m = DONE_RE.search(line)
        if m:
            cid = m.group(1)
            prompt, cached, recomputed = int(m.group(2)), int(m.group(3)), int(m.group(4))
            out, finish = int(m.group(5)), m.group(6)
            queued_s, ttft_s, decode_s = float(m.group(7)), float(m.group(8)), float(m.group(9))
            decode_tok_s = float(m.group(9))
            with self.lock:
                pre = self.current.pop(cid, {})
                prefill_s = pre.get("prefill_s")
                prefill_tok_s = (recomputed / prefill_s) if prefill_s else None
                duration = queued_s + ttft_s + (out / decode_tok_s if decode_tok_s > 0 else 0.0)
                rec = {"time": pre.get("time", stamp_epoch(line)), "finish": finish,
                       "prompt_tokens": prompt, "reused": cached, "prefill_tok_s": prefill_tok_s,
                       "output_tokens": out, "decode_tok_s": decode_tok_s, "hit_rate": None,
                       "duration_s": duration, "api": "openai", "spec_off": False}
                self.requests.append(rec)
                t = self.totals
                t["requests"] += 1
                t["prompt_tokens"] += prompt
                t["reused"] += cached
                t["output_tokens"] += out
                if prefill_s:
                    t["prompt_ms"] += prefill_s * 1000.0
                t["decode_ms"] += (out / decode_tok_s * 1000.0) if decode_tok_s > 0 else 0.0
                if out >= 32 and (t["peak_decode"] is None or decode_tok_s > t["peak_decode"]["tok_s"]):
                    t["peak_decode"] = {"tok_s": decode_tok_s, "time": rec["time"], "tokens": out}
            return
        m = CANCEL_RE.search(line)
        if m:
            cid, out = m.group(1), int(m.group(2))
            with self.lock:
                pre = self.current.pop(cid, {})
                if pre:
                    rec = {"time": pre.get("time", stamp_epoch(line)), "finish": "cancel",
                           "prompt_tokens": pre.get("prompt", 0), "reused": pre.get("reused", 0),
                           "prefill_tok_s": (pre.get("recomputed", 0) / pre["prefill_s"]) if pre.get("prefill_s") else None,
                           "output_tokens": out, "decode_tok_s": None, "hit_rate": None,
                           "duration_s": max(0.0, stamp_epoch(line) - pre.get("time", stamp_epoch(line))),
                           "api": "openai", "spec_off": False}
                    self.requests.append(rec)
                    self.totals["requests"] += 1
                    self.totals["output_tokens"] += out

    # ---------------------------------------------------------------- hardware
    def sample(self) -> None:
        while True:
            t0 = time.time()
            h = http_json("/health") or {}
            gpus = self._gpus()
            cpu = self._cpu()
            ram_used, ram_total = self._ram()
            disk_read, disk_write = self._disk()
            with self.lock:
                self.health = h
                self.gpus = gpus
                self.cpu = cpu
                self.ram_used, self.ram_total = ram_used, ram_total
                self.disk_read_mb, self.disk_write_mb = disk_read, disk_write
                self._history(gpus, cpu, ram_used, disk_read)
            time.sleep(max(0.2, 1.0 - (time.time() - t0)))

    def _gpus(self) -> list[dict]:
        q = ("index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,"
             "pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max")
        try:
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True, timeout=3)
        except Exception:
            return []
        rows = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 11:
                continue
            try:
                idx = int(p[0])
            except ValueError:
                continue
            if idx not in GPU_IDS:
                continue

            def num(x):
                try:
                    return float(x)
                except ValueError:
                    return None
            rows.append({"index": idx, "name": p[1], "util": num(p[2]),
                         "mem_used": (num(p[3]) or 0) * 1024 * 1024,
                         "mem_total": (num(p[4]) or 0) * 1024 * 1024,
                         "temp": num(p[5]), "power": num(p[6]), "power_limit": num(p[7]),
                         "pcie_gen": num(p[8]), "pcie_width": num(p[9]), "pcie_gen_max": num(p[10])})
        return rows

    def _cpu(self) -> float:
        try:
            with open("/proc/stat") as f:
                vals = [int(x) for x in f.readline().split()[1:]]
        except Exception:
            return 0.0
        idle, total = vals[3] + vals[4], sum(vals)
        prev, self._prev_cpu = self._prev_cpu, (idle, total)
        if not prev:
            return 0.0
        dt = total - prev[1]
        return 100.0 * (1.0 - (idle - prev[0]) / dt) if dt > 0 else 0.0

    @staticmethod
    def _ram() -> tuple[int, int]:
        try:
            info = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    info[k] = int(v.split()[0]) * 1024
            total, avail = info.get("MemTotal", 0), info.get("MemAvailable", 0)
            return max(0, total - avail), total
        except Exception:
            return 0, 0

    def _disk(self) -> tuple[float, float]:
        read = write = 0
        try:
            with open("/proc/diskstats") as f:
                for line in f:
                    p = line.split()
                    if len(p) < 10 or p[2].startswith(("loop", "ram", "dm-", "sr")):
                        continue
                    read += int(p[5])
                    write += int(p[9])
        except Exception:
            return 0.0, 0.0
        now = time.time()
        prev, self._prev_disk = self._prev_disk, (now, read, write)
        if not prev:
            return 0.0, 0.0
        dt = max(1e-3, now - prev[0])
        return (read - prev[1]) * 512 / dt / 1e6, (write - prev[2]) * 512 / dt / 1e6

    def _history(self, gpus: list[dict], cpu: float, ram_used: int, disk_read: float) -> None:
        def avg(k):
            v = [g[k] for g in gpus if g.get(k) is not None]
            return sum(v) / len(v) if v else None

        def total(k):
            v = [g[k] for g in gpus if g.get(k) is not None]
            return sum(v) if v else None

        last = self.requests[-1] if self.requests else None
        # a finished request's rate is a spike at the second it landed, not a flat line for the next minute
        fresh = last is not None and time.time() - last["time"] <= 2
        self.history["tok_s"].append(last["decode_tok_s"] if fresh else None)
        self.history["prefill_tok_s_mean"].append(last.get("prefill_tok_s") if fresh else None)
        self.history["gpu_util"].append(avg("util"))
        self.history["gpu_mem_used"].append(total("mem_used"))
        self.history["gpu_power"].append(total("power"))
        self.history["gpu_temp"].append(max((g["temp"] for g in gpus if g.get("temp") is not None), default=None))
        self.history["gpu_pcie_rx_mb"].append(None)
        self.history["cpu"].append(cpu)
        self.history["ram_used"].append(ram_used)
        self.history["disk_read_mb"].append(disk_read)

    # ---------------------------------------------------------------- the page's JSON
    def metrics(self, all_requests: bool = False) -> dict:
        with self.lock:
            h, gpus = dict(self.health), [dict(g) for g in self.gpus]
            reqs = list(self.requests)
            current = dict(self.current)
            totals = dict(self.totals)
            hist = {k: list(v) for k, v in self.history.items()}
            cpu, ram_used, ram_total = self.cpu, self.ram_used, self.ram_total
            disk_read, disk_write = self.disk_read_mb, self.disk_write_mb
        active = int(h.get("active") or 0)
        queued = int(h.get("waiting") or 0)
        running = bool(h.get("running")) or active > 0
        now = time.time()
        cur = next(iter(current.values()), None)
        last = reqs[-1] if reqs else None
        state = "generating" if running else "idle"
        live = {"state": state, "queued": queued, "active": active, "writing": active,
                "spec_off": False, "phase": None,
                "prompt_tokens": (cur or {}).get("prompt") if running and cur else (last or {}).get("prompt_tokens"),
                "prompt_read": None, "prompt_total": None,
                "generated": None, "max_tokens": None,
                "elapsed_s": round(now - cur["time"], 1) if running and cur else None,
                "tok_s": None, "tok_s_mean": None, "prefill_tok_s_mean": None,
                "prompt_reused": (cur or {}).get("reused") if running and cur else (last or {}).get("reused"),
                "tok_s_window_s": 1 if running else None}
        engine = {"model": "inkling-2-mini", "model_label": "Inkling-2-Mini UD-IQ4_XS + DSpark",
                  "max_context": MAX_CONTEXT, "images": False, "version": "atlas-ingine",
                  "engine_kind": "ik2", "speculation": ["DSpark"]}
        gpu_util = [g["util"] for g in gpus if g.get("util") is not None]
        gpu_mem = [g["mem_used"] for g in gpus if g.get("mem_used") is not None]
        gpu_mem_total = [g["mem_total"] for g in gpus if g.get("mem_total") is not None]
        gpu_temp = [g["temp"] for g in gpus if g.get("temp") is not None]
        gpu_power = [g["power"] for g in gpus if g.get("power") is not None]
        gpu_plim = [g["power_limit"] for g in gpus if g.get("power_limit") is not None]
        gen = max((g.get("pcie_gen_max") or g.get("pcie_gen") or 0) for g in gpus) if gpus else 0
        width = max((g.get("pcie_width") or 0) for g in gpus) if gpus else 0
        hardware = {"gpus": gpus,
                    "gpu_util": sum(gpu_util) / len(gpu_util) if gpu_util else None,
                    "gpu_mem_used": sum(gpu_mem) if gpu_mem else None,
                    "gpu_mem_total": sum(gpu_mem_total) if gpu_mem_total else None,
                    "gpu_temp": max(gpu_temp) if gpu_temp else None,
                    "gpu_power": sum(gpu_power) if gpu_power else None,
                    "gpu_power_limit": sum(gpu_plim) if gpu_plim else None,
                    "gpu_pcie_gen": gen or None, "gpu_pcie_gen_max": gen or None,
                    "gpu_pcie_width": width or None, "gpu_pcie_rx_mb": None,
                    "cpu": cpu, "ram_used": ram_used, "ram_total": ram_total,
                    "disk_read_mb": disk_read, "disk_write_mb": disk_write}
        totals["since"] = self.since
        return {"engine": engine, "live": live, "tiers": None, "gpu_map": None,
                "requests": reqs[::-1][:None if all_requests else 12],
                "requests_kept": len(reqs), "totals": totals,
                "loop_guard": {"on": False}, "hardware": hardware,
                "hardware_static": {"gpu_name": "Tesla V100-PCIE-16GB", "cores": 28, "threads": 56,
                                    "psutil": False},
                "history": hist, "time": now, "reload": None, "storage": None, "savings": None}

    def health_json(self) -> dict:
        with self.lock:
            h = dict(self.health)
        return {"status": "ok", "model": "inkling-2-mini",
                "model_label": "Inkling-2-Mini UD-IQ4_XS + DSpark",
                "max_context": MAX_CONTEXT, "images": False, "version": "atlas-ingine",
                "api_key": False, "kv_cache_mode": h.get("kv_cache_mode", "per_pass"),
                "max_batch": h.get("max_batch", 4), "max_queue": h.get("max_queue", 8),
                "defaults": {"temperature": 1.0, "top_p": 1.0, "top_k": 0}}

    def report(self) -> str:
        m = self.metrics(True)
        eng, hw, t = m["engine"], m["hardware"], m["totals"]
        lines = [f"IK2 monitor report - {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"model: {eng['model_label']} ({eng['model']})",
                 f"context: {eng['max_context']}  kv: {m['live'].get('state')}",
                 f"requests: {t['requests']}  prompt {t['prompt_tokens']} (reused {t['reused']})  output {t['output_tokens']}",
                 "GPUs:"]
        for g in hw["gpus"]:
            lines.append(f"  GPU {g['index']} {g['name']}: {g['util']}%  "
                         f"{g['mem_used']/2**30:.1f}/{g['mem_total']/2**30:.1f} GB  {g['temp']} C  {g['power']} W")
        lines.append(f"CPU {hw['cpu']:.0f}%  RAM {hw['ram_used']/2**30:.1f}/{hw['ram_total']/2**30:.1f} GB")
        lines.append("last requests:")
        for r in m["requests"][:8]:
            lines.append(f"  {time.strftime('%H:%M:%S', time.localtime(r['time']))} {r['finish']:6} "
                         f"prompt {r['prompt_tokens']:6} out {r['output_tokens']:5} "
                         f"decode {r['decode_tok_s'] or 0:6.1f} tok/s")
        return "\n".join(lines) + "\n"


COL = Collector()


class Handler(BaseHTTPRequestHandler):
    server_version = "ik2-monitor/1"

    def log_message(self, fmt, *args):  # quiet: one line per poll would be noise
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/health":
            self._json(COL.health_json())
            return
        if path == "/metrics":
            self._json(COL.metrics(all_requests="requests=all" in self.path))
            return
        if path == "/mcp":
            self._json({"servers": [], "tools": 0})
            return
        if path == "/api/report":
            self._send(200, COL.report().encode(), "text/plain; charset=utf-8")
            return
        if path == "/api/context":
            self._json({"context": MAX_CONTEXT, "min": 4096, "max": MAX_CONTEXT,
                        "stops": [4096, 8192, 16384, 32768, 65536, 131072, 262144]})
            return
        if path in ("/api/port", "/api/thinking-budget", "/api/update", "/settings"):
            self._json(None)
            return
        if path.startswith("/web/"):
            f = (ROOT / path.lstrip("/")).resolve()
            if ROOT not in f.parents or not f.is_file():
                self._send(404, b"not found", "text/plain")
                return
            ctype = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                     ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml",
                     ".png": "image/png", ".woff2": "font/woff2"}.get(f.suffix, "application/octet-stream")
            self._send(200, f.read_bytes(), ctype)
            return
        self._send(404, b"not found", "text/plain")


def main() -> None:
    threading.Thread(target=COL.sample, name="sample", daemon=True).start()
    threading.Thread(target=COL.follow_journal, name="journal", daemon=True).start()
    log(f"serving the IK2 monitor on http://{HOST}:{PORT}  (engine {ENGINE}, unit {UNIT})")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
