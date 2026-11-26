// dashboard/web/sidebar.js - the sidebar code Maya's app needs, cut to the two exports the Monitor uses.
import {ctxfmt, fmt} from "./lib.js";
import {S} from "./state.js";

export const setSide = (open) => { S.sideOpen = open; };

// the header's status pill (Maya's serve/web/sidebar.js status(), unchanged)
export function status(plain = false) {
  const m = S.metrics, live = (m && m.live) || {};
  const ctx = S.health.max_context ? `${ctxfmt(S.health.max_context)} context` : "";
  if (S.conn === "down") return {tone: "danger", text: "Server not reachable"};
  if (S.conn === "key") return {tone: "warn", text: "API key needed"};
  if (S.updateWatch) return {tone: "info", text: "Updating"};
  if (S.reload) return {tone: "info", text: "Reloading"};
  if (!m) return {tone: "muted", text: "Connecting…"};
  if (live.queued > 0) return {tone: "warn", text: `${live.queued} queued`};
  if (live.state === "reading") {
    const pct = live.prompt_total && !plain ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    return {tone: "info", text: pct != null ? `Reading · ${pct}%` : "Reading"};
  }
  if (live.state === "generating") {
    const many = live.active > 1 && !plain ? ` · ${live.active} at once` : "";
    return {tone: "accent", text: (live.tok_s && !plain ? `Writing · ${fmt(live.tok_s, 1)} tok/s` : "Writing") + many};
  }
  return {tone: "muted", text: ["Idle", plain ? "" : ctx].filter(Boolean).join(" · ")};
}
