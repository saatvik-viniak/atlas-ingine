// dashboard/web/energy.js - the Electricity card from the Strata/Maya UI (Settings > Usage), on the Monitor.
// GET /api/energy: the machine's own cost tracker (live ComEd price x its draw), polled every 10 s.
import {html, useEffect, useState} from "./preact.js";
import {api} from "./lib.js";

const kwh = (v) => (v == null ? "–" : `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)} kWh`);
const usd2 = (v) => (v == null ? "–" : v < 0.01 && v > 0 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`);
const Stat = ({label, value, sub}) => html`<div class="energy__stat"><span class="energy__label">${label}</span>
  <b class="energy__value">${value}</b>${sub ? html`<span class="energy__sub">${sub}</span>` : null}</div>`;

export function Energy() {
  const [e, setE] = useState(null), [err, setErr] = useState("");
  useEffect(() => {
    let live = true;
    const load = () => api("api/energy").then((x) => live && (setE(x), setErr(""))).catch((x) => live && setErr(x.message));
    load();
    const t = setInterval(load, 10000);
    return () => { live = false; clearInterval(t); };
  }, []);
  if (err) return html`<p class="card__note">Electricity: ${err}</p>`;
  if (!e) return html`<p class="card__note">Reading the electricity tracker…</p>`;
  if (!e.available) return html`<p class="card__note">Electricity: ${e.why}</p>`;
  const n = e.now || {}, t = e.today || {}, m = e.month || {}, tr = e.tracked || {};
  const monthName = m.name ? new Date(`${m.name}-01T12:00:00`).toLocaleDateString([], {month: "long"}) : "This month";
  const trackedSince = tr.since ? new Date(`${tr.since}T12:00:00`).toLocaleDateString([], {month: "short", day: "numeric"}) : "";
  return html`<section class="price energy" aria-label="Electricity">
    <div class="price__head"><span class="energy__icon" aria-hidden="true">⚡</span>
      <span><b>Electricity</b><span>Live from this machine's cost tracker${n.time ? ` · ${n.time.slice(11, 19)}` : ""}${e.error ? ` · last read failed: ${e.error}` : ""}</span></span>
      <span class="pill pill--accent">Live</span></div>
    <div class="energy__grid">
      <${Stat} label=${`All time${trackedSince ? ` (since ${trackedSince})` : ""}`} value=${usd2(tr.usd)}
        sub=${`${kwh(tr.kwh)} · the model working ${usd2(tr.active_usd)}, idle ${usd2(tr.idle_usd)}`}/>
      <${Stat} label="Drawing now" value=${n.ac_w != null ? `${Math.round(n.ac_w)} W` : "–"}
        sub=${`GPUs ${n.gpu_w != null ? Math.round(n.gpu_w) : "–"} W${(n.gpu_each || []).length ? ` (${n.gpu_each.map((w) => Math.round(w)).join(" / ")})` : ""} · CPU ${n.cpu_w != null ? Math.round(n.cpu_w) : "–"} W`}/>
      <${Stat} label="Price now" value=${n.price_cents != null ? `${n.price_cents.toFixed(1)}¢/kWh` : "–"}
        sub=${n.ac_w != null && n.price_cents != null ? `${usd2((n.ac_w / 1000) * (n.price_cents / 100))} an hour at this draw` : ""}/>
      <${Stat} label="Today" value=${usd2(t.usd)} sub=${`${kwh(t.kwh)} · active ${usd2(t.active_usd)}, idle ${usd2(t.idle_usd)}`}/>
      <${Stat} label=${monthName} value=${usd2(m.usd)} sub=${`${kwh(m.kwh)} · avg ${m.avg_w != null ? Math.round(m.avg_w) : "–"} W, peak ${m.peak_w != null ? Math.round(m.peak_w) : "–"} W`}/>
    </div>
    ${(e.days || []).length ? html`<div class="energy__days" role="list" aria-label="Electricity by day">
      ${e.days.slice(-14).map((d) => html`<span role="listitem" title=${`${d.day}: ${kwh(d.kwh)}, ${usd2(d.usd)}`}>
        <span class="energy__day">${d.day ? d.day.slice(5) : ""}</span><b>${usd2(d.usd)}</b><span class="energy__sub">${kwh(d.kwh)}</span></span>`)}
    </div>` : null}
    <div class="price__foot"><span>The live ComEd price times the machine's draw (GPUs measured, CPU and board modelled), from ${e.script ? e.script.replace(/^.*\//, "") : "the tracker"}. Electricity counts every hour the machine is on, not only IK2's.</span></div>
  </section>`;
}
