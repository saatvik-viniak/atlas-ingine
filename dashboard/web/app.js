// dashboard/web/app.js - the IK2 Monitor page. Maya's Monitor view (serve/web/monitor.js) with the chat,
// sidebar, models and settings stripped: this page only reads /health and /metrics and draws the dashboard.
import {html, render} from "./preact.js";
import {S, useStore} from "./state.js";
import {loadHealth, poll} from "./server.js";
import {MonitorPage} from "./monitor.js";

document.documentElement.dataset.theme = "dark";
document.documentElement.dataset.scheme = "ik2";

S.view = "monitor";
S.sideOpen = true;   // no sidebar: the page is the dashboard

function App() {
  useStore();
  return html`<${MonitorPage}/>`;
}

render(html`<${App}/>`, document.getElementById("app"));
loadHealth().then(() => poll());
