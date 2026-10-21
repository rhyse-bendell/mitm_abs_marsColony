"use strict";
// Participant client. Pure view + input: it holds NO game state of its own. Everything drawn comes from the latest `state` message; everything it
// can do is send move / chosen-menu-item / chat / interface events. All server text is rendered as text nodes (never parsed as markup).
(function () {
  const params = new URLSearchParams(location.search);
  const role = params.get("role") || "", token = params.get("token") || "";
  const $ = (id) => document.getElementById(id);
  const canvas = $("map"), ctx = canvas.getContext("2d");
  let layout = null, view = null, ws = null, hover = null, menuObject = null, dest = null;
  const seenFeedback = new Set(), openedSources = new Set();
  let toastTimer = null;

  // ---- world <-> canvas (y up, like the simulator) ------------------------------------------------------------------------
  const W = () => (layout ? layout.layout.bounds[0] : 10), H = () => (layout ? layout.layout.bounds[1] : 7.4);
  const sx = () => canvas.width / W(), sy = () => canvas.height / H();
  const toPx = (x, y) => [x * sx(), canvas.height - y * sy()];
  const toWorld = (px, py) => [px / sx(), (canvas.height - py) / sy()];

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws?role=${encodeURIComponent(role)}&token=${encodeURIComponent(token)}`);
    ws.onopen = () => { $("overlay").hidden = true; };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.t === "layout") { layout = msg; }
      else if (msg.t === "state") { view = msg.view; onState(); }
      else if (msg.t === "ack" && !msg.ok && msg.reason) { toast(reasonText(msg.reason), false); }
    };
    ws.onclose = () => { $("overlay").textContent = "Disconnected from the experiment. You may close this window."; $("overlay").hidden = false; };
    setInterval(() => send({ t: "ping" }), 5000);
  }
  const send = (m) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(m)); };
  const reasonText = (r) => ({ participant_not_connected: "Not connected.", too_many_pending_intents: "Slow down a little.", experiment_ended: "The experiment has ended.", stale_option: "That option is no longer available." }[r] || "That isn't possible right now.");

  function fmt(s) { s = Math.max(0, Math.round(s)); return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0"); }
  function toast(text, ok) {
    const t = $("toast"); t.textContent = text; t.className = "toast" + (ok ? " ok" : ""); t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, 4500);
  }

  function onState() {
    const y = view.you, m = view.mission;
    $("hud-name").textContent = `${y.role}`; $("hud-dot").style.background = y.color;
    $("hud-phase").textContent = m.phase; $("hud-time").textContent = fmt(m.sim_time); $("hud-total").textContent = fmt(m.total_s); $("hud-remaining").textContent = fmt(m.phase_remaining_s);
    $("hud-status").textContent = m.status === "paused" ? "PAUSED" : (m.ended ? "ENDED" : "Running");
    $("hud-carry").textContent = y.carrying; $("hud-busy").textContent = y.busy ? `${y.busy.label} (${Math.ceil(y.busy.remaining_s)}s)` : (y.moving ? "Walking" : "Idle");
    $("hud-dest").textContent = y.moving ? `Heading to (${y.moving[0].toFixed(1)}, ${y.moving[1].toFixed(1)})` : "No destination";
    dest = y.moving;
    for (const f of view.feedback) {
      if (seenFeedback.has(f.id)) continue; seenFeedback.add(f.id);
      if (f.kind !== "accepted" && f.message) toast(f.message, f.level === "ok");
    }
    renderChat(); renderInfo();
    if (menuObject) renderMenu();
    draw();
  }

  // ---- drawing ---------------------------------------------------------------------------------------------------------
  const KIND_COLOR = { info_own: "#ffd166", info_shared: "#7ad3ff", info_other: "#6b7685", artifact: "#c3a6ff", site: "#8a6d4b", project: "#d9a066", pile: "#b5651d", bridge: "#8d97a5" };
  function draw() {
    if (!layout || !view) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#2b211a"; ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (const w of layout.layout.walls) { const [a, b] = toPx(w.x, w.y + w.h); ctx.fillStyle = "#0c0a09"; ctx.fillRect(a, b, w.w * sx(), w.h * sy()); }
    for (const o of layout.layout.objects) {
      ctx.strokeStyle = "#9b8470"; ctx.fillStyle = "#3a2e25"; ctx.lineWidth = 2;
      if (o.shape === "rect") { const [a, b] = toPx(o.x, o.y + o.h); ctx.fillRect(a, b, o.w * sx(), o.h * sy()); ctx.strokeRect(a, b, o.w * sx(), o.h * sy()); label(o.label, a + 2, b - 4); }
      else if (o.shape === "circle") { const [a, b] = toPx(o.x, o.y); ctx.beginPath(); ctx.arc(a, b, o.r * sx(), 0, 6.283); ctx.fill(); ctx.stroke(); label("Site " + o.label, a - 18, b + 4); }
      else if (o.shape === "line") { const [a, b] = toPx(o.x1, o.y1), [c, d] = toPx(o.x2, o.y2); ctx.beginPath(); ctx.moveTo(a, b); ctx.lineTo(c, d); ctx.lineWidth = 6; ctx.strokeStyle = "#8d97a5"; ctx.stroke(); }
    }
    if (view.bridge_bc_visible && layout.layout) { /* bridge B-C object marker is drawn below as an interactive object */ }
    for (const o of view.objects) {
      const [a, b] = toPx(o.x, o.y), col = KIND_COLOR[o.kind] || "#aaa";
      ctx.fillStyle = col; ctx.strokeStyle = o.near ? "#fff" : "#0008"; ctx.lineWidth = o.near ? 3 : 1;
      if (o.kind === "pile") { ctx.beginPath(); ctx.arc(a, b, 10, 0, 6.283); ctx.fill(); ctx.stroke(); label(o.status ? `bricks: ${o.status}` : "bricks", a + 12, b + 4); }
      else if (o.kind === "project") { ctx.fillRect(a - 8, b - 8, 16, 16); ctx.strokeRect(a - 8, b - 8, 16, 16); label(`${o.structure_type.replace("_", " ")} ${o.materials} ${o.status.replace("_", " ")}${o.validation_failed ? " - failed validation" : ""}`, a + 12, b + 4); }
      else if (o.kind === "bridge") { ctx.beginPath(); ctx.moveTo(a - 10, b); ctx.lineTo(a, b - 10); ctx.lineTo(a + 10, b); ctx.lineTo(a, b + 10); ctx.closePath(); ctx.fill(); ctx.stroke(); label(`bridge: ${o.status.replace("_", " ")}`, a + 12, b + 4); }
      else if (o.kind.startsWith("info") || o.kind === "artifact") { ctx.strokeStyle = o.near ? "#fff" : col; ctx.lineWidth = 3; ctx.strokeRect(a - 22, b - 14, 44, 28); }
      if (o.kind === "site") { ctx.strokeStyle = o.near ? "#fff" : "#0000"; ctx.beginPath(); ctx.arc(a, b, o.radius * sx() + 30, 0, 6.283); ctx.stroke(); }
    }
    if (dest) { const [a, b] = toPx(dest[0], dest[1]); ctx.strokeStyle = view.you.color; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(a, b, 8, 0, 6.283); ctx.moveTo(a - 12, b); ctx.lineTo(a + 12, b); ctx.moveTo(a, b - 12); ctx.lineTo(a, b + 12); ctx.stroke(); }
    for (const t of view.agents) { const [a, b] = toPx(t.x, t.y); dot(a, b, 9, t.color, false); label(`${t.role} (${t.activity})`, a + 12, b - 8); }
    const [ya, yb] = toPx(view.you.x, view.you.y); dot(ya, yb, 12, view.you.color, true); label("You", ya + 14, yb - 10);
  }
  function dot(a, b, r, color, ring) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(a, b, r, 0, 6.283); ctx.fill(); if (ring) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 3; ctx.stroke(); } }
  function label(text, a, b) { ctx.fillStyle = "#e6e9ee"; ctx.font = "13px system-ui"; ctx.fillText(text, a, b); }

  // ---- picking ---------------------------------------------------------------------------------------------------------
  function pick(px, py) {
    if (!view) return null; let best = null, bd = 28;
    for (const o of view.objects) { const [a, b] = toPx(o.x, o.y); const d = Math.hypot(a - px, b - py); if (d < bd) { bd = d; best = o; } }
    return best;
  }
  const canvasPoint = (ev) => { const r = canvas.getBoundingClientRect(); return [(ev.clientX - r.left) * canvas.width / r.width, (ev.clientY - r.top) * canvas.height / r.height]; };
  canvas.addEventListener("mousemove", (ev) => {
    const [px, py] = canvasPoint(ev), o = pick(px, py), tip = $("tooltip");
    if (!o) { tip.hidden = true; return; }
    tip.textContent = `${o.title}\n${o.near ? "Click to interact" : "Move closer to interact"}`;
    const r = canvas.getBoundingClientRect(); tip.style.left = (ev.clientX - r.left + 14) + "px"; tip.style.top = (ev.clientY - r.top + 14) + "px"; tip.hidden = false;
  });
  canvas.addEventListener("mouseleave", () => { $("tooltip").hidden = true; });
  canvas.addEventListener("click", (ev) => {
    if (!view) return; const [px, py] = canvasPoint(ev), o = pick(px, py);
    if (o) { if (o.near) { openMenu(o.id); } else { closeMenu(); send({ t: "move_object", object: o.id }); } return; }
    const [x, y] = toWorld(px, py); closeMenu(); send({ t: "move", x: Math.round(x * 100) / 100, y: Math.round(y * 100) / 100 });
  });

  // ---- menu ------------------------------------------------------------------------------------------------------------
  function openMenu(id) { menuObject = id; send({ t: "menu_opened", object: id }); renderMenu(); }
  function closeMenu() { menuObject = null; $("menu-panel").hidden = true; }
  function renderMenu() {
    const o = view.objects.find((x) => x.id === menuObject); if (!o) { closeMenu(); return; }
    $("menu-panel").hidden = false; $("menu-title").textContent = o.title;
    $("menu-sub").textContent = o.near ? (o.status ? `Status: ${String(o.status).replace("_", " ")}` : "") : "Move closer to interact.";
    const ul = $("menu-items"); ul.textContent = "";
    if (!o.items.length) { const li = document.createElement("li"); li.className = "detail"; li.textContent = "Nothing to do here right now."; ul.appendChild(li); }
    for (const it of o.items) {
      const li = document.createElement("li"), b = document.createElement("button");
      b.textContent = it.label; b.disabled = !it.enabled || !o.near; b.title = !o.near ? "Move closer to interact" : (it.reason || "");
      b.onclick = () => { send({ t: "action", item: it.id }); if (!it.id.startsWith("reopen:")) closeMenu(); };
      li.appendChild(b); if (it.detail) { const d = document.createElement("div"); d.className = "detail"; d.textContent = it.detail; li.appendChild(d); }
      ul.appendChild(li);
    }
  }
  $("menu-close").onclick = closeMenu;

  // ---- info + chat -----------------------------------------------------------------------------------------------------
  function renderInfo() {
    const box = $("info-list"); const srcs = view.info.sources; if (!srcs.length) return; box.textContent = "";
    for (const s of srcs) {
      const d = document.createElement("details"), sm = document.createElement("summary"); sm.textContent = `${s.label} (${s.scope})`; d.appendChild(sm);
      if (s.open && !openedSources.has(s.id)) { openedSources.add(s.id); d.open = true; } else if (openedSources.has(s.id) && s.open) { d.open = true; }
      d.addEventListener("toggle", () => { if (!d.open) { openedSources.delete(s.id); send({ t: "close_source", source: s.id }); } });
      const ul = document.createElement("ul"); for (const it of s.items) { const li = document.createElement("li"); li.textContent = it.text; ul.appendChild(li); } d.appendChild(ul); box.appendChild(d);
    }
  }
  let lastChat = "";
  function renderChat() {
    const sig = view.chat.map((c) => c.id).join(","); if (sig === lastChat) return; lastChat = sig;
    const log = $("chat-log"); log.textContent = "";
    for (const c of view.chat) {
      const div = document.createElement("div"); div.className = "msg" + (c.mine ? " mine" : "") + (c.kind === "agent_message" ? " agent" : "");
      const b = document.createElement("b"); b.textContent = (c.mine ? "You" : c.sender) + ": "; div.appendChild(b); div.appendChild(document.createTextNode(c.text)); log.appendChild(div);
    }
    log.scrollTop = log.scrollHeight;
  }
  $("chat-form").addEventListener("submit", (ev) => { ev.preventDefault(); const i = $("chat-input"); const text = i.value; i.value = ""; send({ t: "chat", text }); });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") { closeMenu(); document.activeElement && document.activeElement.blur(); }
    else if (ev.key === "Enter" && document.activeElement !== $("chat-input")) { ev.preventDefault(); $("chat-input").focus(); }
  });
  connect();
})();
