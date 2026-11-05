// The two drawings: the model on Dynamics (stocks as boxes, flows as arrows, outside as a cloud at
// either end) and the event map on Machines (who publishes which type, and who listens for it).
// Both read the page's embedded JSON (#data); plain SVG, no library.
(function () {
  const el = document.getElementById("data");
  if (!el) return;
  const d = JSON.parse(el.textContent);
  const NS = "http://www.w3.org/2000/svg";
  const make = (tag, attrs, text) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };
  const svg = (host, w, h) => {
    const s = make("svg", { viewBox: `0 0 ${w} ${h}` });
    s.appendChild(make("defs", {})).appendChild(make("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }))
      .appendChild(make("path", { d: "M0,0 L10,5 L0,10 z", fill: "currentColor" }));
    host.appendChild(s);
    return s;
  };
  const arrow = (s, x1, y1, x2, y2, label, dashed) => {
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 - 18;
    s.appendChild(make("path", { d: `M${x1},${y1} Q${mx},${my} ${x2},${y2}`, fill: "none", stroke: "currentColor", "stroke-opacity": 0.6,
      "stroke-dasharray": dashed ? "4 3" : "", "marker-end": "url(#arrow)" }));
    if (label) s.appendChild(make("text", { x: mx, y: my, "text-anchor": "middle" }, label));
  };
  const box = (s, x, y, w, h, label, round) => {
    s.appendChild(make("rect", { x: x - w / 2, y: y - h / 2, width: w, height: h, rx: round ? h / 2 : 4, fill: "none", stroke: "currentColor" }));
    s.appendChild(make("text", { x, y: y + 4, "text-anchor": "middle" }, label.length > 28 ? label.slice(0, 27) + "…" : label));
  };

  const model = document.getElementById("model");
  if (model && d.model && d.model.stocks.length) {
    const m = d.model, W = Math.max(600, 260 * (m.stocks.length + 1)), H = 120 + 50 * m.flows.length;
    const s = svg(model, W, H), at = {};
    m.stocks.forEach((st, i) => { at[st.id] = { x: 260 * (i + 1) - 60 + 130, y: H / 2 }; });
    const inX = 60, outX = W - 60;
    m.flows.forEach((f, i) => {
      const off = (i - (m.flows.length - 1) / 2) * 34;
      const a = f.from ? at[f.from] : { x: inX, y: H / 2 + off }, b = f.to ? at[f.to] : { x: outX, y: H / 2 + off };
      if (!f.from) box(s, a.x, a.y, 70, 24, "outside", true);
      if (!f.to) box(s, b.x, b.y, 70, 24, "outside", true);
      arrow(s, a.x + (f.from ? 80 : 35), a.y + (f.from ? off / 3 : 0), b.x - (f.to ? 80 : 35), b.y + (f.to ? off / 3 : 0), f.counter ? f.counter.name : f.id);
    });
    for (const st of m.stocks) box(s, at[st.id].x, at[st.id].y, 160, 44, st.series ? st.series.name : st.id);
  }

  const map = document.getElementById("map");
  if (map && d.edges && d.edges.length) {
    const pubs = [...new Set(d.edges.filter((e) => e.from).map((e) => e.from))];
    const subs = [...new Set(d.edges.filter((e) => e.to).map((e) => e.to))];
    const types = [...new Set(d.edges.map((e) => e.type))];
    const rows = Math.max(pubs.length, subs.length, types.length), W = 900, H = 40 + 36 * rows, s = svg(map, W, H);
    const y = (i, n) => 20 + (i + 0.5) * (H - 40) / n;
    const P = Object.fromEntries(pubs.map((p, i) => [p, y(i, pubs.length)]));
    const T = Object.fromEntries(types.map((t, i) => [t, y(i, types.length)]));
    const S = Object.fromEntries(subs.map((p, i) => [p, y(i, subs.length)]));
    for (const e of d.edges) {
      if (e.from) arrow(s, 210, P[e.from], 340, T[e.type]);
      else arrow(s, 560, T[e.type], 690, S[e.to], "", true);
    }
    pubs.forEach((p) => box(s, 120, P[p], 170, 26, p));
    types.forEach((t) => box(s, 450, T[t], 210, 26, t, true));
    subs.forEach((p) => box(s, 780, S[p], 170, 26, p));
  }
})();
