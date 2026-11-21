// Tiny canvas line chart (no dependency). series: [{name, color, points:[[x,y],...]}]
export function lineChart(canvas, series, { xLabel = "", yFmt = v => (+v).toPrecision(3), xInt = true } = {}) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const W = canvas.clientWidth || 280, H = canvas.clientHeight || 120;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  const fg = css.getPropertyValue("--muted").trim() || "#8b97b0", grid = css.getPropertyValue("--line").trim() || "#222b40";
  const pts = series.flatMap(s => s.points).filter(p => Number.isFinite(p[1]));
  g.font = '13px "Patrick Hand", "Comic Sans MS", cursive'; g.fillStyle = fg;
  if (!pts.length) { g.fillText("no data yet", 8, H / 2); return; }
  let x0 = Math.min(...pts.map(p => p[0])), x1 = Math.max(...pts.map(p => p[0]));
  let y0 = Math.min(...pts.map(p => p[1])), y1 = Math.max(...pts.map(p => p[1]));
  if (x1 === x0) { x0 -= 1; x1 += 1; }
  if (y1 === y0) { const d = Math.abs(y0) * 0.1 || 1; y0 -= d; y1 += d; }
  const padL = 44, padR = 8, padT = 8, padB = 18;
  const X = x => padL + (x - x0) / (x1 - x0) * (W - padL - padR);
  const Y = y => padT + (1 - (y - y0) / (y1 - y0)) * (H - padT - padB);
  g.strokeStyle = grid; g.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = y0 + (y1 - y0) * i / 3, py = Math.round(Y(y)) + 0.5;
    g.beginPath(); g.moveTo(padL, py); g.lineTo(W - padR, py); g.stroke();
    g.fillText(yFmt(y), 2, py + 4);
  }
  const xt = xInt ? [...new Set(pts.map(p => p[0]))].sort((a, b) => a - b) : [x0, x1];
  const stepX = Math.max(1, Math.ceil(xt.length / 8));
  xt.forEach((x, i) => { if (i % stepX === 0) g.fillText(xInt ? String(x) : x.toFixed(1), X(x) - 3, H - 4); });
  if (xLabel) { g.textAlign = "right"; g.fillText(xLabel, W - padR, padT + 2); g.textAlign = "left"; }
  for (const s of series) {
    const p = s.points.filter(q => Number.isFinite(q[1]));
    if (!p.length) continue;
    g.strokeStyle = s.color; g.fillStyle = s.color; g.lineWidth = 2;
    g.beginPath(); p.forEach((q, i) => i ? g.lineTo(X(q[0]), Y(q[1])) : g.moveTo(X(q[0]), Y(q[1]))); g.stroke();
    if (p.length < 40) for (const q of p) { g.beginPath(); g.arc(X(q[0]), Y(q[1]), 2.5, 0, Math.PI * 2); g.fill(); }
  }
}
