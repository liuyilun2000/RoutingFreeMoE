(() => {
  const stage = document.getElementById('rfStage');
  const canvas = document.getElementById('rfCanvas');
  if (!stage || !canvas) return;

  const thrEl = document.getElementById('rfThr');
  const thrVal = document.getElementById('rfThrVal');
  const densEl = document.getElementById('rfDens');
  const densVal = document.getElementById('rfDensVal');
  const actEl = document.getElementById('rfActCount');
  const moeActEl = document.getElementById('rfMoeAct');
  const densLive = document.getElementById('rfDensLive');

  const N = 4;
  const TOP_K = 2;
  const SEED = 42;
  const FONT = 'Figtree, system-ui, sans-serif';
  const LINE = 'rgba(232,236,242,0.18)';
  const LINE_W = 1.15;
  const COL_OFF = '#E8A0A0';
  const COL_ON = 'rgba(168,230,216,0.9)';
  const THR_SLIDER_DEFAULT = 0.42;
  const experts = [];

  function mulberry32(a) {
    return function () {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const rand = mulberry32(SEED);
  for (let i = 0; i < N; i++) {
    experts.push({
      id: i,
      bias: -0.32 + rand() * 0.64,
      thrBias: 0.22 + rand() * 0.36,
      phase: rand() * Math.PI * 2,
      freq: 0.55 + rand() * 1.05,
      amp: 0.16 + rand() * 0.26,
      pulseMoe: 0,
      pulseRf: 0,
    });
  }

  let tokenT = 0;
  let tokenComplexity = 0.55;
  let thrAdd = THR_SLIDER_DEFAULT;
  let dpr = Math.min(window.devicePixelRatio || 1, 2);
  let W = 0;
  let H = 0;
  let running = true;
  let last = performance.now();
  let narrow = false;

  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function effectiveThr(e) {
    return Math.max(0.05, Math.min(0.95, e.thrBias + (thrAdd - THR_SLIDER_DEFAULT)));
  }

  function resize() {
    const rect = stage.getBoundingClientRect();
    W = Math.max(320, Math.floor(rect.width));
    H = Math.max(240, Math.floor(rect.height));
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(W * dpr);
    canvas.height = Math.floor(H * dpr);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    narrow = W < 720;
  }

  function score(e, t, complexity) {
    const wave = 0.5 + 0.5 * Math.sin(t * e.freq + e.phase);
    const base = 0.22 + complexity * 0.55 + e.bias * 0.38;
    return Math.max(0.02, Math.min(1.2, base + e.amp * (wave - 0.5) * 2));
  }

  function softmax(xs) {
    const m = Math.max(...xs);
    const exps = xs.map((v) => Math.exp((v - m) * 4.2));
    const s = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / s);
  }

  function topKMask(probs, k) {
    const idx = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const on = new Array(probs.length).fill(false);
    for (let i = 0; i < k; i++) on[idx[i][1]] = true;
    return on;
  }

  function renormGates(probs, onMask) {
    let s = 0;
    for (let i = 0; i < probs.length; i++) if (onMask[i]) s += probs[i];
    if (s <= 1e-8) return probs.map(() => 0);
    return probs.map((p, i) => (onMask[i] ? p / s : 0));
  }

  function expertXs(cx, span) {
    const xs = [];
    for (let i = 0; i < N; i++) xs.push(cx - span / 2 + (i + 0.5) * (span / N));
    return xs;
  }

  function roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function drawPoly(ctx, pts, color, width) {
    if (!pts || pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.strokeStyle = color || LINE;
    ctx.lineWidth = width == null ? LINE_W : width;
    ctx.lineJoin = 'miter';
    ctx.lineCap = 'square';
    ctx.stroke();
  }

  // Fan-out: corner at mid vertical distance; full horizontal from stem to each target
  function drawFanOut(ctx, cx, yFrom, xs, yTo, color, width) {
    if (!xs.length) return;
    const midY = (yFrom + yTo) / 2;
    const c = color || LINE;
    const w = width == null ? LINE_W : width;
    drawPoly(ctx, [[cx, yFrom], [cx, midY]], c, w);
    const xMin = Math.min(...xs, cx);
    const xMax = Math.max(...xs, cx);
    drawPoly(ctx, [[xMin, midY], [xMax, midY]], c, w);
    xs.forEach((ex) => {
      drawPoly(ctx, [[ex, midY], [ex, yTo]], c, w);
    });
  }

  // Fan-in: corner at mid vertical distance; full horizontal across sources into stem
  function drawFanIn(ctx, xs, yFrom, cx, yTo, color, width) {
    if (!xs.length) return;
    const midY = (yFrom + yTo) / 2;
    const c = color || LINE;
    const w = width == null ? LINE_W : width;
    xs.forEach((ex) => {
      drawPoly(ctx, [[ex, yFrom], [ex, midY]], c, w);
    });
    const xMin = Math.min(...xs, cx);
    const xMax = Math.max(...xs, cx);
    drawPoly(ctx, [[xMin, midY], [xMax, midY]], c, w);
    drawPoly(ctx, [[cx, midY], [cx, yTo]], c, w);
  }

  // Orthogonal link between two points with mid-length corner(s)
  function drawElbow(ctx, x1, y1, x2, y2, color, width) {
    const c = color || LINE;
    const w = width == null ? LINE_W : width;
    if (Math.abs(x1 - x2) < 0.5) {
      drawPoly(ctx, [[x1, y1], [x2, y2]], c, w);
      return;
    }
    if (Math.abs(y1 - y2) < 0.5) {
      drawPoly(ctx, [[x1, y1], [x2, y2]], c, w);
      return;
    }
    const midY = (y1 + y2) / 2;
    drawPoly(ctx, [[x1, y1], [x1, midY], [x2, midY], [x2, y2]], c, w);
  }

  function drawPanelTitle(ctx, x, y, text, accent) {
    ctx.fillStyle = accent;
    ctx.fillRect(x, y - 9, 3, 12);
    ctx.fillStyle = 'rgba(232,236,242,0.88)';
    ctx.font = `600 13px ${FONT}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + 10, y - 2);
  }

  function drawToken(ctx, x, y, r) {
    ctx.beginPath();
    ctx.arc(x, y, r + 7, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(233,196,106,0.1)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = '#E9C46A';
    ctx.fill();
    ctx.fillStyle = '#12161F';
    ctx.font = `600 ${Math.max(11, r * 0.65)}px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('x', x, y + 0.5);
  }

  function drawBox(ctx, x, y, w, h, title, tone) {
    const fill =
      tone === 'warn'
        ? 'rgba(196,92,92,0.14)'
        : tone === 'hot'
          ? 'rgba(233,196,106,0.14)'
          : 'rgba(232,236,242,0.06)';
    const stroke =
      tone === 'warn'
        ? 'rgba(196,92,92,0.55)'
        : tone === 'hot'
          ? 'rgba(233,196,106,0.5)'
          : 'rgba(232,236,242,0.18)';
    roundRect(ctx, x - w / 2, y - h / 2, w, h, 3);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.fillStyle =
      tone === 'warn' ? '#E8A0A0' : tone === 'hot' ? '#FFF2CC' : 'rgba(232,236,242,0.85)';
    ctx.font = `600 ${Math.max(11, Math.min(13, h * 0.42))}px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(title, x, y + 0.5);
  }

  function drawExpertNode(ctx, x, y, r, i, lit, pulse) {
    ctx.beginPath();
    ctx.arc(x, y, r + 5 * pulse, 0, Math.PI * 2);
    ctx.fillStyle = lit
      ? `rgba(42,157,143,${0.12 + 0.22 * pulse})`
      : 'rgba(94,122,140,0.08)';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = lit ? '#2A9D8F' : '#3A4558';
    ctx.fill();
    ctx.strokeStyle = lit ? 'rgba(232,236,242,0.55)' : 'rgba(232,236,242,0.18)';
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.fillStyle = lit ? 'rgba(232,236,242,0.92)' : 'rgba(232,236,242,0.45)';
    ctx.font = `500 ${Math.max(10, r * 0.55)}px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('E' + i, x, y + 0.5);
  }

  function drawConfBar(ctx, x, y, w, h, val, thr, on) {
    ctx.fillStyle = 'rgba(232,236,242,0.12)';
    ctx.fillRect(x - w / 2, y, w, h);
    ctx.fillStyle = on ? '#2A9D8F' : '#8A93A6';
    ctx.fillRect(x - w / 2, y, w * Math.min(1, Math.max(0, val)), h);
    const tx = x - w / 2 + w * Math.min(1, Math.max(0, thr));
    ctx.strokeStyle = '#E9C46A';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(tx, y - 2);
    ctx.lineTo(tx, y + h + 2);
    ctx.stroke();
  }

  function paint(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (running && !prefersReduced) tokenT += dt;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const logits = experts.map((e) => score(e, tokenT, tokenComplexity));
    const probs = softmax(logits);
    const moeOn = topKMask(probs, TOP_K);
    const gates = renormGates(probs, moeOn);
    const thrEach = experts.map(effectiveThr);
    const rfOn = logits.map((p, i) => p >= thrEach[i]);

    let moeActive = 0;
    let rfActive = 0;
    experts.forEach((e, i) => {
      if (moeOn[i]) {
        moeActive += 1;
        e.pulseMoe = Math.min(1, e.pulseMoe + dt * 3);
      } else {
        e.pulseMoe = Math.max(0, e.pulseMoe - dt * 2.2);
      }
      if (rfOn[i]) {
        rfActive += 1;
        e.pulseRf = Math.min(1, e.pulseRf + dt * 3);
      } else {
        e.pulseRf = Math.max(0, e.pulseRf - dt * 2.2);
      }
    });

    if (moeActEl) moeActEl.textContent = String(moeActive);
    if (actEl) actEl.textContent = String(rfActive);
    if (densLive) densLive.textContent = (rfActive / N).toFixed(2);

    const pad = narrow ? 14 : 20;
    const gap = narrow ? 10 : 16;
    const leftX = pad;
    const leftW = narrow ? W - pad * 2 : (W - pad * 2 - gap) / 2;
    const rightX = narrow ? pad : leftX + leftW + gap;
    const rightW = leftW;
    const leftY0 = narrow ? 26 : 22;
    const rightY0 = narrow ? H * 0.515 : 22;
    const panelH = narrow ? H * 0.45 : H - 30;

    if (!narrow) {
      ctx.strokeStyle = 'rgba(232,236,242,0.12)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(W * 0.5, 16);
      ctx.lineTo(W * 0.5, H - 14);
      ctx.stroke();
    } else {
      ctx.strokeStyle = 'rgba(232,236,242,0.12)';
      ctx.beginPath();
      ctx.moveTo(pad, H * 0.5);
      ctx.lineTo(W - pad, H * 0.5);
      ctx.stroke();
    }

    drawTraditional(ctx, leftX, leftY0, leftW, panelH, moeOn, gates);
    drawRoutingFree(ctx, rightX, rightY0, rightW, panelH, logits, rfOn, thrEach);

    requestAnimationFrame(paint);
  }

  function drawTraditional(ctx, ox, oy, pw, ph, onMask, gates) {
    drawPanelTitle(ctx, ox + 4, oy + 4, 'Traditional MoE', '#C45C5C');

    const cx = ox + pw * 0.5;
    const tokenR = Math.min(pw, ph) * 0.04;
    const tokenY = oy + ph * 0.1;
    const boxW = Math.min(pw * 0.36, 108);
    const boxH = Math.max(22, ph * 0.048);

    // Router → Softmax → TopK stack (experts sit fully below this)
    const routerY = oy + ph * 0.2;
    const softY = oy + ph * 0.3;
    const topY = oy + ph * 0.4;

    const expertY = oy + ph * 0.62;
    const expertR = Math.min(pw / (N * 3.1), ph * 0.058, 16);
    const span = Math.min(pw * 0.82, expertR * N * 3.3);
    const xs = expertXs(cx, span);
    const gateY = expertY + expertR + 16;
    const outY = oy + ph * 0.92;

    drawToken(ctx, cx, tokenY, tokenR);
    drawPoly(ctx, [[cx, tokenY + tokenR + 2], [cx, routerY - boxH / 2 - 2]], LINE, LINE_W);
    drawBox(ctx, cx, routerY, boxW, boxH, 'Router', 'hot');
    drawPoly(ctx, [[cx, routerY + boxH / 2 + 2], [cx, softY - boxH / 2 - 2]], LINE, LINE_W);
    drawBox(ctx, cx, softY, boxW, boxH, 'Softmax', 'warn');
    drawPoly(ctx, [[cx, softY + boxH / 2 + 2], [cx, topY - boxH / 2 - 2]], LINE, LINE_W);
    drawBox(ctx, cx, topY, boxW, boxH, 'TopK=' + TOP_K, 'warn');

    // TopK input gate: mid-rail intercept — blocked paths stop here
    const topBottom = topY + boxH / 2 + 2;
    const expTop = expertY - expertR - 3;
    const interceptY = (topBottom + expTop) / 2;
    const activeXs = xs.filter((_, i) => onMask[i]);
    const blockedXs = xs.filter((_, i) => !onMask[i]);

    // stem + full horizontal rail at intercept (covers all expert columns)
    drawPoly(ctx, [[cx, topBottom], [cx, interceptY]], LINE, LINE_W);
    drawPoly(ctx, [[xs[0], interceptY], [xs[N - 1], interceptY]], LINE, LINE_W);

    // blocked inputs: short stub down then cut (never reach expert)
    blockedXs.forEach((ex) => {
      const cutY = interceptY + (expTop - interceptY) * 0.35;
      drawPoly(ctx, [[ex, interceptY], [ex, cutY]], COL_OFF, LINE_W);
      // cut mark
      ctx.strokeStyle = COL_OFF;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(ex - 5, cutY - 3);
      ctx.lineTo(ex + 5, cutY + 3);
      ctx.moveTo(ex - 5, cutY + 3);
      ctx.lineTo(ex + 5, cutY - 3);
      ctx.stroke();
    });

    // allowed inputs: continue from intercept into expert
    activeXs.forEach((ex) => {
      drawPoly(ctx, [[ex, interceptY], [ex, expTop]], LINE, LINE_W);
    });

    xs.forEach((ex, i) => {
      drawExpertNode(ctx, ex, expertY, expertR, i, onMask[i], experts[i].pulseMoe);
    });

    // TopK output gate: × on activated experts, off on the rest
    xs.forEach((ex, i) => {
      const on = onMask[i];
      ctx.fillStyle = on ? COL_ON : COL_OFF;
      ctx.font = `500 11px ${FONT}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(
        on ? '×' + (gates[i] != null ? gates[i].toFixed(2) : '0.00') : 'off',
        ex,
        gateY
      );
    });

    // same TopK path continues through activated expert into × (output scale)
    activeXs.forEach((ex) => {
      drawPoly(ctx, [[ex, expertY + expertR + 3], [ex, gateY - 8]], LINE, LINE_W);
    });
    if (activeXs.length) {
      drawFanIn(ctx, activeXs, gateY + 10, cx, outY - 12, LINE, LINE_W);
    }

    ctx.fillStyle = 'rgba(232,236,242,0.55)';
    ctx.font = `600 12px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('Σ  gated outputs', cx, outY);

    ctx.fillStyle = 'rgba(232,236,242,0.38)';
    ctx.font = `500 11px ${FONT}`;
    ctx.fillText('always ' + TOP_K + ' experts  ·  complexity discarded', cx, oy + ph - 2);
  }

  function drawRoutingFree(ctx, ox, oy, pw, ph, logits, onMask, thrEach) {
    drawPanelTitle(ctx, ox + 4, oy + 4, 'Routing-Free MoE', '#2A9D8F');

    const cx = ox + pw * 0.5;
    const tokenR = Math.min(pw, ph) * 0.04;
    const tokenY = oy + ph * 0.12;
    drawToken(ctx, cx, tokenY, tokenR);

    const expertY = oy + ph * 0.42;
    const expertR = Math.min(pw / (N * 3.1), ph * 0.06, 16);
    const span = Math.min(pw * 0.82, expertR * N * 3.3);
    const xs = expertXs(cx, span);

    drawFanOut(ctx, cx, tokenY + tokenR + 2, xs, expertY - expertR - 3, LINE, LINE_W);

    const barH = 3.2;
    const barY = expertY + expertR + 8;
    xs.forEach((ex, i) => {
      const on = onMask[i];
      drawExpertNode(ctx, ex, expertY, expertR, i, on, experts[i].pulseRf);
      drawConfBar(ctx, ex, barY, expertR * 2.4, barH, logits[i], thrEach[i], on);
      ctx.fillStyle = on ? COL_ON : COL_OFF;
      ctx.font = `500 11px ${FONT}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(on ? 'activate' : 'off', ex, barY + barH + 5);
    });

    const labelBottom = barY + barH + 20;
    const outY = oy + ph * 0.9;
    const activeXs = xs.filter((_, i) => onMask[i]);
    if (activeXs.length) {
      drawFanIn(ctx, activeXs, labelBottom, cx, outY - 12, LINE, LINE_W);
    }

    ctx.fillStyle = 'rgba(232,236,242,0.55)';
    ctx.font = `600 12px ${FONT}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('Σ  emergent outputs', cx, outY);

    const nOn = onMask.filter(Boolean).length;
    ctx.fillStyle = 'rgba(232,236,242,0.38)';
    ctx.font = `500 11px ${FONT}`;
    ctx.fillText(nOn + ' / ' + N + ' paths adapt to complexity', cx, oy + ph - 2);
  }

  function syncUI() {
    if (thrEl) {
      thrAdd = parseFloat(thrEl.value);
      if (thrVal) {
        const delta = thrAdd - THR_SLIDER_DEFAULT;
        const sign = delta >= 0 ? '+' : '';
        thrVal.textContent = sign + delta.toFixed(2);
      }
    }
    if (densEl) {
      tokenComplexity = parseFloat(densEl.value);
      if (densVal) densVal.textContent = tokenComplexity.toFixed(2);
    }
  }

  thrEl && thrEl.addEventListener('input', syncUI);
  densEl && densEl.addEventListener('input', syncUI);

  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        running = e.isIntersecting;
      });
    },
    { threshold: 0.05 }
  );
  io.observe(stage);

  window.addEventListener('resize', resize);

  syncUI();
  resize();
  requestAnimationFrame(paint);
})();
