/**
 * Performance chart — PPL vs FLOPs (log-log), animated like MGM's Polyglot viz.
 * Data exported from plot_performance.py / wandb_trace_cache.json.
 */
(() => {
  const chart = document.getElementById('perfChart');
  const svg = document.getElementById('perfSvg');
  const tip = document.getElementById('perfTip');
  if (!chart || !svg) return;

  const NS = 'http://www.w3.org/2000/svg';
  const W = 800;
  const H = 480;
  const isRecord = document.documentElement.classList.contains('record');
  /* Record mode: outer CSS supplies the title gap; keep SVG top pad minimal */
  const M = { t: isRecord ? 6 : 28, r: 22, b: 34, l: 42 };
  const XMIN = 0.6;
  const XMAX = 30;
  const YMIN = 19;
  const YMAX = 41;
  const plotW = W - M.l - M.r;
  const plotH = H - M.t - M.b;
  const logX0 = Math.log(XMIN);
  const logXSpan = Math.log(XMAX) - logX0;
  const logY0 = Math.log(YMIN);
  const logYSpan = Math.log(YMAX) - logY0;

  const MODEL_STYLE = {
    MoE: { fc: '#f8cecc', stroke: '#B85450', marker: 'circle', z: 2, label: 'MoE' },
    AoE: { fc: '#d5e8d4', stroke: '#82B366', marker: 'square', z: 2, label: 'AoE' },
    ReMoE: { fc: '#dae8fc', stroke: '#6C8EBF', marker: 'triangle', z: 3, label: 'ReMoE' },
    RFMoE: { fc: '#fff2cc', stroke: '#D6B656', marker: 'star', z: 4, label: 'RFMoE (Ours)' },
  };
  const SCALE_DASH = { S: '', M: '7 5', L: '2.5 3.5' };
  const MODEL_ORDER = ['MoE', 'AoE', 'ReMoE', 'RFMoE'];

  const xOf = (flops) => M.l + ((Math.log(flops) - logX0) / logXSpan) * plotW;
  const yOf = (ppl) => M.t + ((Math.log(YMAX) - Math.log(ppl)) / logYSpan) * plotH;

  const el = (tag, attrs = {}, parent = svg) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      node.setAttribute(k, v);
    }
    parent.appendChild(node);
    return node;
  };

  const delayStyle = (sec) => `animation-delay:${sec.toFixed(2)}s`;

  const pathFromXY = (xs, ys) => {
    // Clip segments to the log-log plot box so early high-PPL points do not escape.
    const xMin = M.l;
    const xMax = M.l + plotW;
    const yMin = M.t;
    const yMax = M.t + plotH;
    const pts = [];
    for (let i = 0; i < xs.length; i++) {
      if (!(xs[i] > 0) || !(ys[i] > 0)) continue;
      pts.push([xOf(xs[i]), yOf(ys[i])]);
    }
    if (!pts.length) return '';

    const inside = (p) => p[0] >= xMin && p[0] <= xMax && p[1] >= yMin && p[1] <= yMax;
    const lerpEdge = (a, b) => {
      // Walk a→b; return first intersection with the plot rectangle.
      let t0 = 0;
      let t1 = 1;
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const checks = [
        [-dx, a[0] - xMin],
        [dx, xMax - a[0]],
        [-dy, a[1] - yMin],
        [dy, yMax - a[1]],
      ];
      for (const [p, q] of checks) {
        if (p === 0) {
          if (q < 0) return null;
          continue;
        }
        const r = q / p;
        if (p < 0) {
          if (r > t1) return null;
          if (r > t0) t0 = r;
        } else {
          if (r < t0) return null;
          if (r < t1) t1 = r;
        }
      }
      if (t0 > t1) return null;
      // Entering from outside: use t0; leaving: use t1 handled by caller via both ends.
      return [
        [a[0] + t0 * dx, a[1] + t0 * dy],
        [a[0] + t1 * dx, a[1] + t1 * dy],
      ];
    };

    let d = '';
    let penDown = false;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (inside(p)) {
        if (!penDown) {
          if (i > 0) {
            const hit = lerpEdge(pts[i - 1], p);
            if (hit) {
              d += `M${hit[0][0].toFixed(2)} ${hit[0][1].toFixed(2)}`;
              d += `L${p[0].toFixed(2)} ${p[1].toFixed(2)}`;
            } else {
              d += `M${p[0].toFixed(2)} ${p[1].toFixed(2)}`;
            }
          } else {
            d += `M${p[0].toFixed(2)} ${p[1].toFixed(2)}`;
          }
          penDown = true;
        } else {
          d += `L${p[0].toFixed(2)} ${p[1].toFixed(2)}`;
        }
      } else if (penDown) {
        const hit = lerpEdge(pts[i - 1], p);
        if (hit) d += `L${hit[1][0].toFixed(2)} ${hit[1][1].toFixed(2)}`;
        penDown = false;
      }
    }
    return d;
  };

  const starPath = (px, py, r) => {
    const pts = [];
    for (let i = 0; i < 5; i++) {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / 5;
      const b = a + Math.PI / 5;
      pts.push([px + Math.cos(a) * r, py + Math.sin(a) * r]);
      pts.push([px + Math.cos(b) * r * 0.45, py + Math.sin(b) * r * 0.45]);
    }
    return `M${pts.map((p) => p.join(',')).join('L')}Z`;
  };

  const triPath = (px, py, r) =>
    `M${px},${py - r} L${px + r * 0.92},${py + r * 0.72} L${px - r * 0.92},${py + r * 0.72}Z`;

  const squarePath = (px, py, r) => {
    const s = r * 0.85;
    return `M${px - s},${py - s} L${px + s},${py - s} L${px + s},${py + s} L${px - s},${py + s}Z`;
  };

  const markerAttrs = (style, px, py, ours) => {
    const r = ours ? 9 : 6.2;
    const base = {
      class: `perf-dot${ours ? ' ours' : ''}`,
      fill: style.fc,
      stroke: '#12161F',
      'stroke-width': '1.2',
    };
    if (style.marker === 'star') return { tag: 'path', attrs: { ...base, d: starPath(px, py, r) } };
    if (style.marker === 'triangle') return { tag: 'path', attrs: { ...base, d: triPath(px, py, r) } };
    if (style.marker === 'square') return { tag: 'path', attrs: { ...base, d: squarePath(px, py, r) } };
    return { tag: 'circle', attrs: { ...base, cx: px, cy: py, r: r * 0.72 } };
  };

  const build = (payload) => {
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const defs = el('defs');
    const clip = el('clipPath', { id: 'perfPlotClip' }, defs);
    el('rect', { x: M.l, y: M.t, width: plotW, height: plotH }, clip);

    const shadeGrad = el('linearGradient', {
      id: 'annShadeGrad',
      x1: '100%',
      y1: '0%',
      x2: '0%',
      y2: '100%',
      gradientUnits: 'objectBoundingBox',
    }, defs);
    [
      ['0%', '#DF1F22', '0.16'],
      ['42%', '#B85450', '0.09'],
      ['100%', '#101218', '0.02'],
    ].forEach(([offset, color, opacity]) => {
      el('stop', { offset, 'stop-color': color, 'stop-opacity': opacity }, shadeGrad);
    });

    const gGrid = el('g', { class: 'grid', 'clip-path': 'url(#perfPlotClip)' });
    const xTicks = [1, 2, 3, 5, 10, 20];
    const yTicks = [20, 25, 30, 35, 40];

    yTicks.forEach((y) => {
      el('line', {
        class: 'grid-line',
        x1: M.l, x2: W - M.r, y1: yOf(y), y2: yOf(y),
      }, gGrid);
    });
    xTicks.forEach((x) => {
      el('line', {
        class: 'grid-line',
        x1: xOf(x), x2: xOf(x), y1: M.t, y2: H - M.b,
      }, gGrid);
    });

    yTicks.forEach((y) => {
      const ty = yOf(y);
      const t = el('text', { class: 'tick', x: M.l - 6, y: ty + 3.5, 'text-anchor': 'end' });
      t.textContent = String(y);
    });
    xTicks.forEach((x) => {
      const tx = xOf(x);
      const t = el('text', { class: 'tick', x: tx, y: H - M.b + 12, 'text-anchor': 'middle' });
      t.textContent = String(x);
    });

    const xlab = el('text', {
      class: 'axis-label',
      x: M.l + plotW / 2,
      y: H - 4,
      'text-anchor': 'middle',
    });
    xlab.textContent = 'FLOPs (EFLOPs)';
    const ylabX = 13;
    const ylabY = M.t + plotH / 2;
    const ylab = el('text', {
      class: 'axis-label',
      x: ylabX,
      y: ylabY,
      'text-anchor': 'middle',
      transform: `rotate(-90 ${ylabX} ${ylabY})`,
    });
    ylab.textContent = 'Perplexity (PPL) ↓';

    const gAnn = el('g', { class: 'annotations', 'clip-path': 'url(#perfPlotClip)' });
    const gTraces = el('g', { class: 'traces', 'clip-path': 'url(#perfPlotClip)' });
    const gDots = el('g', { class: 'dots' });
    const gAnnText = el('g', { class: 'annotation-labels' });
    const gScaleTags = el('g', { class: 'scale-tags' });
    const hitNodes = [];
    const pendingTraces = [];
    let clipSeq = 0;

    const annByScale = Object.fromEntries(
      (payload.annotations || []).map((a) => [a.scale, a])
    );

    // Collect runs grouped by scale, models in display order within each group.
    const SCALE_ORDER = ['S', 'M', 'L'];
    const byScale = { S: [], M: [], L: [] };
    MODEL_ORDER.forEach((model) => {
      (payload.models[model] || []).forEach((run) => {
        byScale[run.scale]?.push({ model, run, style: MODEL_STYLE[model] });
      });
    });

    const addTraceWithMarker = (model, run, style, delay) => {
      const d = pathFromXY(run.x, run.y);
      if (!d) return null;

      const finalDash = SCALE_DASH[run.scale] || '';
      const markerX = xOf(run.flops);
      const markerY = yOf(run.ppl);
      // Reveal only up to this curve's endpoint (+ pad), not the full plot width.
      const revealW = Math.max(8, Math.min(plotW, markerX - M.l + 10));

      const clipId = `perfTraceClip${clipSeq++}`;
      const cp = el('clipPath', { id: clipId }, defs);
      const rect = el('rect', {
        x: M.l,
        y: M.t - 4,
        width: '0',
        height: plotH + 8,
      }, cp);

      el('path', {
        class: `trace${model === 'RFMoE' ? ' ours' : ''}`,
        d,
        fill: 'none',
        stroke: style.stroke,
        'stroke-width': model === 'RFMoE' ? '2' : '1.5',
        'stroke-dasharray': finalDash || null,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        'clip-path': `url(#${clipId})`,
        'data-scale': run.scale,
      }, gTraces);

      const ours = model === 'RFMoE';
      const mk = markerAttrs(style, markerX, markerY, ours);
      const marker = el(mk.tag, {
        ...mk.attrs,
        class: `${mk.attrs.class || 'perf-dot'} wait`,
        'data-scale': run.scale,
      }, gDots);
      marker.dataset.name = `${style.label} · ${run.scale}`;
      marker.dataset.ppl = run.ppl.toFixed(2);
      marker.dataset.flops = String(run.flops);
      marker.dataset.kind = ours ? 'Routing-Free MoE' : model;
      hitNodes.push(marker);

      const entry = {
        rect,
        marker,
        delay,
        revealW,
        markerX,
      };
      pendingTraces.push(entry);
      return entry;
    };

    const addScaleTag = (scale, items, delay) => {
      if (!items.length) return;
      // Place tag near the geometric mean of final marker x, above the top of the group.
      const xs = items.map(({ run }) => xOf(run.flops));
      const ys = items.map(({ run }) => yOf(run.ppl));
      const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
      const topY = Math.min(...ys);
      const tag = el('text', {
        class: 'scale-tag anim-ann',
        x: cx,
        y: Math.max(M.t + 14, topY - 16),
        'text-anchor': 'middle',
        style: delayStyle(delay),
      }, gScaleTags);
      tag.textContent = scale;
    };

    const addAnnotations = (ann, delay) => {
      if (!ann) return delay;
      const ls = SCALE_DASH[ann.scale] || '';
      const t = delay;
      const lineAt = t + 0.03;
      const textAt = t + 0.08;
      let end = textAt + 0.2;

      if (ann.rf_flops_at_moe != null && ann.speedup != null) {
        const x0 = xOf(ann.rf_flops_at_moe);
        const x1 = xOf(ann.moe_flops);
        const y = yOf(ann.moe_ppl);
        const yBot = yOf(YMIN);
        el('rect', {
          class: 'ann-shade anim-ann',
          x: Math.min(x0, x1),
          y,
          width: Math.abs(x1 - x0),
          height: Math.max(0, yBot - y),
          fill: 'url(#annShadeGrad)',
          'data-scale': ann.scale,
          style: delayStyle(t),
        }, gAnn);
        el('line', {
          class: 'ann-line anim-ann',
          x1: x0, x2: x1, y1: y, y2: y,
          'stroke-dasharray': ls || null,
          'data-scale': ann.scale,
          style: delayStyle(lineAt),
        }, gAnn);
        const mx = ((x0 + x1) / 2).toFixed(2);
        const tx = el('text', {
          class: 'ann-text anim-ann',
          x: mx,
          y: yBot - 20,
          'text-anchor': 'middle',
          'data-scale': ann.scale,
          style: delayStyle(textAt),
        }, gAnnText);
        const t1 = el('tspan', { x: mx, dy: '0' }, tx);
        t1.textContent = `${ann.speedup.toFixed(1)}×`;
        const t2 = el('tspan', { x: mx, dy: '12' }, tx);
        t2.textContent = 'faster';
      }

      if (ann.pct_better != null && ann.rf_flops != null) {
        const x = xOf(ann.rf_flops);
        const y0 = yOf(ann.rf_ppl);
        const y1 = yOf(ann.moe_ppl);
        el('line', {
          class: 'ann-line anim-ann',
          x1: x, x2: x, y1: y0, y2: y1,
          'stroke-dasharray': ls || null,
          'data-scale': ann.scale,
          style: delayStyle(lineAt),
        }, gAnn);
        const lx = Math.min(x + 10, M.l + plotW - 36);
        const ly = (y0 + y1) / 2;
        const tx = el('text', {
          class: 'ann-text anim-ann',
          x: lx,
          y: ly - 4,
          'text-anchor': 'start',
          'data-scale': ann.scale,
          style: delayStyle(textAt),
        }, gAnnText);
        const t1 = el('tspan', { x: lx, dy: '0' }, tx);
        t1.textContent = `${ann.pct_better.toFixed(0)}%`;
        const t2 = el('tspan', { x: lx, dy: '12' }, tx);
        t2.textContent = 'better';
      }

      return end;
    };

    // Animate by scale group: S → M → L.
    // Per group: scale tag → MoE → baselines → RFMoE → annotations.
    const TIMING = {
      START: 0.06,
      TAG_LEAD: 0.05,
      MODEL_GAP: { MoE: 0, AoE: 0.11, ReMoE: 0.22, RFMoE: 0.38 },
      DRAW_DUR: 1.02,
      ANN_PAUSE: 0.08,
      GROUP_GAP: 0.18,
    };

    let tDelay = TIMING.START;

    SCALE_ORDER.forEach((scale) => {
      const items = byScale[scale] || [];
      if (!items.length) return;

      const groupStart = tDelay;
      addScaleTag(scale, items, groupStart);

      let lastEnd = groupStart;
      let rfEnd = 0;
      items.forEach(({ model, run, style }) => {
        const traceAt = groupStart + TIMING.TAG_LEAD + (TIMING.MODEL_GAP[model] ?? 0);
        addTraceWithMarker(model, run, style, traceAt);
        const endAt = traceAt + TIMING.DRAW_DUR;
        lastEnd = Math.max(lastEnd, endAt);
        if (model === 'RFMoE') rfEnd = endAt;
      });

      const annAt = (rfEnd || lastEnd) + TIMING.ANN_PAUSE;
      tDelay = addAnnotations(annByScale[scale], annAt);
      tDelay = Math.max(tDelay, annAt) + TIMING.GROUP_GAP;
    });

    // Frame on top of clipped traces.
    el('rect', {
      class: 'plot-frame',
      x: M.l, y: M.t, width: plotW, height: plotH,
      fill: 'none', stroke: 'rgba(18,22,31,.35)', 'stroke-width': '1.2',
    });

    // In-plot legend (top-right), two rows: models then scales.
    const gLeg = el('g', { class: 'perf-legend-svg', 'pointer-events': 'none' });
    const legX = M.l + plotW - 12;
    const legY = M.t + 14;
    const row1 = [
      { model: 'MoE', label: 'MoE' },
      { model: 'AoE', label: 'AoE' },
      { model: 'ReMoE', label: 'ReMoE' },
      { model: 'RFMoE', label: 'RFMoE' },
    ];
    const itemGap = 64;
    const row1W = (row1.length - 1) * itemGap + 52;
    let lx = legX - row1W;
    row1.forEach(({ model, label }) => {
      const style = MODEL_STYLE[model];
      const cy = legY;
      const mk = markerAttrs(style, lx + 7, cy, model === 'RFMoE');
      const node = el(mk.tag, {
        ...mk.attrs,
        class: 'legend-mark',
        opacity: '1',
      }, gLeg);
      // markerAttrs sizes are for plot dots; shrink legend marks slightly
      if (mk.tag === 'circle') {
        node.setAttribute('r', '4.2');
      } else if (style.marker === 'star') {
        node.setAttribute('d', starPath(lx + 7, cy, 6.5));
      } else if (style.marker === 'triangle') {
        node.setAttribute('d', triPath(lx + 7, cy, 5.2));
      } else if (style.marker === 'square') {
        node.setAttribute('d', squarePath(lx + 7, cy, 5));
      }
      node.setAttribute('stroke-width', '1');
      const t = el('text', {
        class: 'legend-label',
        x: lx + 16,
        y: cy + 3.5,
        'text-anchor': 'start',
      }, gLeg);
      t.textContent = label;
      lx += itemGap;
    });

    const row2Y = legY + 18;
    const scales = [
      { key: 'S', dash: '', label: 'S' },
      { key: 'M', dash: '7 5', label: 'M' },
      { key: 'L', dash: '2.5 3.5', label: 'L' },
    ];
    const scaleGap = 48;
    const row2W = (scales.length - 1) * scaleGap + 36;
    let sx = legX - row2W;
    scales.forEach(({ dash, label }) => {
      el('line', {
        x1: sx, x2: sx + 18, y1: row2Y, y2: row2Y,
        stroke: 'rgba(18,22,31,.55)',
        'stroke-width': '2',
        'stroke-dasharray': dash || null,
        'stroke-linecap': 'round',
      }, gLeg);
      const t = el('text', {
        class: 'legend-label',
        x: sx + 22,
        y: row2Y + 3.5,
        'text-anchor': 'start',
      }, gLeg);
      t.textContent = label;
      sx += scaleGap;
    });

    const showTip = (node, clientX, clientY) => {
      tip.innerHTML = `<b>${node.dataset.name}</b><br>PPL ${node.dataset.ppl} · ${node.dataset.flops} EFLOPs<br>${node.dataset.kind}`;
      const rect = chart.getBoundingClientRect();
      tip.style.left = `${clientX - rect.left}px`;
      tip.style.top = `${clientY - rect.top}px`;
      tip.classList.add('on');
    };
    const hideTip = () => tip.classList.remove('on');

    hitNodes.forEach((node) => {
      node.addEventListener('pointerenter', (e) => {
        hitNodes.forEach((n) => n.classList.add(n === node ? 'hi' : 'dim'));
        showTip(node, e.clientX, e.clientY);
      });
      node.addEventListener('pointermove', (e) => showTip(node, e.clientX, e.clientY));
      node.addEventListener('pointerleave', () => {
        hitNodes.forEach((n) => n.classList.remove('hi', 'dim'));
        hideTip();
      });
    });

    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let playGen = 0;

    const showMarker = (marker) => {
      marker.classList.remove('wait');
      marker.classList.add('show');
    };

    const resetChart = () => {
      playGen += 1;
      chart.classList.remove('ready');
      pendingTraces.forEach(({ rect, marker }) => {
        rect.setAttribute('width', '0');
        marker.classList.add('wait');
        marker.classList.remove('show', 'dim', 'hi');
      });
    };

    const revealTraces = () => {
      const dur = TIMING.DRAW_DUR * 1000;
      const ease = (p) => 1 - Math.pow(1 - p, 3);
      const gen = playGen;
      pendingTraces.forEach(({ rect, marker, delay, revealW, markerX }) => {
        if (prefersReduced) {
          rect.setAttribute('width', String(revealW));
          showMarker(marker);
          return;
        }
        const t0 = performance.now() + delay * 1000;
        const showAt = Math.max(4, markerX - M.l);
        let shown = false;
        const tick = (now) => {
          if (gen !== playGen) return;
          if (now < t0) {
            requestAnimationFrame(tick);
            return;
          }
          const p = Math.min(1, (now - t0) / dur);
          const w = revealW * ease(p);
          rect.setAttribute('width', String(w));
          if (!shown && w >= showAt) {
            shown = true;
            showMarker(marker);
          }
          if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
    };

    const play = () => {
      chart.classList.add('ready');
      revealTraces();
    };

    if (isRecord) {
      const page = chart.closest('.page');
      if (page) {
        page.addEventListener('record:enter', play);
        page.addEventListener('record:leave', resetChart);
        if (page.querySelector('.rv.in, .rv-st.in')) play();
      } else if (prefersReduced) {
        play();
      }
    } else {
      const io = new IntersectionObserver((entries) => {
        entries.forEach((e) => {
          if (!e.isIntersecting) return;
          play();
          io.disconnect();
        });
      }, { threshold: 0.25 });
      io.observe(chart);
      if (prefersReduced) play();
    }
  };

  fetch('assets/performance-data.json')
    .then((r) => {
      if (!r.ok) throw new Error(String(r.status));
      return r.json();
    })
    .then(build)
    .catch((err) => {
      console.error('performance chart failed to load', err);
      const t = el('text', {
        class: 'axis-label',
        x: W / 2,
        y: H / 2,
        'text-anchor': 'middle',
      });
      t.textContent = 'Could not load performance data.';
    });
})();
