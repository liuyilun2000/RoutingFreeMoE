/**
 * Downstream results as a 2×3 grid of mini bar charts
 * (Average + 5 benchmarks), styled like small multiples panels.
 * Palette matches the LM quality chart.
 */
(() => {
  const root = document.getElementById('resultsBars');
  if (!root) return;

  const NS = 'http://www.w3.org/2000/svg';
  const PW = 200;
  const PH = 168;
  const isRecord = document.documentElement.classList.contains('record');
  /* Panel titles still need room; slightly tighter in record mode */
  const PM = { t: isRecord ? 28 : 36, r: 8, b: 20, l: 28 };
  const plotW = PW - PM.l - PM.r;
  const plotH = PH - PM.t - PM.b;
  const BASE_Y = PM.t + plotH;

  const PALETTE = {
    MoE: { fill: '#E08A86', stroke: '#f8cecc' },
    AoE: { fill: '#9BC984', stroke: '#d5e8d4' },
    ReMoE: { fill: '#8AA9D4', stroke: '#dae8fc' },
    RFMoE: { fill: '#E2C86A', stroke: '#fff2cc' },
  };

  const BENCH_RAW = ['PIQA', 'ARCe', 'OBQA', 'QQP', 'QNLI'];
  const BENCH_IDX = { PIQA: 0, HellaS: 1, WinoG: 2, ARCe: 3, ARCc: 4, OBQA: 5, QQP: 6, QNLI: 7, 'SST-2': 8 };
  const SUB = {
    Average: 'Performance across 9 benchmarks',
    PIQA: 'Physical Interaction QA',
    ARCe: 'AI2 Reasoning Challenge · Easy',
    OBQA: 'OpenBook Question Answering',
    QQP: 'Quora Question Pairs',
    QNLI: 'Question-answering NLI',
  };

  const RUNS = [
    { scale: 'S', arch: 'MoE', scores: [57.56, 27.19, 51.22, 33.42, 21.33, 24.60, 36.82, 49.46, 49.08], avg: 38.96 },
    { scale: 'S', arch: 'AoE', scores: [56.09, 27.01, 50.20, 33.84, 21.93, 25.00, 36.82, 49.46, 49.08], avg: 38.82 },
    { scale: 'S', arch: 'ReMoE', scores: [56.58, 26.69, 51.62, 33.38, 22.27, 26.00, 36.83, 49.48, 49.08], avg: 39.10 },
    { scale: 'S', arch: 'RFMoE', scores: [58.49, 27.09, 50.59, 35.77, 21.59, 24.40, 36.98, 49.44, 53.56], avg: 39.77 },
    { scale: 'M', arch: 'MoE', scores: [58.32, 28.26, 52.33, 36.20, 21.42, 24.60, 36.85, 49.46, 49.31], avg: 39.64 },
    { scale: 'M', arch: 'RFMoE', scores: [58.92, 27.85, 49.72, 35.27, 21.50, 26.40, 37.06, 49.51, 57.34], avg: 40.40 },
    { scale: 'L', arch: 'MoE', scores: [59.19, 29.37, 50.83, 36.41, 23.29, 25.00, 37.57, 49.28, 49.08], avg: 40.00 },
    { scale: 'L', arch: 'RFMoE', scores: [58.87, 28.27, 50.51, 37.46, 21.67, 26.60, 39.93, 49.86, 53.67], avg: 40.76 },
  ];

  const BENCH = BENCH_RAW
    .map((name) => {
      const i = BENCH_IDX[name];
      return {
        name,
        mean: RUNS.reduce((s, r) => s + r.scores[i], 0) / RUNS.length,
        i,
      };
    })
    .sort((a, b) => a.mean - b.mean);

  const PANELS = [
    { title: 'Average', values: RUNS.map((r) => r.avg) },
    ...BENCH.map(({ name, i }) => ({
      title: name,
      values: RUNS.map((r) => r.scores[i]),
    })),
  ];

  const el = (tag, attrs = {}, parent) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      node.setAttribute(k, v);
    }
    parent.appendChild(node);
    return node;
  };

  const niceTicks = (lo, hi, n = 4) => {
    const span = hi - lo || 1;
    const step = Math.pow(10, Math.floor(Math.log10(span / n)));
    const err = (n * step) / span;
    const step2 = err <= 0.15 ? step * 10 : err <= 0.35 ? step * 5 : err <= 0.75 ? step * 2 : step;
    const start = Math.ceil(lo / step2) * step2;
    const ticks = [];
    for (let v = start; v <= hi + 1e-9; v += step2) ticks.push(+v.toFixed(6));
    return ticks;
  };

  const easeOut = (p) => 1 - Math.pow(1 - p, 3);
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let playGen = 0;

  root.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'rb-grid';
  root.appendChild(grid);

  const tip = document.createElement('div');
  tip.className = 'rb-tip';
  tip.id = 'resultsBarsTip';
  tip.setAttribute('aria-hidden', 'true');
  root.appendChild(tip);

  const showTip = (html, clientX, clientY) => {
    tip.innerHTML = html;
    const rect = root.getBoundingClientRect();
    tip.style.left = `${clientX - rect.left}px`;
    tip.style.top = `${clientY - rect.top}px`;
    tip.classList.add('on');
  };
  const hideTip = () => tip.classList.remove('on');

  /** @type {{bar: SVGRectElement, val: SVGTextElement, y: number, h: number, delay: number}[]} */
  const animBars = [];
  /** @type {HTMLElement[]} */
  const panels = [];

  PANELS.forEach((panel, panelIdx) => {
    const wrap = document.createElement('div');
    wrap.className = 'rb-panel';
    wrap.style.setProperty('--panel-i', String(panelIdx));
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${PW} ${PH}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `${panel.title} accuracy by scale and architecture`);
    wrap.appendChild(svg);
    grid.appendChild(wrap);
    panels.push(wrap);

    const vals = panel.values;
    const vmin = Math.min(...vals);
    const vmax = Math.max(...vals);
    const pad = Math.max(1.5, (vmax - vmin) * 0.18);
    let y0 = Math.max(0, vmin - pad);
    let y1 = vmax + pad;
    if (y1 - y0 < 6) {
      const mid = (y0 + y1) / 2;
      y0 = mid - 3;
      y1 = mid + 3;
    }
    const yOf = (v) => PM.t + ((y1 - v) / (y1 - y0)) * plotH;

    const title = el('text', { class: 'rb-title rb-chrome', x: PM.l, y: 14 }, svg);
    title.textContent = panel.title;
    const sub = el('text', { class: 'rb-sub rb-chrome', x: PM.l, y: 26 }, svg);
    sub.textContent = SUB[panel.title] || '';

    const ticks = niceTicks(y0, y1, 3);
    ticks.forEach((t) => {
      const yy = yOf(t);
      if (yy < PM.t - 1 || yy > PM.t + plotH + 1) return;
      el('line', {
        class: 'rb-grid-line rb-chrome',
        x1: PM.l, x2: PW - PM.r, y1: yy, y2: yy,
      }, svg);
      const lab = el('text', {
        class: 'rb-tick rb-chrome',
        x: PM.l - 4,
        y: yy + 3,
        'text-anchor': 'end',
      }, svg);
      lab.textContent = Number.isInteger(t) ? String(t) : t.toFixed(1);
    });

    const groups = [
      { scale: 'S', count: 4 },
      { scale: 'M', count: 2 },
      { scale: 'L', count: 2 },
    ];
    const barGap = 4;
    const groupGap = 8;
    const nBars = RUNS.length;
    const usable = plotW - groupGap * (groups.length - 1);
    const barW = Math.min(14, (usable - barGap * (nBars - groups.length)) / nBars);

    let x = PM.l + (plotW - (nBars * barW + barGap * (nBars - groups.length) + groupGap * (groups.length - 1))) / 2;
    let runIdx = 0;
    const panelDelay = panelIdx * 0.12;

    groups.forEach((g, gi) => {
      const gStart = x;
      for (let k = 0; k < g.count; k++) {
        const run = RUNS[runIdx];
        const v = vals[runIdx];
        const pal = PALETTE[run.arch];
        const y = yOf(v);
        const h = Math.max(0, yOf(y0) - y);
        const barDelay = panelDelay + runIdx * 0.045;

        const bar = el('rect', {
          class: 'rb-bar',
          x,
          y: BASE_Y,
          width: barW,
          height: 0,
          rx: '2',
          fill: pal.fill,
          stroke: pal.stroke,
          'stroke-width': '1',
        }, svg);
        bar.dataset.label = `${run.arch} · ${run.scale}`;
        bar.dataset.value = v.toFixed(2);
        bar.addEventListener('pointerenter', (e) => {
          showTip(`<b>${panel.title}</b><br>${run.arch} · ${run.scale}: ${v.toFixed(2)}`, e.clientX, e.clientY);
        });
        bar.addEventListener('pointermove', (e) => {
          showTip(`<b>${panel.title}</b><br>${run.arch} · ${run.scale}: ${v.toFixed(2)}`, e.clientX, e.clientY);
        });
        bar.addEventListener('pointerleave', hideTip);

        const valLab = el('text', {
          class: 'rb-val',
          x: x + barW / 2,
          y: y - 3,
          'text-anchor': 'middle',
          opacity: '0',
        }, svg);
        valLab.textContent = v.toFixed(1);

        animBars.push({ bar, val: valLab, y, h, delay: barDelay });

        x += barW + (k < g.count - 1 ? barGap : 0);
        runIdx += 1;
      }
      const gEnd = x;
      const scaleLab = el('text', {
        class: 'rb-scale rb-chrome',
        x: (gStart + gEnd) / 2,
        y: BASE_Y + 12,
        'text-anchor': 'middle',
      }, svg);
      scaleLab.textContent = g.scale;

      if (gi < groups.length - 1) {
        x += groupGap;
        const divx = x - groupGap / 2;
        el('line', {
          class: 'rb-sep rb-chrome',
          x1: divx, x2: divx, y1: PM.t, y2: PM.t + plotH,
        }, svg);
      }
    });

    el('line', {
      class: 'rb-baseline rb-chrome',
      x1: PM.l, x2: PW - PM.r, y1: BASE_Y, y2: BASE_Y,
    }, svg);
  });

  const legend = document.createElement('div');
  legend.className = 'rb-legend';
  ['MoE', 'AoE', 'ReMoE', 'RFMoE'].forEach((arch) => {
    const item = document.createElement('span');
    item.className = 'rb-legend-item';
    const sw = document.createElement('i');
    sw.className = 'rb-swatch';
    sw.style.background = PALETTE[arch].fill;
    sw.style.borderColor = PALETTE[arch].stroke;
    item.appendChild(sw);
    item.appendChild(document.createTextNode(arch));
    legend.appendChild(item);
  });
  root.appendChild(legend);

  const finishBar = (item) => {
    item.bar.setAttribute('y', String(item.y));
    item.bar.setAttribute('height', String(item.h));
    item.val.setAttribute('opacity', '1');
  };

  const resetChart = () => {
    playGen += 1;
    root.classList.remove('rb-ready');
    panels.forEach((p) => p.classList.remove('in'));
    animBars.forEach((item) => {
      item.bar.setAttribute('y', String(BASE_Y));
      item.bar.setAttribute('height', '0');
      item.val.setAttribute('opacity', '0');
    });
  };

  const play = () => {
    root.classList.add('rb-ready');
    panels.forEach((p) => p.classList.add('in'));

    if (prefersReduced) {
      animBars.forEach(finishBar);
      return;
    }

    const DUR = 720;
    const gen = playGen;
    const t0 = performance.now();
    const tick = (now) => {
      if (gen !== playGen) return;
      let pending = false;
      animBars.forEach((item) => {
        const local = now - t0 - item.delay * 1000;
        if (local < 0) {
          pending = true;
          return;
        }
        const p = Math.min(1, local / DUR);
        const e = easeOut(p);
        const h = item.h * e;
        item.bar.setAttribute('height', String(h));
        item.bar.setAttribute('y', String(BASE_Y - h));
        if (p > 0.55) {
          const op = Math.min(1, (p - 0.55) / 0.45);
          item.val.setAttribute('opacity', String(op));
        }
        if (p < 1) pending = true;
        else finishBar(item);
      });
      if (pending) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  if (isRecord) {
    const page = root.closest('.page');
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
    }, { threshold: 0.2 });
    io.observe(root);
    if (prefersReduced) play();
  }
})();
