/**
 * Expert-parallel throughput — grouped bars for prefill and decode (Table 5).
 * Styled to match downstream benchmark mini bar charts.
 */
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const W = 400;
  const H = 280;
  const isRecord = document.documentElement.classList.contains('record');
  /* Record mode: outer CSS supplies the title gap; keep SVG top pad minimal */
  const M = { t: isRecord ? 6 : 14, r: 14, b: 34, l: 52 };
  const TICK_OFF = { xL: 4, dy: 3 };

  const PALETTE = {
    MoE: { fill: '#E08A86', stroke: '#f8cecc' },
    RFMoE: { fill: '#E2C86A', stroke: '#fff2cc' },
  };

  const CHARTS = [
    {
      svgId: 'epPrefillSvg',
      stage: 'Prefill',
      yFmt: (v) => `${(v / 1000).toFixed(0)}k`,
      barValFmt: (v) => `${(v / 1000).toFixed(1)}k`,
      valFmt: (v) => `${v.toLocaleString('en-US', { maximumFractionDigits: 1 })} tok/s`,
      devices: [1, 2, 3, 4],
      series: [
        { name: 'MoE', values: [22346.66, 18034.84, 18558.08, 18355.43] },
        { name: 'RFMoE', values: [22277.77, 22268.24, 21822.29, 21784.77] },
      ],
      yMin: 9000,
      yMax: 27000,
    },
    {
      svgId: 'epDecodeSvg',
      stage: 'Decode',
      yFmt: (v) => v.toFixed(0),
      barValFmt: (v) => v.toFixed(1),
      valFmt: (v) => `${v.toFixed(2)} tok/s`,
      devices: [1, 2, 3, 4],
      series: [
        { name: 'MoE', values: [57.9, 45.65, 45.61, 44.67] },
        { name: 'RFMoE', values: [52.14, 52.05, 50.11, 49.84] },
      ],
      yMin: 10,
      yMax: 70,
    },
  ];

  const plotW = W - M.l - M.r;

  const el = (tag, attrs = {}, parent) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      node.setAttribute(k, v);
    }
    parent.appendChild(node);
    return node;
  };

  const niceTicks = (min, max, count = 4) => {
    const span = max - min || 1;
    const raw = span / Math.max(count - 1, 1);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = Math.ceil(raw / mag) * mag;
    const start = Math.floor(min / step) * step;
    const ticks = [];
    for (let v = start; v <= max + step * 0.01; v += step) {
      if (v >= min - step * 0.01 && v <= max + step * 0.01) ticks.push(+v.toFixed(10));
    }
    if (ticks.length < 2) return [min, max];
    return ticks;
  };

  const buildLegend = () => {
    const legend = document.getElementById('epLegend');
    if (!legend || legend.childElementCount) return;
    legend.removeAttribute('aria-hidden');
    Object.keys(PALETTE).forEach((name) => {
      const pal = PALETTE[name];
      const item = document.createElement('span');
      item.className = 'rb-legend-item';
      const sw = document.createElement('i');
      sw.className = 'rb-swatch';
      sw.style.background = pal.fill;
      sw.style.borderColor = pal.stroke;
      item.appendChild(sw);
      item.appendChild(document.createTextNode(name));
      legend.appendChild(item);
    });
  };

  const easeOut = (p) => 1 - Math.pow(1 - p, 3);
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const root = document.getElementById('epBars');
  let playGen = 0;
  /** @type {{bar: SVGRectElement, val: SVGTextElement, y: number, h: number, delay: number}[]} */
  const animBars = [];
  /** @type {HTMLElement[]} */
  const panels = [];

  const buildChart = (cfg, chartIdx) => {
    const svg = document.getElementById(cfg.svgId);
    if (!svg) return;

    const chart = svg.closest('.ep-chart');
    const tip = chart?.querySelector('.ep-tip');
    if (!chart || !tip) return;

    chart.style.setProperty('--panel-i', String(chartIdx));
    panels.push(chart);

    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const yMin = cfg.yMin;
    const yMax = cfg.yMax;
    const plotTop = M.t;
    const plotH = H - plotTop - M.b;
    const plotBaseY = plotTop + plotH;
    const ySpan = yMax - yMin || 1;
    const yScale = (v) => plotTop + plotH - ((v - yMin) / ySpan) * plotH;
    const tickXLeft = () => M.l - TICK_OFF.xL;
    const panelDelay = chartIdx * 0.12;
    let barIdx = 0;

    const showTip = (html, clientX, clientY) => {
      tip.innerHTML = html;
      const rect = chart.getBoundingClientRect();
      tip.style.left = `${clientX - rect.left}px`;
      tip.style.top = `${clientY - rect.top}px`;
      tip.classList.add('on');
    };
    const hideTip = () => tip.classList.remove('on');

    const yTicks = niceTicks(yMin, yMax);
    yTicks.forEach((v) => {
      const y = yScale(v);
      if (y < plotTop - 1 || y > plotBaseY + 1) return;
      el('line', {
        class: 'rb-grid-line rb-chrome',
        x1: M.l, x2: W - M.r, y1: y, y2: y,
      }, svg);
      const t = el('text', {
        class: 'rb-tick rb-chrome',
        x: tickXLeft(),
        y: y + TICK_OFF.dy,
        'text-anchor': 'end',
      }, svg);
      t.textContent = cfg.yFmt(v);
    });

    const nGroups = cfg.devices.length;
    const nSeries = cfg.series.length;
    const groupW = plotW / nGroups;
    const barGap = 4;
    const barW = Math.min(22, (groupW - barGap * (nSeries + 1)) / nSeries);

    cfg.devices.forEach((device, gi) => {
      const groupCx = M.l + groupW * gi + groupW / 2;
      const groupSpan = nSeries * barW + (nSeries - 1) * barGap;
      let bx = groupCx - groupSpan / 2;

      cfg.series.forEach((s) => {
        const val = s.values[gi];
        const x = bx;
        const y = yScale(val);
        const h = Math.max(0, plotBaseY - y);
        const pal = PALETTE[s.name] || PALETTE.MoE;
        const barDelay = panelDelay + barIdx * 0.045;
        barIdx += 1;

        const bar = el('rect', {
          class: 'rb-bar',
          x,
          y: plotBaseY,
          width: barW,
          height: 0,
          fill: pal.fill,
          stroke: pal.stroke,
          'stroke-width': '1',
          rx: '2',
        }, svg);
        const tipHtml = `<b>${cfg.stage}</b><br>${s.name} · M=${device}: ${cfg.valFmt(val)}`;
        bar.addEventListener('pointerenter', (e) => showTip(tipHtml, e.clientX, e.clientY));
        bar.addEventListener('pointermove', (e) => showTip(tipHtml, e.clientX, e.clientY));
        bar.addEventListener('pointerleave', hideTip);

        const valLab = el('text', {
          class: 'rb-val',
          x: x + barW / 2,
          y: y - 3,
          'text-anchor': 'middle',
          opacity: '0',
        }, svg);
        valLab.textContent = cfg.barValFmt(val);

        animBars.push({ bar, val: valLab, y, h, delay: barDelay, baseY: plotBaseY });

        bx += barW + barGap;
      });

      el('text', {
        class: 'rb-scale rb-chrome',
        x: groupCx,
        y: plotBaseY + 12,
        'text-anchor': 'middle',
      }, svg).textContent = String(device);
    });

    el('line', {
      class: 'rb-baseline rb-chrome',
      x1: M.l, x2: W - M.r, y1: plotBaseY, y2: plotBaseY,
    }, svg);

    el('text', {
      class: 'ep-axis-label rb-chrome',
      x: tickXLeft(),
      y: plotBaseY,
      'text-anchor': 'end',
      'dominant-baseline': 'middle',
    }, svg).textContent = 'tok/s';

    el('text', {
      class: 'ep-axis-label rb-chrome',
      x: M.l + plotW / 2,
      y: H - 4,
      'text-anchor': 'middle',
    }, svg).textContent = 'Devices (M)';
  };

  const finishBar = (item) => {
    item.bar.setAttribute('y', String(item.y));
    item.bar.setAttribute('height', String(item.h));
    item.val.setAttribute('opacity', '1');
  };

  const resetChart = () => {
    playGen += 1;
    if (!root) return;
    root.classList.remove('ep-ready');
    panels.forEach((p) => p.classList.remove('in'));
    animBars.forEach((item) => {
      item.bar.setAttribute('y', String(item.baseY));
      item.bar.setAttribute('height', '0');
      item.val.setAttribute('opacity', '0');
    });
  };

  const play = () => {
    if (!root) return;
    root.classList.add('ep-ready');
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
        item.bar.setAttribute('y', String(item.baseY - h));
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

  CHARTS.forEach((cfg, i) => buildChart(cfg, i));
  buildLegend();

  if (root) {
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
  }
})();
