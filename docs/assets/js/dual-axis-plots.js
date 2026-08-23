/**
 * Dual-axis line plots — μ interpolation and θ sweep.
 * Left y-axis: performance; right y-axis: efficiency (independent scales).
 */
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const W = 400;
  const H = 300;
  const isRecord = document.documentElement.classList.contains('record');
  /* Record mode: outer CSS supplies the title gap; keep SVG top pad minimal */
  const M = { t: isRecord ? 6 : 28, r: 42, b: 32, l: 42 };
  const TICK_OFF = { xL: 6, xR: 6, dy: 3.5 };
  const LEGEND_INSET = 0.1;
  const plotW = W - M.l - M.r;
  const tickXLeft = () => M.l - TICK_OFF.xL;
  const tickXRight = () => W - M.r + TICK_OFF.xR;

  const PLOTS = [
    {
      svgId: 'muSvg',
      perfLabel: 'PPL ↓',
      effLabel: 'throughput ↑',
      effAxisName: 's/s',
      effLegend: 'throughput',
      x: [0, 0.2, 0.5, 0.8, 1.0],
      perf: [28.41, 28.35, 28.34, 28.38, 28.43],
      eff: [645.7, 648.3, 662.3, 643.9, 648.8],
      perfLim: [28.32, 28.44],
      perfTicks: [28.34, 28.38, 28.42],
      effLim: [640, 670],
      effTicks: [645, 655, 665, 675],
      highlight: 2,
      perfFmt: (v) => v.toFixed(2),
      effFmt: (v) => String(Math.round(v)),
      xFmt: (v) => ({ 0: 'TB only', 0.5: 'Balanced', 1: 'EB only' })[v] ?? '',
    },
    {
      svgId: 'thetaSvg',
      perfLabel: 'Avg. ↑',
      effLabel: 'FLOPs ↓',
      x: [0.5, 0.8, 0.9, 1.0, 1.1, 1.2],
      perf: [39.12, 39.55, 39.73, 39.80, 39.98, 39.77],
      eff: [118.41, 105.40, 100.32, 96.21, 92.97, 90.37],
      perfLim: [39, 40.2],
      perfTicks: [39.2, 39.6, 40.0],
      effLim: [87, 123],
      effTicks: [93, 105, 117],
      highlight: 4,
      perfFmt: (v) => v.toFixed(1),
      effFmt: (v) => `${Math.round(v)}M`,
    },
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

  const padRange = (vals, padBottom = 0.12, padTop = 0.12) => {
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    const span = max - min || Math.abs(max) * 0.1 || 1;
    return [min - span * padBottom, max + span * padTop];
  };

  const linePath = (xs, ys, xScale, yScale) => {
    let d = '';
    xs.forEach((x, i) => {
      const px = xScale(x);
      const py = yScale(ys[i]);
      d += `${i ? 'L' : 'M'}${px.toFixed(2)} ${py.toFixed(2)} `;
    });
    return d.trim();
  };

  const easeOut = (p) => 1 - Math.pow(1 - p, 3);
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const root = document.querySelector('#balance .dual-plots');
    /** @type {{path: SVGPathElement, len: number, delay: number, dots: SVGCircleElement[], dashed?: boolean}[]} */
  const animLines = [];
  /** @type {HTMLElement[]} */
  const panels = [];

  const buildPlot = (cfg, plotIdx) => {
    const svg = document.getElementById(cfg.svgId);
    if (!svg) return;

    const chart = svg.closest('.dual-chart');
    if (!chart) return;
    chart.style.setProperty('--panel-i', String(plotIdx));
    panels.push(chart);

    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    const [perfMin, perfMax] = cfg.perfLim ?? padRange(cfg.perf, 0.12, 0.12 + LEGEND_INSET);
    const [effMin, effMax] = cfg.effLim ?? padRange(cfg.eff, 0.12, 0.12 + LEGEND_INSET);

    const xMin = Math.min(...cfg.x);
    const xMax = Math.max(...cfg.x);
    const xPad = (xMax - xMin) * 0.08 || 0.1;

    const plotTop = M.t;
    const plotH = H - plotTop - M.b;

    const defs = el('defs', {}, svg);

    const xScale = (v) => M.l + ((v - (xMin - xPad)) / (xMax - xMin + 2 * xPad)) * plotW;
    const perfScale = (v) => plotTop + plotH - ((v - perfMin) / (perfMax - perfMin)) * plotH;
    const effScale = (v) => plotTop + plotH - ((v - effMin) / (effMax - effMin)) * plotH;

    const perfTicks = cfg.perfTicks ?? niceTicks(perfMin, perfMax);
    const effTicks = cfg.effTicks ?? niceTicks(effMin, effMax);
    const xTicks = cfg.x;
    const panelDelay = plotIdx * 0.14;

    const gGrid = el('g', { class: 'grid dual-chrome' }, svg);
    perfTicks.forEach((v) => {
      const y = perfScale(v);
      el('line', {
        class: 'grid-line',
        x1: M.l, x2: W - M.r, y1: y, y2: y,
      }, gGrid);
    });

    perfTicks.forEach((v) => {
      const y = perfScale(v);
      const t = el('text', {
        class: 'tick tick-left dual-chrome',
        x: tickXLeft(),
        y: y + TICK_OFF.dy,
        'text-anchor': 'end',
      }, svg);
      t.textContent = cfg.perfFmt(v);
    });

    effTicks.forEach((v) => {
      const y = effScale(v);
      const t = el('text', {
        class: 'tick tick-right dual-chrome',
        x: tickXRight(),
        y: y + TICK_OFF.dy,
        'text-anchor': 'start',
      }, svg);
      t.textContent = cfg.effFmt(v);
    });

    xTicks.forEach((v) => {
      const x = xScale(v);
      el('line', {
        class: 'grid-line grid-line-x dual-chrome',
        x1: x, x2: x, y1: plotTop, y2: plotTop + plotH,
      }, gGrid);
      const label = cfg.xFmt ? cfg.xFmt(v) : String(v);
      if (!label) return;
      const atMin = v === xMin;
      const atMax = v === xMax;
      const t = el('text', {
        class: 'tick tick-x dual-chrome',
        x: atMin ? x + 2 : atMax ? x - 2 : x,
        y: plotTop + plotH + 14,
        'text-anchor': atMin ? 'start' : atMax ? 'end' : 'middle',
      }, svg);
      t.textContent = label;
    });

    ['left', 'right', 'bottom', 'top'].forEach((side, i) => {
      const attrs = { class: 'axis-line dual-chrome', 'stroke-width': '1' };
      if (side === 'left') Object.assign(attrs, { x1: M.l, x2: M.l, y1: plotTop, y2: plotTop + plotH });
      if (side === 'right') Object.assign(attrs, { x1: W - M.r, x2: W - M.r, y1: plotTop, y2: plotTop + plotH });
      if (side === 'bottom') Object.assign(attrs, { x1: M.l, x2: W - M.r, y1: plotTop + plotH, y2: plotTop + plotH });
      if (side === 'top') Object.assign(attrs, { x1: M.l, x2: W - M.r, y1: plotTop, y2: plotTop });
      el('line', attrs, svg);
    });

    const perfDots = [];
    const effDots = [];

    cfg.x.forEach((xv, i) => {
      const hi = i === cfg.highlight;
      const r = hi ? 4.5 : 3.2;
      const perfDot = el('circle', {
        class: hi ? 'dot-perf hi dual-dot' : 'dot-perf dual-dot',
        cx: xScale(xv),
        cy: perfScale(cfg.perf[i]),
        r,
        opacity: '0',
      }, svg);
      const effDot = el('circle', {
        class: hi ? 'dot-eff hi dual-dot' : 'dot-eff dual-dot',
        cx: xScale(xv),
        cy: effScale(cfg.eff[i]),
        r,
        opacity: '0',
      }, svg);
      perfDots.push(perfDot);
      effDots.push(effDot);
    });

    const perfPath = el('path', {
      class: 'line-perf dual-line',
      d: linePath(cfg.x, cfg.perf, xScale, perfScale),
      fill: 'none',
      'stroke-width': '2',
    }, svg);

    const effPath = el('path', {
      class: 'line-eff dual-line',
      d: linePath(cfg.x, cfg.eff, xScale, effScale),
      fill: 'none',
      'stroke-width': '2',
      'stroke-dasharray': '6 4',
    }, svg);

    const perfLen = perfPath.getTotalLength();
    perfPath.style.strokeDasharray = String(perfLen);
    perfPath.style.strokeDashoffset = String(perfLen);

    const effClipId = `eff-clip-${cfg.svgId}`;
    const effClip = el('clipPath', { id: effClipId }, defs);
    const effClipRect = el('rect', {
      x: M.l,
      y: plotTop - 4,
      width: 0,
      height: plotH + 8,
    }, effClip);
    effPath.setAttribute('clip-path', `url(#${effClipId})`);

    animLines.push({ path: perfPath, len: perfLen, delay: panelDelay + 0.08, dots: perfDots });
    animLines.push({
      path: effPath,
      clipRect: effClipRect,
      clipW: plotW,
      delay: panelDelay + 0.32,
      dots: effDots,
      dashed: true,
    });

    const perfAxisName = cfg.perfLabel.replace(/\s*[↑↓]\s*$/, '').trim();
    const effAxisName = (cfg.effAxisName ?? cfg.effLabel).replace(/\s*[↑↓]\s*$/, '').trim();
    el('text', {
      class: 'axis-label axis-label-left dual-chrome',
      x: tickXLeft(),
      y: plotTop,
      'text-anchor': 'end',
      'dominant-baseline': 'middle',
    }, svg).textContent = perfAxisName;

    el('text', {
      class: 'axis-label axis-label-right dual-chrome',
      x: tickXRight(),
      y: plotTop,
      'text-anchor': 'start',
      'dominant-baseline': 'middle',
    }, svg).textContent = effAxisName;

    if (cfg.xLabel) {
      el('text', {
        class: 'axis-label axis-label-x dual-chrome',
        x: M.l + plotW / 2,
        y: H - 6,
        'text-anchor': 'middle',
      }, svg).textContent = cfg.xLabel;
    }

    const perfLegend = cfg.perfLabel.trim();
    const effLegend = (cfg.effLegend ?? cfg.effLabel).trim();

    const legendY = plotTop + 8;
    const legendCenterX = M.l + plotW / 2;
    const lineLen = 16;
    const labelGap = 5;
    const itemGap = 22;
    const charW = 6.2;
    const perfLabelW = perfLegend.length * charW;
    const effLabelW = effLegend.length * charW;
    const totalLegendW = (lineLen + labelGap + perfLabelW) + itemGap + (lineLen + labelGap + effLabelW);
    let legendX = legendCenterX - totalLegendW / 2;
    const lg = el('g', { class: 'legend dual-chrome' }, svg);
    el('line', {
      x1: legendX, y1: legendY, x2: legendX + lineLen, y2: legendY,
      class: 'line-perf', 'stroke-width': '2',
    }, lg);
    el('text', {
      class: 'legend-label',
      x: legendX + lineLen + labelGap, y: legendY + 3.5,
    }, lg).textContent = perfLegend;
    legendX += lineLen + labelGap + perfLabelW + itemGap;
    el('line', {
      x1: legendX, y1: legendY, x2: legendX + lineLen, y2: legendY,
      class: 'line-eff', 'stroke-width': '2', 'stroke-dasharray': '6 4',
    }, lg);
    el('text', {
      class: 'legend-label',
      x: legendX + lineLen + labelGap, y: legendY + 3.5,
    }, lg).textContent = effLegend;
  };

  const finishLine = (item) => {
    if (item.dashed && item.clipRect) {
      item.clipRect.setAttribute('width', String(item.clipW));
    } else {
      item.path.style.strokeDashoffset = '0';
    }
    item.dots.forEach((d) => d.setAttribute('opacity', '1'));
  };

  let playGen = 0;

  const resetChart = () => {
    playGen += 1;
    if (root) root.classList.remove('dual-ready');
    panels.forEach((p) => p.classList.remove('in'));
    animLines.forEach((item) => {
      if (item.dashed && item.clipRect) {
        item.clipRect.setAttribute('width', '0');
      } else {
        item.path.style.strokeDashoffset = String(item.len);
        item.path.style.strokeDasharray = String(item.len);
      }
      item.dots.forEach((d) => d.setAttribute('opacity', '0'));
    });
  };

  const play = () => {
    if (root) root.classList.add('dual-ready');
    panels.forEach((p) => p.classList.add('in'));

    if (prefersReduced) {
      animLines.forEach(finishLine);
      return;
    }

    const DUR = 880;
    const gen = playGen;
    const t0 = performance.now();
    const tick = (now) => {
      if (gen !== playGen) return;
      let pending = false;
      animLines.forEach((item) => {
        const local = now - t0 - item.delay * 1000;
        if (local < 0) {
          pending = true;
          return;
        }
        const p = Math.min(1, local / DUR);
        const e = easeOut(p);
        if (item.dashed && item.clipRect) {
          item.clipRect.setAttribute('width', String(item.clipW * e));
        } else {
          item.path.style.strokeDashoffset = String(item.len * (1 - e));
        }
        const dotCount = item.dots.length;
        const revealCount = Math.min(dotCount, Math.ceil(e * dotCount));
        item.dots.forEach((dot, i) => {
          if (i < revealCount) dot.setAttribute('opacity', '1');
        });
        if (p < 1) pending = true;
        else finishLine(item);
      });
      if (pending) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  PLOTS.forEach((cfg, i) => buildPlot(cfg, i));

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
