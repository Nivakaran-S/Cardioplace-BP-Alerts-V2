/* Inline-SVG chart primitives. No domain knowledge, no dependencies, no build step.
 *
 * Everything reads colours from CSS custom properties rather than hardcoding them, which is
 * why the theme toggle has to REDRAW the charts rather than restyle them: an SVG attribute
 * resolved at creation time does not follow a variable that changed afterwards.
 */
(function (global) {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function svgEl(tag, attrs, text) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) {
      e.setAttribute(k, attrs[k]);
    }
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  /* Grid, y ticks and the y scale. Returns Y(value) -> pixel. */
  function axes(svg, m, iw, ih, yMin, yMax, fmt, ticks) {
    var grid = cssVar("--grid"), axis = cssVar("--axis"), muted = cssVar("--text-muted");
    var span = (yMax - yMin) || 1;
    function Y(v) { return m.t + ih - ((v - yMin) / span) * ih; }
    for (var i = 0; i <= ticks; i++) {
      var v = yMin + (span * i) / ticks, y = Y(v);
      svg.appendChild(svgEl("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y, stroke: grid,
                                      "stroke-width": 1 }));
      svg.appendChild(svgEl("text", { x: m.l - 7, y: y + 4, "text-anchor": "end",
                                      fill: muted, "font-size": 10.5 }, fmt(v)));
    }
    svg.appendChild(svgEl("line", { x1: m.l, x2: m.l, y1: m.t, y2: m.t + ih, stroke: axis }));
    svg.appendChild(svgEl("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih,
                                    stroke: axis }));
    return Y;
  }

  function xTicks(svg, m, ih, indices, X, labelFn) {
    var muted = cssVar("--text-muted");
    indices.forEach(function (i) {
      svg.appendChild(svgEl("text", { x: X(i), y: m.t + ih + 18, "text-anchor": "middle",
                                      fill: muted, "font-size": 10.5 }, labelFn(i)));
    });
  }

  /* A horizontal reference line (threshold, cut, emergency floor). No-ops off-domain, so a
   * caller never has to check whether the line is visible before asking for it. */
  function refLine(svg, Y, value, m, iw, colour, label, yMin, yMax) {
    if (value === null || value === undefined || !isFinite(value)) return;
    if (value < yMin || value > yMax) return;
    var y = Y(value);
    svg.appendChild(svgEl("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y, stroke: colour,
                                    "stroke-width": 1.5, "stroke-dasharray": "5 4",
                                    opacity: .9 }));
    if (label) {
      svg.appendChild(svgEl("text", { x: m.l + iw - 3, y: y - 5, "text-anchor": "end",
                                      fill: colour, "font-size": 10.5 }, label));
    }
  }

  function shadeBand(svg, x0, x1, m, ih, colour, label) {
    if (x1 <= x0) return;
    svg.appendChild(svgEl("rect", { x: x0, y: m.t, width: x1 - x0, height: ih,
                                    fill: colour, opacity: .09 }));
    if (label) {
      svg.appendChild(svgEl("text", { x: x0 + 5, y: m.t + 13, fill: colour,
                                      "font-size": 10 }, label));
    }
  }

  function path(points) {
    return points.map(function (p, i) {
      return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1);
    }).join(" ");
  }

  function line(svg, points, colour, opts) {
    opts = opts || {};
    svg.appendChild(svgEl("path", { d: path(points), fill: "none", stroke: colour,
                                    "stroke-width": opts.width || 2,
                                    "stroke-dasharray": opts.dash || null,
                                    "stroke-linejoin": "round",
                                    "stroke-linecap": "round" }));
  }

  function dots(svg, points, colour, r) {
    points.forEach(function (p) {
      svg.appendChild(svgEl("circle", { cx: p[0], cy: p[1], r: r || 2.5, fill: colour }));
    });
  }

  /* Keyboard-reachable crosshair. The previous dashboard's hover layer was mouse-only, so
   * every chart was unreadable without a pointer. Arrow keys move the cursor and the focused
   * point is announced through an aria-live region. */
  function hoverLayer(svg, host, m, iw, ih, n, X, Y, valueAt, describe) {
    if (n < 1) return;
    var axis = cssVar("--axis"), s1 = cssVar("--series-1");
    var cross = svgEl("line", { y1: m.t, y2: m.t + ih, stroke: axis, "stroke-width": 1,
                                opacity: 0 });
    var dot = svgEl("circle", { r: 4, fill: s1, opacity: 0 });
    svg.appendChild(cross); svg.appendChild(dot);

    var tip = document.createElement("div");
    tip.setAttribute("role", "status");
    tip.className = "sr-only";
    host.appendChild(tip);

    var hit = svgEl("rect", { x: m.l, y: m.t, width: iw, height: ih, fill: "transparent",
                              tabindex: 0, role: "application",
                              "aria-label": "chart cursor; use the left and right arrow keys" });
    svg.appendChild(hit);

    var cur = -1;
    function show(i) {
      if (i < 0 || i >= n) return;
      var v = valueAt(i);
      if (!v || !isFinite(v.y)) return;
      cur = i;
      var x = X(i), y = Y(v.y);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("opacity", .55);
      dot.setAttribute("cx", x); dot.setAttribute("cy", y); dot.setAttribute("opacity", 1);
      tip.textContent = describe(i, v);
    }
    function hide() {
      cross.setAttribute("opacity", 0); dot.setAttribute("opacity", 0); tip.textContent = "";
    }
    hit.addEventListener("mousemove", function (ev) {
      var box = svg.getBoundingClientRect();
      var sc = box.width / svg.viewBox.baseVal.width;
      var rel = (ev.clientX - box.left) / sc - m.l;
      show(Math.max(0, Math.min(n - 1, Math.round((rel / iw) * (n - 1)))));
    });
    hit.addEventListener("mouseleave", hide);
    hit.addEventListener("blur", hide);
    hit.addEventListener("focus", function () { show(cur < 0 ? n - 1 : cur); });
    hit.addEventListener("keydown", function (ev) {
      if (ev.key === "ArrowLeft") { show(Math.max(0, cur - 1)); ev.preventDefault(); }
      else if (ev.key === "ArrowRight") { show(Math.min(n - 1, cur + 1)); ev.preventDefault(); }
      else if (ev.key === "Home") { show(0); ev.preventDefault(); }
      else if (ev.key === "End") { show(n - 1); ev.preventDefault(); }
    });
  }

  function legendInto(host, items) {
    if (!host) return;
    host.innerHTML = "";
    items.forEach(function (it) {
      var k = document.createElement("span");
      k.className = "k";
      var sw = document.createElement("span");
      sw.className = "sw" + (it.dash ? " dash" : "");
      if (it.colour) sw.style.background = it.dash
        ? "repeating-linear-gradient(90deg," + it.colour + " 0 5px,transparent 5px 9px)"
        : it.colour;
      k.appendChild(sw);
      k.appendChild(document.createTextNode(it.label));
      host.appendChild(k);
    });
  }

  /* Nice round y-domain with padding, so the line never touches the frame. */
  function domain(values, pad) {
    var v = values.filter(function (x) { return isFinite(x); });
    if (!v.length) return [0, 1];
    var lo = Math.min.apply(null, v), hi = Math.max.apply(null, v);
    var p = Math.max((hi - lo) * (pad === undefined ? 0.12 : pad), 1e-6);
    return [lo - p, hi + p];
  }

  global.Charts = { cssVar: cssVar, svgEl: svgEl, axes: axes, xTicks: xTicks,
                    refLine: refLine, shadeBand: shadeBand, line: line, dots: dots,
                    hoverLayer: hoverLayer, legendInto: legendInto, domain: domain,
                    NS: NS };
})(window);
