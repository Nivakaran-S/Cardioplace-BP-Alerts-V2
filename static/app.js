/* Cardioplace BP Alerts -- dashboard controller.
 *
 * Vanilla, no framework, no build step. One IIFE, sections in load order:
 * state -> theme -> form -> renderers -> network -> boot.
 *
 * Every renderer is pure `(data) -> void` and defensive against null, because `paint` calls
 * them inside individual try blocks: one malformed block must degrade to an empty panel, not
 * a blank page. On a clinical dashboard a section that fails silently is bad; a section that
 * takes the emergency banner down with it is much worse.
 */
(function () {
  "use strict";
  var C = window.Charts;
  var $ = function (id) { return document.getElementById(id); };
  var LAST = null, SCHEMA = null, GOV = null;

  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
  function pretty(s) {
    if (!s) return "";
    return String(s).replace(/^RULE_/, "").replace(/_/g, " ").toLowerCase()
      .replace(/^./, function (c) { return c.toUpperCase(); });
  }
  function fmt(v, d) {
    return (v === null || v === undefined || !isFinite(v)) ? "–" : Number(v).toFixed(d || 1);
  }

  /* ------------------------------------------------------------------ theme */
  function applyTheme(mode) {
    document.documentElement.setAttribute("data-theme", mode);
    try { localStorage.setItem("cp-theme", mode); } catch (e) { /* private mode */ }
    $("btn-theme").setAttribute("aria-pressed", mode === "dark" ? "true" : "false");
    // Charts resolve CSS variables at creation time, so a theme change needs a redraw
    // rather than a restyle.
    if (LAST) paint(LAST);
  }

  /* ------------------------------------------------------------------- form */
  function buildChecks(hostId, items, counterId) {
    var host = $(hostId);
    if (!host) return;
    host.innerHTML = "";
    var groups = {};
    (items || []).forEach(function (it) {
      (groups[it.group] = groups[it.group] || []).push(it);
    });
    var names = Object.keys(groups);
    names.forEach(function (g) {
      if (names.length > 1) {
        var h = document.createElement("p");
        h.className = "hint";
        h.style.cssText = "width:100%;margin:6px 0 2px";
        h.textContent = g;
        host.appendChild(h);
      }
      groups[g].forEach(function (it) {
        var lab = document.createElement("label");
        // red_flag comes from the server, which is also what evaluates the rules. The
        // previous dashboard kept its own camelCase copy of this list and never matched.
        lab.className = "chk" + (it.red_flag ? " red" : "");
        lab.innerHTML = '<input type="checkbox" value="' + esc(it.key) + '">'
          + (it.red_flag ? '<span class="dot" aria-hidden="true"></span>' : "")
          + "<span>" + esc(it.label) + "</span>"
          + (it.red_flag ? '<span class="sr-only"> (red flag)</span>' : "");
        host.appendChild(lab);
      });
    });
    if (counterId) {
      host.addEventListener("change", function () {
        $(counterId).textContent = checked(hostId).length;
      });
    }
  }

  function checked(hostId) {
    return Array.prototype.slice
      .call($(hostId).querySelectorAll("input:checked"))
      .map(function (i) { return i.value; });
  }

  function parseReadings(raw) {
    var rows = [], errors = [], seen = {};
    String(raw || "").split(/\r?\n/).forEach(function (line, n) {
      var t = line.trim();
      if (!t || t.charAt(0) === "#") return;
      var p = t.split(/[,;\t]+/).map(function (x) { return x.trim(); });
      if (p.length < 3) { errors.push("line " + (n + 1) + ": need date, SBP, DBP"); return; }
      var d = p[0], sbp = Number(p[1]), dbp = Number(p[2]);
      if (!/^\d{4}-\d{2}-\d{2}/.test(d)) { errors.push("line " + (n + 1) + ": date must be YYYY-MM-DD"); return; }
      if (!isFinite(sbp) || !isFinite(dbp)) { errors.push("line " + (n + 1) + ": SBP/DBP must be numeric"); return; }
      if (sbp - dbp < 10) { errors.push("line " + (n + 1) + ": pulse pressure below 10 mmHg"); return; }
      if (seen[d]) { errors.push("line " + (n + 1) + ": duplicate date " + d + " — merge same-day sessions"); return; }
      seen[d] = 1;
      var r = { date: d, sbp: Math.round(sbp), dbp: Math.round(dbp) };
      if (p[3] && isFinite(Number(p[3]))) r.pulse = Number(p[3]);
      if (p[4] && isFinite(Number(p[4]))) r.weight = Number(p[4]);
      if (p[5] && isFinite(Number(p[5]))) r.idwg = Number(p[5]);
      rows.push(r);
    });
    rows.sort(function (a, b) { return a.date < b.date ? -1 : 1; });
    return { rows: rows, errors: errors };
  }

  /* Deterministic sample so the page looks the same on every load and a demo is repeatable. */
  function sampleReadings() {
    var seed = 20260730, out = [], base = 138, d = new Date("2026-04-06T00:00:00Z");
    function rnd() { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; }
    for (var i = 0; i < 42; i++) {
      base += (rnd() - 0.44) * 4.2;
      var sbp = Math.round(Math.max(105, Math.min(196, base + (rnd() - 0.5) * 9)));
      var dbp = Math.round(Math.max(58, Math.min(112, sbp * 0.55 + (rnd() - 0.5) * 7)));
      out.push(d.toISOString().slice(0, 10) + ", " + sbp + ", " + dbp + ", "
               + Math.round(64 + rnd() * 22) + ", " + (68 + rnd() * 2.4).toFixed(1));
      d = new Date(d.getTime() + (i % 3 === 2 ? 3 : 2) * 86400000);
    }
    return out.join("\n");
  }

  function updateCount() {
    var p = parseReadings($("readings").value);
    var n = p.rows.length;
    var note = n + " session" + (n === 1 ? "" : "s");
    if (n < 7) note += " — below the 7-reading cold-start floor; no forecast will be issued";
    else if (n < 48) note += " — below the 48-reading steady state; low-confidence badge";
    $("reading-count").textContent = note;
  }

  /* -------------------------------------------------------------- renderers */

  function renderNotices(d) {
    var host = $("notices");
    host.innerHTML = "";
    function add(cls, html) {
      var e = document.createElement("p");
      e.className = "notice " + cls;
      e.innerHTML = html;
      host.appendChild(e);
    }
    if (d.degraded && d.degraded.model_loaded === false) {
      add("warn", "<strong>No trained model is loaded.</strong> "
        + esc(d.degraded.reason || "") + " The rule engine below is unaffected — it is the "
        + "safety-critical layer and needs no model. " + esc(d.degraded.remedy || ""));
    }
    if (d.truncated) {
      add("", "Charts and the backtest use the most recent "
        + d.truncated.enrichment_readings + " of " + d.truncated.submitted
        + " readings. The forecast itself uses all of them.");
    }
    var st = d.staleness || {};
    if (st.days_since_last_reading > (st.max_forecast_age_days || 14)) {
      add("warn", "The most recent reading is " + fmt(st.days_since_last_reading, 0)
        + " days old, beyond the " + (st.max_forecast_age_days || 14)
        + "-day limit. The personalised threshold still stands; no forecast is issued.");
    }
  }

  function renderBanner(d) {
    var pers = d.personalisation || {}, ew = d.early_warning || {};
    var eng = (d.rule_engine || {}).current || {};
    var pa = (d.predicted_alert || {}).horizons || [];
    var firstFire = pa.filter(function (h) { return h.fired; })[0];
    var cls, icon, title, detail;

    if (eng.is_emergency) {
      cls = "banner-critical"; icon = "⚠";
      title = "Rule engine fired an emergency on the latest reading";
      detail = pretty(eng.rule_id) + ". The emergency floor is never personalised and the "
             + "engine is authoritative here.";
    } else if (ew.flagged) {
      cls = "banner-critical"; icon = "⚠";
      title = "Early-warning detector flagged this patient";
      detail = "Score " + fmt(ew.score, 3) + " at or above the " + ew.budget_pct
             + "% cut of " + fmt(ew.cut, 3) + ", roughly " + fmt(ew.est_lead_days, 1)
             + " days of lead time.";
    } else if (firstFire) {
      cls = "banner-watch"; icon = "◆";
      title = pretty(firstFire.tier) + " forecast in about " + fmt(firstFire.days_ahead, 1) + " days";
      detail = "Predicted SBP " + fmt(firstFire.sbp) + " mmHg would fire "
             + pretty(firstFire.rule_id) + " if the trend holds.";
    } else if (eng.fired) {
      cls = "banner-watch"; icon = "◆";
      title = pretty(eng.tier) + " on the latest reading";
      detail = pretty(eng.rule_id) + ". No further breach forecast in the coming horizons.";
    } else if (d.confidence_tier === "stale") {
      cls = ""; icon = "○"; title = "History too old for a forecast"; detail = d.note || "";
    } else if (d.confidence_tier === "cold_start") {
      cls = ""; icon = "○"; title = "Cold start — no forecast issued"; detail = d.note || "";
    } else if (d.confidence_tier === "no_model") {
      cls = ""; icon = "○"; title = "Rule engine only — no model loaded";
      detail = "Nothing fired on the latest reading.";
    } else {
      cls = "banner-good"; icon = "✓"; title = "Nothing due";
      detail = "The engine fired nothing on the latest reading, the forecast stays below the "
             + "personalised threshold of " + fmt(pers.threshold) + " mmHg, and the detector "
             + "is below its cut.";
    }
    $("banner").className = "banner " + cls;
    $("banner-icon").textContent = icon;
    $("banner-title").textContent = title;
    $("banner-detail").textContent = detail;
    var t = d.timings || {};
    $("banner-meta").textContent = (d.n_observations || 0) + " readings"
      + (t.predict_ms != null ? " · predict " + t.predict_ms + " ms" : "")
      + (t.total_ms != null ? " · total " + t.total_ms + " ms" : "")
      + (d.model_version ? " · " + d.model_version : "");
  }

  function renderTiles(d) {
    var pers = d.personalisation || {}, eng = (d.rule_engine || {}).current || {};
    var chip = $("v-engine-tier");
    chip.textContent = eng.tier ? pretty(eng.tier) : "No alert";
    chip.className = "chip " + (eng.is_emergency ? "chip-critical"
                     : eng.fired ? "chip-warning" : "chip-good");
    $("v-engine-note").textContent = eng.rule_id
      ? pretty(eng.rule_id) + (eng.axis_mode === "PERSONALIZED" ? " · personalised" : "")
      : (eng.gate_reason && eng.gate_reason !== "NONE"
         ? "gated: " + pretty(eng.gate_reason) : "nothing fired on this reading");

    $("v-threshold").textContent = pers.threshold != null ? fmt(pers.threshold) : "–";
    $("v-threshold-note").textContent = pers.offset != null
      ? "offset " + (pers.offset > 0 ? "+" : "") + fmt(pers.offset) + " mmHg"
        + (pers.capped ? " — bound by the governance cap" : "")
        + (pers.cohort_key ? " · cohort " + pers.cohort_key : "")
        + (pers.n_warm != null ? " · " + pers.n_warm + " readings" : "")
      : (d.confidence_tier === "no_model" ? "no model loaded" : "");

    var labels = { cold_start: "Cold start", bootstrapping: "Bootstrapping",
                   steady: "Steady", stale: "Stale", no_model: "No model" };
    $("v-tier").textContent = labels[d.confidence_tier] || d.confidence_tier || "–";
    $("v-tier-note").textContent = (d.n_observations || 0) + " readings"
      + (d.note ? " — " + d.note : "");

    var ew = d.early_warning, c = $("v-ew-chip"), m = $("v-ew-meter");
    if (!ew) {
      c.textContent = "not issued"; c.className = "chip chip-muted";
      m.hidden = true;
      $("v-ew-note").textContent = d.confidence_tier === "no_model"
        ? "requires a trained model" : "below the cold-start floor";
    } else {
      c.textContent = ew.flagged ? "⚠ Flagged" : "✓ Not flagged";
      c.className = "chip " + (ew.flagged ? "chip-critical" : "chip-good");
      m.hidden = false;
      var pct = Math.max(2, Math.min(100, (ew.score / (ew.cut || 1)) * 80));
      var fill = $("v-ew-fill");
      fill.style.width = pct + "%";
      fill.className = "meter-fill" + (ew.flagged ? " over" : "");
      m.setAttribute("aria-valuenow", Math.round(pct));
      $("v-ew-note").textContent = "score " + fmt(ew.score, 3) + " of cut " + fmt(ew.cut, 3)
        + " · est. lead " + fmt(ew.est_lead_days, 1) + " d";
    }
  }

  function renderForecastChart(d) {
    var host = $("chart");
    host.innerHTML = "";
    var hist = d.history || [];
    if (!hist.length) return;
    var fc = (d.forecast && d.forecast.sbp) || {};
    var pts = Object.keys(fc).map(function (k) { return fc[k]; })
      .sort(function (a, b) { return a.steps_ahead - b.steps_ahead; });

    var s1 = C.cssVar("--series-1"), s2 = C.cssVar("--series-2");
    var warn = C.cssVar("--status-warning"), crit = C.cssVar("--status-critical");
    var muted = C.cssVar("--text-muted");
    var thr = (d.personalisation || {}).threshold, floor = d.emergency_floor_mmHg;

    var vals = hist.map(function (h) { return h.sbp; })
      .concat(pts.map(function (p) { return p.point; }))
      .concat(pts.map(function (p) { return p.lo80; }))
      .concat(pts.map(function (p) { return p.hi80; }))
      .concat([thr]).filter(function (v) { return v != null && isFinite(v); });
    var dom = C.domain(vals, 0.1);
    var yMin = Math.floor(dom[0] / 10) * 10, yMax = Math.ceil(dom[1] / 10) * 10;

    var W = 900, H = 300, m = { t: 16, r: 18, b: 34, l: 46 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    var n = hist.length + pts.length;
    var X = function (i) { return m.l + (n <= 1 ? 0 : (i / (n - 1)) * iw); };
    var svg = C.svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
                               preserveAspectRatio: "xMidYMid meet",
                               "aria-label": hist.length + " observed sessions and "
                                 + pts.length + " forecast horizons" });
    var Y = C.axes(svg, m, iw, ih, yMin, yMax, function (v) { return String(Math.round(v)); }, 4);

    C.refLine(svg, Y, floor, m, iw, crit, "emergency floor " + fmt(floor, 0), yMin, yMax);
    C.refLine(svg, Y, thr, m, iw, warn, "threshold " + fmt(thr), yMin, yMax);

    var obs = hist.map(function (h, i) { return [X(i), Y(h.sbp)]; });
    C.line(svg, obs, s1, { width: 2 });
    C.dots(svg, obs.slice(-1), s1, 3);

    if (pts.length) {
      var start = obs[obs.length - 1];
      var fpts = [start].concat(pts.map(function (p, i) {
        return [X(hist.length + i), Y(p.point)];
      }));
      C.line(svg, fpts, s2, { width: 2, dash: "6 4" });
      pts.forEach(function (p, i) {
        var x = X(hist.length + i);
        if (p.lo80 != null && p.hi80 != null) {
          svg.appendChild(C.svgEl("line", { x1: x, x2: x, y1: Y(p.lo80), y2: Y(p.hi80),
                                            stroke: s2, "stroke-width": 1.5, opacity: .5 }));
          [p.lo80, p.hi80].forEach(function (v) {
            svg.appendChild(C.svgEl("line", { x1: x - 4, x2: x + 4, y1: Y(v), y2: Y(v),
                                              stroke: s2, "stroke-width": 1.5, opacity: .6 }));
          });
        }
        svg.appendChild(C.svgEl("circle", { cx: x, cy: Y(p.point), r: 3.5, fill: s2 }));
      });
    }

    var idx = [0, Math.floor(hist.length / 2), hist.length - 1]
      .filter(function (v, i, a) { return v >= 0 && a.indexOf(v) === i; });
    C.xTicks(svg, m, ih, idx, X, function (i) {
      return (hist[i] && hist[i].ts ? hist[i].ts.slice(5) : "");
    });

    C.hoverLayer(svg, host, m, iw, ih, hist.length, X, Y,
      function (i) { return { y: hist[i].sbp }; },
      function (i, v) {
        return hist[i].ts + ": " + v.y + " over " + hist[i].dbp + " mmHg";
      });

    host.appendChild(svg);
    C.legendInto($("chart-legend"), [
      { colour: s1, label: "Observed SBP" },
      { colour: s2, dash: true, label: "Forecast with 80% band" },
      { colour: warn, label: "Personalised threshold" },
      { colour: crit, label: "Emergency floor (never personalised)" }]);
  }

  function renderForecastTable(d) {
    var fc = (d.forecast && d.forecast.sbp) || {}, thr = (d.personalisation || {}).threshold;
    var tb = $("forecast-table").querySelector("tbody");
    tb.innerHTML = "";
    Object.keys(fc).sort(function (a, b) { return fc[a].steps_ahead - fc[b].steps_ahead; })
      .forEach(function (k) {
        var f = fc[k], delta = thr != null ? f.point - thr : null;
        var tr = document.createElement("tr");
        tr.innerHTML = "<td>+" + f.steps_ahead + " sessions</td>"
          + '<td class="num">' + fmt(f.point) + "</td>"
          + '<td class="num">' + (f.lo80 != null ? fmt(f.lo80) + " – " + fmt(f.hi80) : "–") + "</td>"
          + '<td class="num">' + fmt(f.days_ahead_est, 1) + "</td>"
          + "<td>" + (delta == null ? "–" : (delta >= 0 ? "▲ " + fmt(delta) + " over"
                                                        : "▼ " + fmt(Math.abs(delta)) + " under")) + "</td>";
        tb.appendChild(tr);
      });
    var band = Object.keys(fc).map(function (k) { return fc[k]; })
      .filter(function (f) { return f.interval_basis; })[0];
    $("interval-basis").textContent = band ? "Interval basis: " + band.interval_basis
      : (Object.keys(fc).length ? "No conformal interval on these horizons."
                                : "No forecast issued.");
    $("forecast-card").hidden = !Object.keys(fc).length && !(d.history || []).length;
  }

  function renderPredicted(d) {
    var pa = d.predicted_alert || {}, host = $("predicted-alerts");
    host.innerHTML = "";
    var hs = pa.horizons || [];
    if (!hs.length) {
      host.innerHTML = '<p class="hint">No forecast issued, so no predicted alerts.</p>';
      $("predicted-note").textContent = pa.basis || "";
      return;
    }
    hs.forEach(function (h) {
      var cls = h.is_emergency ? "chip-critical" : h.fired ? "chip-warning" : "chip-good";
      var card = document.createElement("div");
      card.className = "horizon";
      card.innerHTML =
        '<p class="horizon-when">in ~' + fmt(h.days_ahead, 1) + " days · +" + h.steps_ahead + " sessions</p>"
        + '<p class="horizon-bp">' + fmt(h.sbp)
        + (h.dbp != null ? " / " + fmt(h.dbp) : "") + '<span class="unit">mmHg</span></p>'
        + '<p class="horizon-band">' + (h.lo80 != null ? "80% " + fmt(h.lo80) + " – " + fmt(h.hi80)
                                                       : "no interval") + "</p>"
        + '<span class="chip ' + cls + '">' + esc(h.tier ? pretty(h.tier) : "No alert") + "</span>"
        + (h.rule_id ? '<p class="horizon-rule">' + esc(pretty(h.rule_id)) + "</p>" : "");
      host.appendChild(card);
    });
    $("predicted-note").textContent = [pa.basis, pa.symptom_note].filter(Boolean).join(" ");
  }

  function renderEngine(d) {
    var eng = d.rule_engine;
    if (!eng) { $("engine-card").hidden = true; return; }
    $("engine-card").hidden = false;
    if (eng.error) {
      $("engine-note").textContent = "Rule engine unavailable: " + eng.error;
      return;
    }
    var tl = $("engine-timeline");
    tl.innerHTML = "";
    (eng.history || []).forEach(function (h) {
      var s = document.createElement("span");
      s.className = "tl" + (h.tier ? " tl-" + h.tier : "");
      s.title = h.ts + (h.tier ? " · " + pretty(h.tier) + " · " + pretty(h.rule_id)
                               : " · no alert");
      tl.appendChild(s);
    });
    $("engine-count").textContent = eng.fired_count + " of " + (eng.history || []).length
      + " readings fired" + (eng.emergency_count ? ", " + eng.emergency_count + " emergency" : "");

    var tb = $("engine-table").querySelector("tbody");
    tb.innerHTML = "";
    (eng.history || []).slice().reverse().filter(function (h) { return h.fired; })
      .slice(0, 12).forEach(function (h) {
        var tr = document.createElement("tr");
        tr.innerHTML = "<td>" + esc(h.ts) + "</td><td>" + esc(pretty(h.tier)) + "</td><td>"
                     + esc(pretty(h.rule_id)) + "</td>";
        tb.appendChild(tr);
      });
    var counts = Object.keys(eng.tier_counts || {}).map(function (k) {
      return pretty(k) + " " + eng.tier_counts[k];
    }).join(" · ");
    $("engine-note").textContent = (counts ? counts + ". " : "")
      + (eng.note || "")
      + " The engine is deterministic: whatever you enter above is evaluated immediately."
      + (eng.personalised ? "" : " Personalisation is inactive — no model threshold available.");
  }

  function renderAnomaly(d) {
    var an = d.anomaly, panel = $("anomaly-panel"), host = $("anomaly-chart");
    host.innerHTML = "";
    if (!an || !an.points || an.points.length < 2) { panel.hidden = true; return; }
    panel.hidden = false;
    var pts = an.points, cut = an.cut;
    var s1 = C.cssVar("--series-1"), warn = C.cssVar("--status-warning");
    var crit = C.cssVar("--status-critical"), muted = C.cssVar("--text-muted");

    var vals = pts.map(function (p) { return p.score; });
    if (cut != null) vals.push(cut);
    var dom = C.domain(vals, 0.15);
    var W = 900, H = 250, m = { t: 16, r: 60, b: 34, l: 52 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b, n = pts.length;
    var X = function (i) { return m.l + (n <= 1 ? 0 : (i / (n - 1)) * iw); };
    var svg = C.svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
                               preserveAspectRatio: "xMidYMid meet",
                               "aria-label": "detector score across " + n + " sessions" });
    var Y = C.axes(svg, m, iw, ih, dom[0], dom[1],
                   function (v) { return v.toFixed(2); }, 4);

    var warmN = pts.filter(function (p) { return p.warmup; }).length;
    if (warmN > 0) C.shadeBand(svg, m.l, X(Math.max(warmN - 1, 0)), m, ih, muted, "warm-up");
    C.refLine(svg, Y, cut, m, iw, warn, "cut " + fmt(cut, 3), dom[0], dom[1]);
    C.line(svg, pts.map(function (p, i) { return [X(i), Y(p.score)]; }), s1, { width: 1.8 });
    pts.forEach(function (p, i) {
      if (p.flagged) svg.appendChild(C.svgEl("circle", { cx: X(i), cy: Y(p.score), r: 3.5,
                                                         fill: crit }));
    });
    C.xTicks(svg, m, ih, [0, Math.floor(n / 2), n - 1].filter(function (v, i, a) {
      return a.indexOf(v) === i;
    }), X, function (i) { return pts[i].ts ? pts[i].ts.slice(5) : ""; });
    C.hoverLayer(svg, host, m, iw, ih, n, X, Y,
      function (i) { return { y: pts[i].score }; },
      function (i, v) { return pts[i].ts + ": score " + v.y.toFixed(3)
                        + (pts[i].flagged ? ", flagged" : ""); });
    host.appendChild(svg);

    $("anomaly-sub").textContent = an.event_definition || "";
    C.legendInto($("anomaly-legend"), [
      { colour: s1, label: "Detector score" },
      { colour: warn, dash: true, label: "Alert cut at the " + (an.budget_pct || 5) + "% budget" },
      { colour: crit, label: "Flagged session" }]);
    $("anomaly-note").textContent = an.n_flagged + " of " + an.n_settled
      + " settled sessions crossed the cut. The first " + an.warmup_readings
      + " sessions are warm-up: the score's own inputs are not yet defined there.";
  }

  function renderBacktest(d) {
    var bt = d.backtest, panel = $("backtest-panel"), host = $("backtest-chart");
    host.innerHTML = "";
    var tb = $("backtest-table").querySelector("tbody");
    tb.innerHTML = "";
    if (!bt || !bt.horizons || !bt.horizons.length) { panel.hidden = true; return; }
    panel.hidden = false;
    bt.horizons.forEach(function (h) {
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>+" + h.horizon + " sessions</td>"
        + '<td class="num">' + fmt(h.days_ahead, 1) + "</td>"
        + '<td class="num">' + h.n + "</td>"
        + '<td class="num">' + fmt(h.mae, 2) + "</td>"
        + '<td class="num">' + Math.round(h.within_10 * 100) + "%</td>";
      tb.appendChild(tr);
    });

    var first = bt.horizons[0];
    var rows = (bt.series || {})["h" + first.horizon] || [];
    if (rows.length < 2) { host.innerHTML = ""; return; }
    var s1 = C.cssVar("--series-1"), s2 = C.cssVar("--series-2"), crit = C.cssVar("--status-critical");
    var vals = rows.map(function (r) { return r.actual; })
      .concat(rows.map(function (r) { return r.predicted; }));
    var dom = C.domain(vals, 0.12);
    var yMin = Math.floor(dom[0] / 10) * 10, yMax = Math.ceil(dom[1] / 10) * 10;
    var W = 900, H = 260, m = { t: 16, r: 18, b: 34, l: 46 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b, n = rows.length;
    var X = function (i) { return m.l + (n <= 1 ? 0 : (i / (n - 1)) * iw); };
    var svg = C.svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
                               preserveAspectRatio: "xMidYMid meet",
                               "aria-label": "actual against predicted over " + n + " sessions" });
    var Y = C.axes(svg, m, iw, ih, yMin, yMax, function (v) { return String(Math.round(v)); }, 4);
    C.line(svg, rows.map(function (r, i) { return [X(i), Y(r.actual)]; }), s1, { width: 2 });
    C.line(svg, rows.map(function (r, i) { return [X(i), Y(r.predicted)]; }), s2,
           { width: 1.8, dash: "5 4" });

    var worst = rows.reduce(function (a, b, i) {
      return Math.abs(b.error) > Math.abs(rows[a].error) ? i : a;
    }, 0);
    svg.appendChild(C.svgEl("circle", { cx: X(worst), cy: Y(rows[worst].actual), r: 4,
                                        fill: crit }));
    svg.appendChild(C.svgEl("text", { x: X(worst), y: Y(rows[worst].actual) - 9,
                                      "text-anchor": "middle", fill: crit, "font-size": 10.5 },
                            "worst miss " + fmt(rows[worst].error) + " mmHg"));
    C.xTicks(svg, m, ih, [0, Math.floor(n / 2), n - 1].filter(function (v, i, a) {
      return a.indexOf(v) === i;
    }), X, function (i) { return rows[i].ts ? rows[i].ts.slice(5) : ""; });
    host.appendChild(svg);
    C.legendInto($("backtest-legend"), [
      { colour: s1, label: "Actual" },
      { colour: s2, dash: true, label: "Predicted +" + first.horizon + " sessions ahead" }]);
    $("backtest-note").textContent = bt.caption || "";
  }

  function renderSymptoms(d) {
    var sr = d.symptom_risk, panel = $("symptom-panel");
    if (!sr) { panel.hidden = true; return; }
    panel.hidden = false;
    // Non-dismissible and first: these labels were generated, and a reader who misses that
    // will read a synthetic hazard model as a clinical prediction.
    $("symptom-warning").textContent = sr.warning || "";
    var tb = $("symptom-table").querySelector("tbody");
    tb.innerHTML = "";
    if (!sr.available) {
      $("symptom-note").textContent = sr.reason || "";
      return;
    }
    (sr.items || []).forEach(function (it) {
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>" + esc(it.label) + (it.red_flag
          ? ' <span class="chip chip-critical">red flag</span>' : "") + "</td>"
        + "<td>" + esc(it.mechanism || "–") + "</td>"
        + '<td class="num">' + fmt(it.prob * 100, 1) + "%</td>"
        + '<td class="num">' + (it.cut != null ? fmt(it.cut * 100, 1) + "%" : "–") + "</td>"
        + "<td>" + (it.flagged ? '<span class="chip chip-warning">above cut</span>'
                               : '<span class="chip chip-muted">below</span>') + "</td>";
      tb.appendChild(tr);
    });
    $("symptom-note").textContent = sr.n_flagged + " of " + (sr.items || []).length
      + " symptom heads are above their operating cut.";
  }

  function renderChain(d) {
    // The forecast-conditioned answer, deliberately a SEPARATE panel from renderSymptoms.
    // They answer different questions -- risk at the next session from observed history, vs
    // risk further out given the predicted trajectory -- and only the first is what the heads
    // were trained to do. Replacing one with the other would hide that.
    var ch = d.symptom_chained, panel = $("chain-panel");
    if (!ch) { panel.hidden = true; return; }
    panel.hidden = false;
    $("chain-warning").textContent = (d.symptom_risk && d.symptom_risk.warning) || "";
    var tb = $("chain-table").querySelector("tbody");
    tb.innerHTML = "";
    if (!ch.available) {
      $("chain-limits").textContent = "";
      $("chain-note").textContent = ch.reason || "";
      return;
    }
    (ch.items || []).forEach(function (it) {
      var gap = (it.jensen_gap || 0) * 100;
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>" + esc(pretty(it.key)) + (it.red_flag
          ? ' <span class="chip chip-critical">red flag</span>' : "") + "</td>"
        + '<td class="num">' + esc(String(it.sessions_ahead)) + "</td>"
        + "<td>session " + esc(String(it.conditioned_through_session))
        + " (SBP " + fmt(it.conditioned_through_sbp, 1) + ")</td>"
        + '<td class="num">' + fmt(it.prob * 100, 1) + "%</td>"
        + '<td class="num">' + (gap >= 0 ? "+" : "") + fmt(gap, 1) + " pp</td>"
        + "<td>" + esc(it.mechanism || "–") + "</td>";
      tb.appendChild(tr);
    });
    // Both limits are load-bearing and neither is visible in the numbers.
    $("chain-limits").textContent = (ch.session_1_note || "") + " "
      + (ch.conditioning_note || "");
    $("chain-note").textContent = (ch.cut_note || "") + " " + (ch.reach_note || "");
  }

  function renderGovernance(d) {
    var g = (GOV || {});
    var host = $("gov-list");
    host.innerHTML = "";
    var t = d.timings || {};
    var items = [
      ["Emergency floor", fmt(d.emergency_floor_mmHg || g.emergency_floor_mmHg, 0) + " mmHg (never personalised)"],
      ["Population threshold", fmt(g.population_threshold_mmHg, 0) + " mmHg"],
      ["Offset caps", "−" + fmt(g.offset_cap_tighten, 0) + " / +" + fmt(g.offset_cap_loosen, 0) + " mmHg"],
      ["Alert budget", fmt(g.alert_budget_pct, 0) + "% (detector only)"],
      ["Warn window", (g.warn_window || "–") + " sessions"],
      ["Event quantile", "p" + Math.round((g.event_quantile || 0.95) * 100) + " of the patient’s own SBP"],
      ["Stale-forecast limit", (g.stale_forecast_max_days || 14) + " days"],
      ["Model version", d.model_version || "none loaded"],
      ["Latency", (t.predict_ms != null ? t.predict_ms + " ms core / " + t.total_ms + " ms total"
                                        : "–")]
    ];
    items.forEach(function (kv) {
      var e = document.createElement("div");
      e.innerHTML = '<div class="k">' + esc(kv[0]) + '</div><div class="v">' + esc(kv[1]) + "</div>";
      host.appendChild(e);
    });
    var e2 = document.createElement("div");
    e2.innerHTML = '<div class="k">Journaling input</div><div class="v" style="font-size:11.5px">'
      + "idwg, sbp_drop and uf_total are absent from journaled sessions but present in "
      + "training. Those features are served as missing, never zero-filled.</div>";
    host.appendChild(e2);
  }

  function paint(d) {
    LAST = d;
    $("empty-state").hidden = true;
    $("output").hidden = false;
    [["notices", renderNotices], ["banner", renderBanner], ["tiles", renderTiles],
     ["forecast chart", renderForecastChart], ["forecast table", renderForecastTable],
     ["predicted", renderPredicted], ["engine", renderEngine], ["anomaly", renderAnomaly],
     ["backtest", renderBacktest], ["symptoms", renderSymptoms],
     ["chain", renderChain],
     ["governance", renderGovernance]].forEach(function (pair) {
      try { pair[1](d); }
      catch (e) { console.error("render " + pair[0] + " failed", e); }
    });
  }

  /* ---------------------------------------------------------------- network */
  function showError(msg) {
    var b = $("form-error");
    b.textContent = msg || "";
    b.hidden = !msg;
  }

  async function predict() {
    showError("");
    var parsed = parseReadings($("readings").value);
    if (parsed.errors.length) { showError(parsed.errors.slice(0, 3).join(" · ")); return; }
    if (!parsed.rows.length) { showError("Enter at least one reading."); return; }

    var rows = parsed.rows;
    rows[rows.length - 1].symptoms = checked("symptoms");
    rows[rows.length - 1].position = $("position").value;
    rows[rows.length - 1].n_meas = Number($("n-meas").value) || 2;

    var pt = $("provider-target").value;
    var body = {
      patient_id: $("patient-id").value || "demo",
      profile: {
        age: Number($("age").value) || 65,
        is_male: Number($("sex").value),
        is_dm: Number($("dm").value),
        is_pregnant: Number($("pregnant").value),
        hf_type: $("hf-type").value,
        conditions: checked("conditions"),
        medications: checked("medications"),
        provider_target: pt === "" ? null : Number(pt),
        missed_3d: Number($("missed-3d").value) || 0,
        adherence_7d: Number($("adherence").value),
        step_offset: Number($("step-offset").value) || 0
      },
      readings: rows,
      // Off unless the user ticks the box. Measured on a 60-reading history this block is
      // 5.0 s of a 7.1 s request -- 70% of it -- because it rebuilds the causal feature
      // frame once per horizon and again per quadrature node. EnrichFlags defaults it off
      // for exactly that reason, and requesting it unconditionally here overrode that
      // decision and made every page load pay for a panel most users are not reading.
      enrich: { symptom_chained: $("opt-chain") ? $("opt-chain").checked : false }
    };

    var btn = $("btn-predict"), results = document.querySelector(".results");
    btn.disabled = true; btn.textContent = "Working…";
    results.setAttribute("aria-busy", "true");
    var ctl = new AbortController();
    var timer = setTimeout(function () { ctl.abort(); }, 120000);
    try {
      var res = await fetch("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body), signal: ctl.signal });
      var out = await res.json();
      if (!res.ok) { showError(out.detail || ("Request failed (" + res.status + ")")); return; }
      paint(out);
    } catch (e) {
      showError(e.name === "AbortError" ? "The request timed out after 120 s."
                                        : "Could not reach the API: " + e.message);
    } finally {
      clearTimeout(timer);
      btn.disabled = false; btn.textContent = "Get advisory";
      results.setAttribute("aria-busy", "false");
    }
  }

  var pollTimer = null;
  function renderTrain(s) {
    var chip = $("train-state");
    if (s.running) {
      chip.textContent = "running"; chip.className = "chip chip-warning";
      $("train-note").textContent = "Started " + (s.started_at || "") + " (pid " + s.pid + ").";
    } else if (s.returncode === 0) {
      chip.textContent = "finished"; chip.className = "chip chip-good";
      $("train-note").textContent = "Completed " + (s.finished_at || "") + ".";
    } else if (s.returncode != null) {
      chip.textContent = "failed (" + s.returncode + ")"; chip.className = "chip chip-critical";
      $("train-note").textContent = "Exited " + (s.finished_at || "") + " with code " + s.returncode + ".";
    } else {
      chip.textContent = "idle"; chip.className = "chip chip-muted";
    }
    var pre = $("train-tail");
    if (s.tail && s.tail.length) {
      pre.hidden = false;
      pre.textContent = s.tail.join("\n");
      pre.scrollTop = pre.scrollHeight;
    }
  }

  async function pollTrain() {
    try {
      var s = await (await fetch("/api/train/status")).json();
      renderTrain(s);
      if (!s.running) {
        clearInterval(pollTimer); pollTimer = null;
        refreshModel();
      }
    } catch (e) { /* keep polling */ }
  }

  async function refreshModel() {
    var chip = $("model-chip");
    try {
      var h = await (await fetch("/api/health")).json();
      if (h.training && h.training.warning) $("train-warning").textContent = h.training.warning;
      if (h.training && h.training.running) {
        renderTrain(h.training);
        if (!pollTimer) pollTimer = setInterval(pollTrain, 3000);
      }
      if (!h.model_loaded) {
        chip.textContent = "no model"; chip.className = "chip chip-critical";
        chip.title = h.detail || "";
        return;
      }
      var m = await (await fetch("/api/model")).json();
      GOV = m.governance || {};
      chip.textContent = m.model_version;
      chip.className = "chip chip-good";
      chip.title = m.n_features + " features · " + JSON.stringify(m.shipped || {})
                 + " · sklearn " + (m.sklearn_runtime || "?");
    } catch (e) {
      chip.textContent = "API unreachable"; chip.className = "chip chip-critical";
    }
  }

  /* ------------------------------------------------------------------- boot */
  (async function boot() {
    try { applyTheme(localStorage.getItem("cp-theme")
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")); }
    catch (e) { applyTheme("light"); }

    try {
      SCHEMA = await (await fetch("/api/schema")).json();
      buildChecks("conditions", SCHEMA.conditions, "cond-count");
      buildChecks("medications", SCHEMA.medications, "med-count");
      buildChecks("symptoms", SCHEMA.symptoms, null);
    } catch (e) {
      $("symptoms").innerHTML = '<p class="hint">Could not load the field vocabulary.</p>';
    }
    $("readings").value = sampleReadings();
    updateCount();
    refreshModel();
  })();

  $("btn-predict").addEventListener("click", predict);
  $("btn-sample").addEventListener("click", function () {
    $("readings").value = sampleReadings(); updateCount(); showError("");
  });
  $("btn-clear").addEventListener("click", function () {
    $("readings").value = ""; updateCount(); showError("");
  });
  $("readings").addEventListener("input", updateCount);
  $("btn-theme").addEventListener("click", function () {
    applyTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });
  $("btn-train").addEventListener("click", async function () {
    if (!confirm("Start a full training run?\n\n" + ($("train-warning").textContent || ""))) return;
    var out = await (await fetch("/api/train", { method: "POST" })).json();
    $("train-note").textContent = out.detail || "";
    if (!pollTimer) pollTimer = setInterval(pollTrain, 3000);
    pollTrain();
  });
  $("btn-train-cancel").addEventListener("click", async function () {
    var out = await (await fetch("/api/train/cancel", { method: "POST" })).json();
    $("train-note").textContent = out.detail || "";
  });
})();
