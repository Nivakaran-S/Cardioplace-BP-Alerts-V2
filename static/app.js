/* Cardioplace BP Alerts — dashboard controller.
 *
 * One IIFE, no build step, no dependencies. Every renderer is pure and null-defensive: the
 * API returns a deliberately open payload where blocks are absent rather than empty (no
 * forecast on a cold start, no interval at most horizons, no detector before warm-up), so a
 * renderer that assumes its block exists takes the whole page down with it. `paint` runs each
 * section in its own try, for the same reason.
 *
 * Colours come from CSS custom properties, never literals. That is also why the theme toggle
 * REDRAWS the charts rather than restyling them: an SVG attribute resolved at creation time
 * does not follow a variable that changed afterwards.
 */
(function () {
  "use strict";

  var C = window.Charts;
  var LAST = null;               // last payload, so the theme toggle can redraw from it
  var VOCAB = null;

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function fmt(v, d) {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    return Number(v).toFixed(d === undefined ? 0 : d);
  }
  function pretty(s) {
    if (!s) return "–";
    var t = String(s).replace(/^RULE_/, "").replace(/_/g, " ").toLowerCase();
    return t.charAt(0).toUpperCase() + t.slice(1);
  }

  /* Tier codes are engine identifiers, not prose. Sentence-casing `BP_LEVEL_1_HIGH` yields
   * "Bp level 1 high", which reads like a bug; these are the clinician-facing names. Anything
   * not listed falls back to `pretty` rather than rendering blank, so a tier added to the
   * engine still shows something legible. */
  var TIER_LABEL = {
    BP_LEVEL_1_HIGH: "Stage 1 high", BP_LEVEL_1_LOW: "Stage 1 low",
    BP_LEVEL_2: "Stage 2 high", BP_LEVEL_2_SYMPTOM_OVERRIDE: "Stage 2, symptom override",
    EMERGENCY: "Emergency", HYPOTENSIVE: "Hypotensive", ELDERLY_LOW: "Low, elderly",
    NARROW_PP: "Narrow pulse pressure", WIDE_PP: "Wide pulse pressure",
    CONTRAINDICATION: "Contraindication", AFIB_RVR: "AF with rapid rate"
  };
  function tierName(t) {
    if (!t) return "Clear";
    return TIER_LABEL[t] || pretty(t);
  }
  /* Anything at or above stage 2 is red; a stage-1 breach is amber. Driven by the tier code,
   * not by a substring match on its label. */
  function tierKind(t, isEmergency) {
    if (isEmergency || t === "EMERGENCY") return "critical";
    if (!t) return "muted";
    return /BP_LEVEL_2|ANGIO|CONTRAINDICATION/.test(t) ? "critical" : "warning";
  }
  function shortDate(iso) {
    var d = new Date(iso);
    return isNaN(d) ? String(iso)
      : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }
  function clear(el) { while (el && el.firstChild) el.removeChild(el.firstChild); }

  var ICON = {
    critical: '<path d="M12 3 1.7 21h20.6z"/><path d="M12 10v4M12 17.5h.01" stroke-width="2"/>',
    watch:    '<circle cx="12" cy="12" r="9"/><path d="M12 7v6M12 16.5h.01"/>',
    good:     '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.6 2.6L16 9.5"/>',
    info:     '<circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v4h1"/>'
  };
  function glyph(kind) {
    return '<svg class="glyph" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
           'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" ' +
           'aria-hidden="true">' + (ICON[kind] || ICON.info) + "</svg>";
  }
  /* Status is never colour alone: every pill carries an icon and a word. */
  function pill(kind, label) {
    var mark = { good: "m5 8.5 2 2 4.5-4.5", critical: "M8 3 1 15h14z",
                 warning: "M8 3 1 15h14z", muted: "" }[kind] || "";
    var svg = mark ? '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" ' +
                     'stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="' +
                     mark + '"/></svg>' : "";
    return '<span class="pill ' + kind + '">' + svg + esc(label) + "</span>";
  }

  // ------------------------------------------------------------------- theme

  function applyTheme(mode) {
    document.documentElement.setAttribute("data-theme", mode || "");
    $("btn-theme").setAttribute("aria-pressed", mode === "dark" ? "true" : "false");
    try { localStorage.setItem("cp-theme", mode || ""); } catch (e) { /* private mode */ }
    if (LAST) paint(LAST);          // redraw, do not restyle -- see the file header
  }
  $("btn-theme").addEventListener("click", function () {
    var now = document.documentElement.getAttribute("data-theme");
    var dark = now === "dark" ||
      (now !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
    applyTheme(dark ? "light" : "dark");
  });

  // -------------------------------------------------------------------- boot

  /* 40 sessions, because the `_z` and `_slope30` features need a 30-session window and a
   * shorter sample leaves them NaN -- the demo would then understate what the model does. */
  var SAMPLE = (function () {
    var out = [], d = new Date(2026, 3, 1);
    for (var i = 0; i < 40; i++) {
      var sbp = 138 + (i % 7) * 3 - (i % 3) + Math.round(i / 8);
      out.push(d.toISOString().slice(0, 10) + ", " + sbp + ", " + (78 + (i % 5)) +
               ", " + (74 + (i % 9)) +
               ", w=" + (73.4 + (i % 6) * 0.3).toFixed(1) +
               ", idwg=" + (1.9 + (i % 6) * 0.15).toFixed(2) +
               ", meds=" + (i % 5 === 0 ? "n" : "y") +
               ", uf=" + (2.2 + (i % 4) * 0.1).toFixed(1) + ", hrs=4");
      d.setDate(d.getDate() + 2);
    }
    return out.join("\n");
  })();

  function checkboxes(host, items, name) {
    clear(host);
    items.forEach(function (it) {
      var l = document.createElement("label");
      l.className = "check";
      l.innerHTML = '<input type="checkbox" name="' + name + '" value="' + esc(it.key) + '">' +
                    "<span>" + esc(it.label) + "</span>";
      host.appendChild(l);
    });
  }

  async function boot() {
    $("readings").value = SAMPLE;
    try {
      VOCAB = await (await fetch("/api/schema")).json();
      checkboxes($("conditions"), VOCAB.conditions || [], "cond");
      checkboxes($("medications"), VOCAB.medications || [], "med");
      checkboxes($("symptoms"), VOCAB.symptoms || [], "sym");
    } catch (e) { /* the form still works with the free-text fields alone */ }
    refreshHealth();
  }

  async function refreshHealth() {
    var chip = $("model-chip"), text = $("model-chip-text");
    try {
      var h = await (await fetch("/api/health")).json();
      chip.className = "chip " + (h.model_loaded ? "is-live" : "is-degraded");
      text.textContent = h.model_loaded
        ? (h.model_version || "model loaded")
        : "rule engine only";
      chip.title = h.model_loaded
        ? "Serving " + (h.model_version || "") + " on scikit-learn " + (h.sklearn_runtime || "")
        : (h.detail || "no model on disk; the rule engine is unaffected");
    } catch (e) {
      chip.className = "chip"; text.textContent = "offline";
    }
  }

  // ----------------------------------------------------------------- request

  /* `date, sbp, dbp[, pulse]` positionally, then any number of `key=value` tokens.
   *
   * The extra fields are per-SESSION, not per-patient, and that is the whole point: weight
   * and same-day adherence change between sessions, and the model was fitted on their lagged
   * and rolling forms. Supplying them as a profile constant would be a different -- and
   * wrong -- statement. Positional-only would have needed nine columns on every line with
   * commas holding the empty ones; keyed tokens let a user give what they have.
   *
   * An unknown key is an ERROR, not an ignored token. The schema forbids unknown fields for
   * the same reason: a mistyped `weight=` that silently did nothing would produce a
   * confident forecast built on a feature the user believes they supplied. */
  var TOKENS = {
    w: ["weight", "kg"], weight: ["weight", "kg"],
    idwg: ["idwg", "kg"],
    uf: ["uf_total", "L"], hrs: ["session_hours", "h"], drop: ["sbp_drop", "mmHg"]
  };

  function parseReadings(text) {
    var rows = [];
    text.split("\n").forEach(function (raw, i) {
      var line = raw.trim();
      if (!line || line[0] === "#") return;
      var where = "line " + (i + 1) + ": ";
      var parts = line.split(/[,;\t]+/).map(function (x) { return x.trim(); }).filter(Boolean);
      var pos = parts.filter(function (x) { return x.indexOf("=") < 0; });
      var kv = parts.filter(function (x) { return x.indexOf("=") >= 0; });

      if (pos.length < 3) throw new Error(where + "expected date, systolic, diastolic");
      var r = { date: pos[0], sbp: Math.round(Number(pos[1])), dbp: Math.round(Number(pos[2])) };
      if (!isFinite(r.sbp) || !isFinite(r.dbp)) {
        throw new Error(where + "systolic and diastolic must be numbers");
      }
      if (pos.length > 3) {
        if (!isFinite(Number(pos[3]))) throw new Error(where + "pulse must be a number");
        r.pulse = Number(pos[3]);
      }

      kv.forEach(function (tok) {
        var eq = tok.indexOf("=");
        var k = tok.slice(0, eq).trim().toLowerCase(), v = tok.slice(eq + 1).trim();
        if (k === "meds") {
          if (!/^(y|yes|n|no|1|0)$/i.test(v)) {
            throw new Error(where + "meds= must be y or n, got " + JSON.stringify(v));
          }
          r.took_all_meds = /^(y|yes|1)$/i.test(v);
          return;
        }
        if (k === "sym") {
          // `+`-joined so the token survives the comma split that separates fields.
          r.symptoms = v.split("+").map(function (s) { return s.trim(); }).filter(Boolean);
          return;
        }
        var spec = TOKENS[k];
        if (!spec) {
          throw new Error(where + "unknown field " + JSON.stringify(k) + ". Known: " +
                          Object.keys(TOKENS).concat(["meds", "sym"]).join(", "));
        }
        if (!isFinite(Number(v))) {
          throw new Error(where + k + "= must be a number in " + spec[1]);
        }
        r[spec[0]] = Number(v);
      });
      rows.push(r);
    });
    if (!rows.length) throw new Error("no readings entered");
    return rows;
  }

  function picked(name) {
    return Array.prototype.slice
      .call(document.querySelectorAll('input[name="' + name + '"]:checked'))
      .map(function (el) { return el.value; });
  }
  function num(id) {
    var v = $(id).value;
    return v === "" ? null : Number(v);
  }

  function buildRequest() {
    var rows = parseReadings($("readings").value);
    // Symptoms are per-READING in the schema, not per-profile: they describe what the patient
    // felt at a measurement. The checkboxes ask "at the latest reading", so they attach to the
    // last row only -- back-filling would invent a symptom record that was never reported. A
    // `sym=` token on that line already said the same thing, so it wins over the checkboxes.
    var syms = picked("sym");
    if (syms.length && !rows[rows.length - 1].symptoms) {
      rows[rows.length - 1].symptoms = syms;
    }

    var profile = {
      age: num("age") || 65, is_male: Number($("sex").value),
      is_dm: Number($("dm").value), is_pregnant: Number($("pregnant").value),
      hf_type: $("hf-type").value,
      conditions: picked("cond"), medications: picked("med"),
      missed_3d: num("missed-3d") || 0,
      adherence_7d: (num("adherence") === null ? 100 : num("adherence")) / 100,
      step_offset: num("step-offset") || 0
    };
    if (num("provider-target") !== null) profile.provider_target = num("provider-target");
    if (num("dryweight") !== null) profile.dryweight = num("dryweight");
    if ($("first-dialysis").value) profile.first_dialysis = $("first-dialysis").value;

    return {
      patient_id: $("patient-id").value || "demo",
      profile: profile,
      readings: rows,
      // Off unless asked: this block rebuilds the feature frame per horizon and per
      // quadrature node, which measured at 5.0 s of a 7.1 s request.
      enrich: { symptom_chained: $("opt-chain") ? $("opt-chain").checked : false }
    };
  }

  function showError(msg) {
    var e = $("error");
    e.textContent = msg; e.hidden = false;
  }

  async function predict() {
    $("error").hidden = true;
    var body;
    try { body = buildRequest(); }
    catch (err) { showError(err.message); return; }

    var btn = $("btn-predict"), results = document.querySelector(".results");
    btn.disabled = true; btn.textContent = "Assessing…";
    results.setAttribute("aria-busy", "true");
    var ctl = new AbortController();
    var timer = setTimeout(function () { ctl.abort(); }, 120000);
    try {
      var res = await fetch("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body), signal: ctl.signal
      });
      var out = await res.json();
      if (!res.ok) { showError(out.detail || ("Request failed (" + res.status + ")")); return; }
      LAST = out;
      paint(out);
    } catch (err) {
      showError(err.name === "AbortError"
        ? "The request timed out after 120 seconds."
        : "Could not reach the API: " + err.message);
    } finally {
      clearTimeout(timer);
      btn.disabled = false; btn.textContent = "Assess";
      results.setAttribute("aria-busy", "false");
      refreshHealth();
    }
  }
  $("btn-predict").addEventListener("click", predict);

  // ---------------------------------------------------------------- renderers

  /* Clinical priority order. An emergency outranks a detector flag, which outranks a
   * forecast breach, which outranks a rule that fired on today's reading. */
  function renderBanner(d) {
    var pers = d.personalisation || {}, ew = d.early_warning || {};
    var eng = (d.rule_engine || {}).current || {};
    var horizons = (d.predicted_alert || {}).horizons || [];
    var firstFire = horizons.filter(function (h) { return h.fired; })[0];
    var kind = "good", title = "Nothing due", detail = "";

    if (eng.is_emergency) {
      kind = "critical";
      title = "Emergency on the latest reading";
      detail = pretty(eng.rule_id) + ". The emergency floor is never personalised, and the " +
               "rule engine is authoritative here.";
    } else if (ew.flagged) {
      kind = "critical";
      title = "Early-warning signal raised";
      detail = "Score " + fmt(ew.score, 3) + " at or above the " + fmt(ew.cut, 3) +
               " cut, roughly " + fmt(ew.est_lead_days, 0) + " days of lead time.";
    } else if (firstFire) {
      kind = "watch";
      title = tierName(firstFire.tier) + " forecast in about " +
              fmt(firstFire.days_ahead, 0) + " days";
      detail = "Predicted systolic " + fmt(firstFire.sbp, 0) + " mmHg would fire " +
               pretty(firstFire.rule_id) + " if the trend holds.";
    } else if (eng.fired) {
      kind = "watch";
      title = tierName(eng.tier) + " on the latest reading";
      detail = pretty(eng.rule_id) + ". No further breach is forecast.";
    } else if (d.confidence_tier === "no_model") {
      kind = "info"; title = "Rule engine only";
      detail = "No model is loaded. Nothing fired on the latest reading.";
    } else if (d.confidence_tier === "stale") {
      kind = "info"; title = "History too old for a forecast"; detail = d.note || "";
    } else if (d.confidence_tier === "cold_start") {
      kind = "info"; title = "Cold start — no forecast issued"; detail = d.note || "";
    } else {
      detail = "The engine fired nothing, and the forecast stays below the personalised " +
               "threshold of " + fmt(pers.threshold, 0) + " mmHg.";
    }
    var b = $("banner");
    b.className = "banner is-" + kind;
    b.innerHTML = glyph(kind) + "<div><h2>" + esc(title) + "</h2><p>" + esc(detail) + "</p></div>";
  }

  function tile(k, v, unit, sub, accent) {
    return '<div class="tile' + (accent ? " accent-" + accent : "") + '">' +
           '<div class="k">' + esc(k) + "</div>" +
           '<div class="v">' + esc(v) + (unit ? '<span class="u">' + esc(unit) + "</span>" : "") +
           "</div>" + (sub ? '<div class="sub">' + esc(sub) + "</div>" : "") + "</div>";
  }

  function renderTiles(d) {
    var host = $("tiles");
    var pers = d.personalisation || {}, ew = d.early_warning || {};
    var sbp = ((d.forecast || {}).sbp || {}).h0 || {};
    var bt = (d.backtest || {}).horizons || [];
    var out = [];

    if (sbp.point !== undefined) {
      out.push(tile("Next session", fmt(sbp.point, 0), "mmHg",
                    "in about " + fmt(sbp.days_ahead_est, 0) + " days"));
    }
    if (pers.threshold != null) {
      out.push(tile("Alert threshold", fmt(pers.threshold, 0), "mmHg",
                    (pers.offset >= 0 ? "+" : "") + fmt(pers.offset, 1) +
                    " mmHg vs the population 140"));
    }
    if (ew.score != null) {
      out.push(tile("Early-warning score", fmt(ew.score, 2), "",
                    ew.flagged ? "above the " + fmt(ew.cut, 2) + " cut"
                               : "below the " + fmt(ew.cut, 2) + " cut",
                    ew.flagged ? "critical" : "good"));
    }
    if (bt.length) {
      out.push(tile("Typical error", fmt(bt[0].mae, 1), "mmHg",
                    fmt(bt[0].within_10 * 100, 0) + "% within 10 mmHg, next session"));
    }
    out.push(tile("Readings", fmt(d.n_observations, 0), "",
                  (d.confidence_tier || "").replace("_", " ")));
    host.innerHTML = out.join("");
    host.hidden = !out.length;
  }

  /* Observed systolic and diastolic, the forecast continuing from the last reading, the
   * personalised threshold and the emergency floor. One y-axis in mmHg -- never two. */
  function renderTrend(d) {
    var card = $("trend-card"), svg = $("trend-chart");
    var hist = d.history || [];
    var fsbp = (d.forecast || {}).sbp || {}, fdbp = (d.forecast || {}).dbp || {};
    card.hidden = hist.length < 2;
    if (card.hidden) return;
    clear(svg);
    Array.prototype.slice.call(svg.parentNode.querySelectorAll(".tip,[role=status]"))
      .forEach(function (n) { n.remove(); });

    var fkeys = Object.keys(fsbp).sort();
    var sbpObs = hist.map(function (r) { return r.sbp; });
    var dbpObs = hist.map(function (r) { return r.dbp; });
    var fPts = fkeys.map(function (k) { return fsbp[k].point; });
    var pers = (d.personalisation || {}).threshold;
    var floor = (d.governance || {}).emergency_floor_mmHg;

    // The right margin is a LABEL GUTTER, not padding: direct labels live outside the plot
    // so they cannot land on the marks they name.
    var GUT = 62;
    var W = 940, H = 320, m = { l: 46, r: 16 + GUT, t: 14, b: 30 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);

    var n = hist.length + fkeys.length;
    var all = sbpObs.concat(dbpObs, fPts, [pers, floor].filter(function (x) { return x != null; }));
    var dom = C.domain(all, 0.10);
    var yMin = dom[0], yMax = dom[1];

    /* Three forecast points among forty observed ones get 3/43 of the width on a plain index
     * scale -- about sixty pixels for the part of the chart the reader actually came for. The
     * forecast is given a fixed share instead, so it stays legible however long the history
     * is. The scale break is not hidden: the shaded band marks exactly where it happens. */
    var fShare = fkeys.length ? 0.24 : 0;
    var iwH = iw * (1 - fShare);
    function X(i) {
      if (i < hist.length) {
        return m.l + (hist.length <= 1 ? iwH / 2 : (i / (hist.length - 1)) * iwH);
      }
      return m.l + iwH + ((i - hist.length + 1) / fkeys.length) * (iw - iwH);
    }
    var Y = C.axes(svg, m, iw, ih, yMin, yMax, function (v) { return fmt(v, 0); }, 4);

    // The forecast region, so "measured" and "predicted" are visually separable. Shaded in a
    // NEUTRAL, because the band is a region of the x-axis, not a series -- painting it in a
    // categorical hue would imply it carries an identity of its own.
    if (fkeys.length) {
      C.shadeBand(svg, X(hist.length - 1), X(n - 1), m, ih, C.cssVar("--axis"), "forecast");
    }
    C.refLine(svg, Y, floor, m, iw, C.cssVar("--critical"),
              "emergency floor " + fmt(floor, 0), yMin, yMax, "left");
    C.refLine(svg, Y, pers, m, iw, C.cssVar("--ink-muted"),
              "alert threshold " + fmt(pers, 0), yMin, yMax, "left");

    var s1 = C.cssVar("--series-1"), s3 = C.cssVar("--series-3");
    C.line(svg, dbpObs.map(function (v, i) { return [X(i), Y(v)]; }), s3, { width: 2 });
    C.line(svg, sbpObs.map(function (v, i) { return [X(i), Y(v)]; }), s1, { width: 2 });

    /* Colour follows the ENTITY, so the forecast keeps its signal's hue and is marked
     * predicted by the dash. Giving the forecast a hue of its own would say systolic-measured
     * and systolic-forecast are two different things; they are one thing, half of it observed.
     * The join starts at the last measurement so there is no gap at the handover. */
    function drawForecast(per, obs, colour) {
      var ks = Object.keys(per || {}).sort()
        .filter(function (k) { return per[k] && isFinite(per[k].point); });
      if (!ks.length || !obs.length) return ks;
      var pts = [[X(hist.length - 1), Y(obs[obs.length - 1])]].concat(
        ks.map(function (k, i) { return [X(hist.length + i), Y(per[k].point)]; }));
      C.line(svg, pts, colour, { width: 2, dash: "6 4" });
      C.dots(svg, pts.slice(1), colour, 3.5);
      // The interval exists at one horizon only (only ("sbp", h1) has a fitted band). Drawn
      // where it exists rather than implied everywhere by a smooth ribbon -- a band spanning
      // horizons that were never given one would be a claim the bundle does not support.
      ks.forEach(function (k, i) {
        var nd = per[k];
        if (nd.lo80 == null || nd.hi80 == null) return;
        svg.appendChild(C.svgEl("line", {
          x1: X(hist.length + i), x2: X(hist.length + i),
          y1: Y(nd.hi80), y2: Y(nd.lo80), stroke: colour, "stroke-width": 6,
          "stroke-linecap": "round", opacity: .22
        }));
      });
      return ks;
    }
    drawForecast(fsbp, sbpObs, s1);
    drawForecast(fdbp, dbpObs, s3);

    /* Direct labels in the gutter, at the height each series ends on -- the relief rule for
     * the aqua series, which measures 2.74:1 on the light surface, and quicker to read than a
     * legend hunt. Nudged apart if the two series end within a label's height of each other. */
    function lastY(per, obs) {
      var ks = Object.keys(per || {}).sort();
      var v = ks.length ? per[ks[ks.length - 1]].point : obs[obs.length - 1];
      return Y(v);
    }
    if (sbpObs.length && dbpObs.length) {
      var ys = lastY(fsbp, sbpObs), yd = lastY(fdbp, dbpObs);
      if (Math.abs(ys - yd) < 13) { ys -= 7; yd += 7; }
      [[ys, s1, "systolic"], [yd, s3, "diastolic"]].forEach(function (t) {
        svg.appendChild(C.svgEl("text", {
          x: m.l + iw + 8, y: t[0] + 4, fill: t[1], "font-size": 11.5, "font-weight": 600
        }, t[2]));
      });
    }

    C.xTicks(svg, m, ih, [0, Math.floor(hist.length / 2), hist.length - 1].filter(
      function (v, i, a) { return a.indexOf(v) === i && v >= 0; }),
      X, function (i) { return shortDate(hist[i].ts); });

    C.hoverLayer(svg, svg.parentNode, m, iw, ih, n, X, Y,
      function (i) {
        return i < hist.length ? { y: hist[i].sbp, r: hist[i] }
                               : { y: fsbp[fkeys[i - hist.length]].point, f: true };
      },
      function (i, v) {
        return i < hist.length
          ? shortDate(hist[i].ts) + " — " + fmt(hist[i].sbp, 0) + "/" + fmt(hist[i].dbp, 0) + " mmHg"
          : "forecast " + fmt(v.y, 0) + " mmHg, session " + (i - hist.length + 1) + " ahead";
      });

    C.legendInto($("trend-legend"), [
      { label: "Systolic", colour: s1 },
      { label: "Diastolic", colour: s3 },
      { label: "Forecast (dashed)", colour: C.cssVar("--ink-muted"), dash: true }
    ]);
    $("trend-hint").textContent = hist.length + " sessions";
    $("trend-desc").textContent =
      "Systolic and diastolic over " + hist.length + " sessions. To the right of the last " +
      "measurement both signals continue as a dashed forecast; the vertical band on the " +
      "systolic forecast is the 80% interval, shown only at the horizon where one is fitted.";

    // Table view -- the relief rule for the aqua series, and the accessible equivalent of the
    // chart. It is also where the interval and the coherence verdict live, neither of which
    // the plot can carry legibly.
    var NAME = { sbp: "Systolic", dbp: "Diastolic", idwg: "Weight gain" };
    var tb = $("forecast-table").querySelector("tbody");
    clear(tb);
    ["sbp", "dbp", "idwg"].forEach(function (sig) {
      var per = (d.forecast || {})[sig];
      if (!per) return;
      Object.keys(per).sort().forEach(function (k) {
        var nd = per[k], co = nd.coherence || {};
        var unit = sig === "idwg" ? " kg" : " mmHg";
        var tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + NAME[sig] + "</td>" +
          '<td class="num">' + fmt(nd.days_ahead_est, 0) + " d</td>" +
          '<td class="num">' + fmt(nd.point, 1) + unit + "</td>" +
          '<td class="num">' + (nd.lo80 != null
            ? fmt(nd.lo80, 0) + " – " + fmt(nd.hi80, 0)
            : '<span class="dim">not fitted here</span>') + "</td>" +
          "<td>" + (co.ok === undefined ? "" : co.ok
            ? pill("good", "coherent")
            : pill("critical", (co.violations || [])[0] || "check failed")) + "</td>";
        tb.appendChild(tr);
      });
    });

    /* A signal that is in the bundle but could not be scored on this input must SAY so. If it
     * simply vanished from the table, an absent weight-gain forecast would be indistinguishable
     * from a model that never had one. */
    var un = d.forecast_unavailable, note = $("forecast-missing");
    if (un && (un.signals || []).length) {
      note.hidden = false;
      note.textContent = un.signals.map(function (s) { return NAME[s] || s; }).join(", ") +
        (un.signals.length > 1 ? " could not be scored" : " could not be scored") +
        " on this input. " + (un.note || "");
    } else { note.hidden = true; }
  }

  function renderOutlook(d) {
    var card = $("outlook-card");
    var hz = (d.predicted_alert || {}).horizons || [];
    card.hidden = !hz.length;
    if (card.hidden) return;
    var tb = $("outlook-table").querySelector("tbody");
    clear(tb);
    hz.forEach(function (h) {
      var kind = h.fired ? tierKind(h.tier, h.is_emergency) : "good";
      var word = h.fired ? tierName(h.tier) : "Nothing fires";
      var tr = document.createElement("tr");
      tr.innerHTML = '<td class="num">' + fmt(h.days_ahead, 0) + " d</td>" +
                     '<td class="num">' + fmt(h.sbp, 0) +
                     (h.dbp != null ? " / " + fmt(h.dbp, 0) : "") + "</td>" +
                     "<td>" + pill(kind, word) +
                     (h.fired ? ' <span class="caption">' + esc(pretty(h.rule_id)) + "</span>" : "") +
                     "</td>";
      tb.appendChild(tr);
    });
    $("outlook-note").textContent =
      "Each forecast is re-evaluated by the same rule engine that judges a real reading.";
  }

  function renderRisk(d) {
    var card = $("risk-card"), ew = d.early_warning || {}, an = d.anomaly || {};
    card.hidden = ew.score == null;
    if (card.hidden) return;

    /* The detector score is NOT bounded to [0,1] -- `d_forecast_level` returns 1.08 on this
     * fixture. Clamping to 1 would pin the bar at full and hide exactly the thing worth
     * seeing: how far past the cut it sits. The track scales to the data instead. */
    var hi = Math.max(1, ew.score || 0, ew.cut || 0) * 1.06;
    var meter = $("risk-meter");
    meter.className = "meter" + (ew.flagged ? " over" : "");
    meter.querySelector(".fill").style.width =
      (Math.max(0, Math.min(1, (ew.score || 0) / hi)) * 100).toFixed(1) + "%";
    meter.querySelector(".cut").style.left =
      (Math.max(0, Math.min(1, (ew.cut || 0) / hi)) * 100).toFixed(1) + "%";
    meter.setAttribute("role", "meter");
    meter.setAttribute("aria-valuenow", ew.score);
    meter.setAttribute("aria-valuemin", 0);
    meter.setAttribute("aria-valuemax", hi.toFixed(2));
    meter.setAttribute("aria-label",
      "early-warning score " + fmt(ew.score, 2) + " against a cut of " + fmt(ew.cut, 2));
    $("risk-scale-max").textContent = fmt(hi, 1);
    $("risk-cut-label").textContent = "cut " + fmt(ew.cut, 2) +
      " · score " + fmt(ew.score, 2);
    $("risk-hint").textContent = ew.detector || "";
    $("risk-note").textContent =
      (ew.event_definition ? ew.event_definition.charAt(0).toUpperCase() +
        ew.event_definition.slice(1) + ". " : "") +
      "The cut is set so roughly " + fmt(ew.budget_pct, 0) + "% of sessions are flagged.";

    var pts = (an.points || []).filter(function (p) { return !p.warmup && isFinite(p.score); });
    var svg = $("risk-chart");
    clear(svg);
    Array.prototype.slice.call(svg.parentNode.querySelectorAll(".tip,[role=status]"))
      .forEach(function (n) { n.remove(); });
    if (pts.length < 2) { $("risk-desc").textContent = ""; return; }

    var W = 620, H = 190, m = { l: 40, r: 14, t: 14, b: 26 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var dom = C.domain(pts.map(function (p) { return p.score; }).concat([an.cut || 0]), 0.15);
    var yMin = Math.max(0, dom[0]), yMax = dom[1];
    function X(i) { return m.l + (i / (pts.length - 1)) * iw; }
    var Y = C.axes(svg, m, iw, ih, yMin, yMax, function (v) { return fmt(v, 2); }, 2);
    C.refLine(svg, Y, an.cut, m, iw, C.cssVar("--critical"), "cut", yMin, yMax);
    C.line(svg, pts.map(function (p, i) { return [X(i), Y(p.score)]; }),
           C.cssVar("--series-1"), { width: 2 });
    var hot = pts.map(function (p, i) { return p.flagged ? [X(i), Y(p.score)] : null; })
                 .filter(Boolean);
    if (hot.length) C.dots(svg, hot, C.cssVar("--critical"), 3.5);
    C.xTicks(svg, m, ih, [0, pts.length - 1], X, function (i) { return shortDate(pts[i].ts); });
    C.hoverLayer(svg, svg.parentNode, m, iw, ih, pts.length, X, Y,
      function (i) { return { y: pts[i].score }; },
      function (i, v) {
        return shortDate(pts[i].ts) + " — score " + fmt(v.y, 3) +
               (pts[i].flagged ? ", flagged" : "");
      });
    $("risk-desc").textContent =
      an.n_flagged + " of " + an.n_settled + " settled sessions scored above the cut.";
  }

  function renderEngine(d) {
    var card = $("engine-card"), eng = d.rule_engine || {};
    var hist = eng.history || [], readings = d.history || [];
    card.hidden = !hist.length;
    if (card.hidden) return;

    var byDate = {};
    readings.forEach(function (r) { byDate[String(r.ts).slice(0, 10)] = r; });

    var tb = $("engine-table").querySelector("tbody");
    clear(tb);
    // Newest first: the latest verdict is the one being acted on.
    hist.slice().reverse().slice(0, 14).forEach(function (h) {
      var r = byDate[String(h.ts).slice(0, 10)] || {};
      var kind = h.fired ? tierKind(h.tier, h.is_emergency) : "muted";
      var word = tierName(h.tier);
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>" + esc(shortDate(h.ts)) + "</td>" +
                     '<td class="num">' + (r.sbp != null ? fmt(r.sbp, 0) + "/" + fmt(r.dbp, 0) : "–") + "</td>" +
                     "<td>" + pill(kind, word) + "</td>" +
                     '<td class="caption">' + (h.rule_id ? esc(pretty(h.rule_id)) : "") + "</td>";
      tb.appendChild(tr);
    });
    $("engine-hint").textContent = eng.fired_count + " of " + hist.length + " sessions fired";
    $("engine-note").textContent = eng.note || "";
  }

  /* The backtest is the most useful honest number on the page: the shipped forecaster
   * replayed over this patient's own history, scored against what actually happened. */
  function renderAccuracy(d) {
    var card = $("accuracy-card"), bt = d.backtest || {};
    var hz = bt.horizons || [], series = (bt.series || {}).h1 || [];
    card.hidden = !hz.length;
    if (card.hidden) return;

    var tb = $("accuracy-table").querySelector("tbody");
    clear(tb);
    hz.forEach(function (h) {
      var tr = document.createElement("tr");
      tr.innerHTML = '<td class="num">' + fmt(h.days_ahead, 0) + " d</td>" +
                     '<td class="num">' + fmt(h.n, 0) + "</td>" +
                     '<td class="num">' + fmt(h.mae, 1) + " mmHg</td>" +
                     '<td class="num">' + fmt(h.within_10 * 100, 0) + "%</td>";
      tb.appendChild(tr);
    });
    $("accuracy-note").textContent = bt.caption || "";

    var svg = $("accuracy-chart");
    clear(svg);
    Array.prototype.slice.call(svg.parentNode.querySelectorAll(".tip,[role=status]"))
      .forEach(function (n) { n.remove(); });
    if (series.length < 2) { $("accuracy-desc").textContent = ""; return; }

    var W = 900, H = 220, m = { l: 46, r: 16, t: 12, b: 26 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var dom = C.domain(series.map(function (p) { return p.actual; })
                       .concat(series.map(function (p) { return p.predicted; })), 0.12);
    function X(i) { return m.l + (i / (series.length - 1)) * iw; }
    var Y = C.axes(svg, m, iw, ih, dom[0], dom[1], function (v) { return fmt(v, 0); }, 3);
    var s1 = C.cssVar("--series-1"), s2 = C.cssVar("--series-2");
    C.line(svg, series.map(function (p, i) { return [X(i), Y(p.actual)]; }), s1, { width: 2 });
    C.line(svg, series.map(function (p, i) { return [X(i), Y(p.predicted)]; }), s2,
           { width: 2, dash: "5 4" });
    C.xTicks(svg, m, ih, [0, Math.floor(series.length / 2), series.length - 1],
             X, function (i) { return shortDate(series[i].ts); });
    C.hoverLayer(svg, svg.parentNode, m, iw, ih, series.length, X, Y,
      function (i) { return { y: series[i].actual }; },
      function (i) {
        var p = series[i];
        return shortDate(p.ts) + " — actual " + fmt(p.actual, 0) +
               ", predicted " + fmt(p.predicted, 0) + " (" +
               (p.error >= 0 ? "+" : "") + fmt(p.error, 1) + ")";
      });
    C.legendInto($("accuracy-legend"), [
      { label: "What happened", colour: s1 },
      { label: "What the model said", colour: s2, dash: true }
    ]);
    $("accuracy-desc").textContent =
      "Next-session forecast replayed over " + series.length +
      " past sessions against the measured value.";
  }

  /* The chained block returns every head at every session -- 60+ rows. Rendering that flat is
   * a data dump, not a read. Grouped by session, ranked, and split: the `mech_*`/`any`/
   * `red_flag` aggregates are a different kind of thing from a named symptom and belong in
   * their own row of summary chips rather than interleaved by probability.
   *
   * No row is flagged. `cut_applies` is false throughout -- the operating cut was chosen on
   * observed rows and does not carry its alert-budget meaning here -- so this section ranks
   * and describes, and never raises an alert. */
  var AGG = { any: "Any symptom", red_flag: "Priority symptom", mech_hypertensive: "Pressure-driven",
              mech_hypotensive: "Drop-driven", mech_volume: "Volume-driven", mech_drug: "Medication-driven" };
  var TOP_N = 6;

  function renderChain(d) {
    var card = $("chain-card"), ch = d.symptom_chained;
    card.hidden = !ch;
    if (!ch) return;
    var host = $("chain-body");
    clear(host);
    if (!ch.available) {
      $("chain-hint").textContent = "";
      $("chain-note").textContent = ch.reason || ch.session_1_note || "";
      return;
    }

    var items = ch.items || [];
    var sessions = (ch.sessions_ahead || []).slice().sort(function (a, b) { return a - b; });
    var basis = ch.uncertainty_basis || {};

    sessions.forEach(function (s) {
      var mine = items.filter(function (it) { return it.sessions_ahead === s; });
      if (!mine.length) return;
      var named = mine.filter(function (it) { return !AGG[it.key]; })
                      .sort(function (a, b) { return b.prob - a.prob; });
      var aggs = mine.filter(function (it) { return AGG[it.key]; })
                     .sort(function (a, b) { return b.prob - a.prob; });
      var ref = mine[0];

      var block = document.createElement("div");
      block.className = "chain-block";
      var head = "Session " + s + " ahead" +
        (ref.days_ahead != null ? " · about " + fmt(ref.days_ahead, 0) + " days" : "");
      // The marginalised probability and the point estimate differ only where a band exists;
      // where they do, the gap IS the Jensen correction and is worth showing.
      var gaps = mine.map(function (it) { return Math.abs(it.jensen_gap || 0); });
      var maxGap = gaps.length ? Math.max.apply(null, gaps) : 0;

      block.innerHTML =
        '<div class="chain-head"><h4>' + esc(head) + "</h4>" +
        '<span class="spacer"></span><span class="caption">conditioned on the forecast ' +
        "through session " + esc(String(ref.conditioned_through_session)) + " (" +
        fmt(ref.conditioned_through_sbp, 0) + " mmHg)</span></div>" +
        '<div class="chips">' + aggs.map(function (it) {
          return '<span class="chip-stat"><span class="cs-k">' + esc(AGG[it.key]) +
                 '</span><span class="cs-v">' + fmt(it.prob * 100, 0) + "%</span></span>";
        }).join("") + "</div>";

      var wrap = document.createElement("div");
      wrap.className = "table-wrap";
      var t = document.createElement("table");
      t.innerHTML = "<thead><tr><th>Most likely symptom</th><th>Driver</th>" +
                    '<th class="num">Projected</th><th>Relative</th></tr></thead>';
      var body = document.createElement("tbody");
      var top = named.slice(0, TOP_N);
      var scale = top.length ? top[0].prob : 1;
      top.forEach(function (it) {
        var tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + esc(pretty(it.key)) +
          (it.red_flag ? " " + pill("critical", "priority") : "") + "</td>" +
          '<td class="caption">' + esc(pretty(it.mechanism)) + "</td>" +
          '<td class="num">' + fmt(it.prob * 100, 1) + "%" +
          (Math.abs(it.jensen_gap || 0) >= 0.005
            ? ' <span class="dim">(' + (it.jensen_gap > 0 ? "+" : "") +
              fmt(it.jensen_gap * 100, 1) + ")</span>"
            : "") + "</td>" +
          '<td><span class="bar"><i style="width:' +
          Math.max(2, (it.prob / (scale || 1)) * 100).toFixed(1) + '%"></i></span></td>';
        body.appendChild(tr);
      });
      t.appendChild(body);
      wrap.appendChild(t);
      block.appendChild(wrap);

      var foot = document.createElement("p");
      foot.className = "caption";
      foot.textContent =
        (named.length > TOP_N ? "Showing the " + TOP_N + " highest of " + named.length +
                                " symptoms. " : "") +
        (basis[String(s)] || "") +
        (maxGap >= 0.005 ? "" : "");
      block.appendChild(foot);
      host.appendChild(block);
    });

    $("chain-hint").textContent = ch.n_heads + " heads · sessions " + sessions.join(", ");
    $("chain-note").textContent = [ch.reach_note, ch.conditioning_note, ch.cut_note,
                                   ch.session_1_note].filter(Boolean).join(" ");
  }

  /* How much of the fitted model this request actually fed. The point is not the percentage
   * -- it is the gap list, which turns "your forecast used 68% of the model" into a specific
   * thing the user can go and supply. */
  function renderCoverage(d) {
    var card = $("coverage-card"), fc = d.feature_coverage;
    card.hidden = !fc || fc.error || !fc.fitted;
    if (card.hidden) return;

    $("coverage-fill").style.width = Math.max(1, fc.pct).toFixed(1) + "%";
    $("coverage-hint").textContent = fc.pct.toFixed(0) + "% of the fitted model";
    $("coverage-headline").innerHTML =
      "<strong>" + fc.resolved + " of " + fc.fitted + "</strong> inputs the model was " +
      "trained on carried a value for this request" +
      (fc.missing ? ", " + fc.missing + " did not." : ".");

    var gaps = fc.gaps || [];
    $("coverage-gaps-wrap").hidden = !gaps.length;
    var tb = $("coverage-table").querySelector("tbody");
    clear(tb);
    gaps.forEach(function (g) {
      var tr = document.createElement("tr");
      // `how` carries backtick-quoted tokens from the API; rendered as code, escaped first.
      tr.innerHTML = "<td>" + esc(g.supply) + "</td>" +
                     '<td class="num">' + fmt(g.features, 0) + "</td>" +
                     '<td class="caption">' +
                     esc(g.how || "").replace(/`([^`]+)`/g, function (_, c) {
                       return "<code>" + c + "</code>";
                     }) + "</td>";
      tb.appendChild(tr);
    });

    $("coverage-note").textContent =
      (fc.needs_more_sessions
        ? fc.needs_more_sessions + " more need a longer history rather than a new field — " +
          "the 30-session windows fill in as sessions accumulate. "
        : "") + (fc.note || "");
  }

  function renderGovernance(d) {
    var card = $("governance-card"), g = d.governance || {};
    var pers = d.personalisation || {};
    card.hidden = !Object.keys(g).length;
    if (card.hidden) return;
    var rows = [
      ["Emergency floor", fmt(g.emergency_floor_mmHg, 0) + " mmHg",
       "never personalised, for any patient"],
      ["Population threshold", fmt(g.population_threshold_mmHg, 0) + " mmHg",
       "the starting point before personalisation"]
    ];
    if (pers.threshold != null) {
      rows.push(["This patient's threshold", fmt(pers.threshold, 0) + " mmHg",
                 "cohort " + (pers.cohort_key || "") +
                 (pers.capped ? ", capped at the governance limit" : "")]);
    }
    if (d.model_version) rows.push(["Model", d.model_version, "serving this advisory"]);
    var tb = $("governance-table").querySelector("tbody");
    clear(tb);
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      tr.innerHTML = "<th scope=\"row\" style=\"text-transform:none;font-size:13.5px;color:var(--ink)\">" +
                     esc(r[0]) + "</th>" +
                     '<td class="num">' + esc(r[1]) + "</td>" +
                     '<td class="caption">' + esc(r[2]) + "</td>";
      tb.appendChild(tr);
    });
  }

  /* Each section in its own try: one renderer meeting an unexpected shape must not blank the
   * page, and the console keeps the cause. */
  function paint(d) {
    [["banner", renderBanner], ["tiles", renderTiles], ["trend", renderTrend],
     ["outlook", renderOutlook], ["risk", renderRisk], ["engine", renderEngine],
     ["accuracy", renderAccuracy], ["chain", renderChain],
     ["coverage", renderCoverage], ["governance", renderGovernance]].forEach(function (pair) {
      try { pair[1](d); }
      catch (e) { console.error("render " + pair[0] + " failed:", e); }
    });
  }

  try {
    var saved = localStorage.getItem("cp-theme");
    if (saved) applyTheme(saved);
  } catch (e) { /* private mode */ }
  boot();
})();
