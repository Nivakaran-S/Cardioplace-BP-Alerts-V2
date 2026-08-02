/* Cardioplace BP Alerts — dashboard controller.
 *
 * One IIFE, no build step, no dependencies. Every renderer is pure and null-defensive: the
 * API returns a deliberately open payload where blocks are absent rather than empty (no
 * forecast on a cold start, no interval at most horizons, no detector before warm-up), so a
 * renderer that assumes its block exists takes the whole page down with it. `paint` runs each
 * section in its own try, for the same reason.
 *
 * One card per question, because they are answered by different models carrying different
 * evidence: systolic and diastolic are separate forecasters shipping separate families,
 * symptoms are calibrated classifier heads, the threshold is a shrinkage estimator under a
 * governance cap, and the anomaly score is a detector. Stacking them on one chart implied a
 * single model with a single confidence, which was never true.
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
  var SYM_TAB = 0;               // which symptom horizon the segmented control shows

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

  /* Tier codes are engine identifiers, not prose. Sentence-casing `BP_LEVEL_1_HIGH` gives
   * "Bp level 1 high", which reads like a bug; these are the clinician-facing names. */
  var TIER_LABEL = {
    BP_LEVEL_1_HIGH: "Stage 1 high", BP_LEVEL_1_LOW: "Stage 1 low",
    BP_LEVEL_2: "Stage 2 high", BP_LEVEL_2_SYMPTOM_OVERRIDE: "Stage 2, symptom override",
    EMERGENCY: "Emergency", HYPOTENSIVE: "Hypotensive", ELDERLY_LOW: "Low, elderly",
    NARROW_PP: "Narrow pulse pressure", WIDE_PP: "Wide pulse pressure",
    CONTRAINDICATION: "Contraindication", AFIB_RVR: "AF with rapid rate"
  };
  function tierName(t) { return t ? (TIER_LABEL[t] || pretty(t)) : "Clear"; }
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
  function wipeTips(svg) {
    Array.prototype.slice.call(svg.parentNode.querySelectorAll(".tip,[role=status]"))
      .forEach(function (n) { n.remove(); });
  }

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
  function chip(k, v) {
    return '<span class="stat"><span class="sk">' + esc(k) + '</span><span class="sv">' +
           v + "</span></span>";
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
               ", meds=" + (i % 5 === 0 ? "n" : "y"));
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
      // Symptoms have no checkbox group: they are per-READING, and the only row a checkbox
      // could honestly describe is the last one. `sym=` on the line says the same thing
      // without implying the others were symptom-free. The vocabulary still has to be
      // discoverable, though -- the schema rejects an unknown key with "unknown symptom
      // keys: [...]" and does not list the valid ones -- so it is rendered into the format
      // help from /api/schema, which means it cannot drift from what the API accepts.
      var st = $("sym-tokens");
      if (st && (VOCAB.symptoms || []).length) {
        st.textContent = VOCAB.symptoms.map(function (s) { return s.key; }).join(", ");
      }
    } catch (e) { /* the form still works with the free-text fields alone */ }
    refreshHealth();
  }

  async function refreshHealth() {
    var chipEl = $("model-chip"), text = $("model-chip-text");
    try {
      var h = await (await fetch("/api/health")).json();
      chipEl.className = "chip " + (h.model_loaded ? "is-live" : "is-degraded");
      text.textContent = h.model_loaded ? (h.model_version || "model loaded")
                                        : "rule engine only";
      chipEl.title = h.model_loaded
        ? "Serving " + (h.model_version || "") + " on scikit-learn " + (h.sklearn_runtime || "")
        : (h.detail || "no model on disk; the rule engine is unaffected");
    } catch (e) {
      chipEl.className = "chip"; text.textContent = "offline";
    }
  }

  // ----------------------------------------------------------------- request

  /* `date, sbp, dbp[, pulse]` positionally, then any number of `key=value` tokens.
   *
   * The keyed fields are per-SESSION, not per-patient, and that is the point: weight and
   * same-day adherence change between sessions, and the model was fitted on their lagged and
   * rolling forms. Supplying them as a profile constant would be a different -- and wrong --
   * statement about the patient.
   *
   * An unknown key is an ERROR, not an ignored token. The schema forbids unknown fields for
   * the same reason: a mistyped `weight=` that silently did nothing would produce a confident
   * forecast built on a feature the user believes they supplied. */
  var TOKENS = { w: ["weight", "kg"], weight: ["weight", "kg"] };

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
        if (!isFinite(Number(v))) throw new Error(where + k + "= must be a number in " + spec[1]);
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
    // Symptoms are per-READING in the schema and are entered per reading, via `sym=`. The
    // "Symptoms at the latest reading" checkbox group used to sit alongside this and attach
    // to the last row only; it was removed because it could describe exactly one reading
    // while the token describes any of them, and two inputs for one field meant a precedence
    // rule (`sym=` won) that nothing in the UI explained.
    var rows = parseReadings($("readings").value);
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
    var timer = setTimeout(function () { ctl.abort(); }, 180000);
    try {
      var res = await fetch("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body), signal: ctl.signal
      });
      var out = await res.json();
      if (!res.ok) { showError(out.detail || ("Request failed (" + res.status + ")")); return; }
      LAST = out;
      SYM_TAB = 0;
      paint(out);
    } catch (err) {
      showError(err.name === "AbortError"
        ? "The request timed out after 180 seconds."
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
      kind = "critical"; title = "Emergency on the latest reading";
      detail = pretty(eng.rule_id) + ". The emergency floor is never personalised, and the " +
               "rule engine is authoritative here.";
    } else if (ew.flagged) {
      kind = "critical"; title = "Anomaly detected";
      detail = "Detector score " + fmt(ew.score, 3) + " at or above the " + fmt(ew.cut, 3) +
               " cut, roughly " + fmt(ew.est_lead_days, 0) + " days of lead time.";
    } else if (firstFire) {
      kind = "watch";
      title = tierName(firstFire.tier) + " forecast in about " +
              fmt(firstFire.days_ahead, 0) + " days";
      detail = "Predicted systolic " + fmt(firstFire.sbp, 0) + " mmHg would fire " +
               pretty(firstFire.rule_id) + " if the trend holds.";
    } else if (eng.fired) {
      kind = "watch"; title = tierName(eng.tier) + " on the latest reading";
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
    var dbp = ((d.forecast || {}).dbp || {}).h0 || {};
    var out = [];

    if (sbp.point !== undefined) {
      out.push(tile("Next systolic", fmt(sbp.point, 0), "mmHg",
                    "in about " + fmt(sbp.days_ahead_est, 0) + " days"));
    }
    if (dbp.point !== undefined) {
      out.push(tile("Next diastolic", fmt(dbp.point, 0), "mmHg",
                    "in about " + fmt(dbp.days_ahead_est, 0) + " days"));
    }
    if (pers.threshold != null) {
      out.push(tile("Alert threshold", fmt(pers.threshold, 0), "mmHg",
                    (pers.offset >= 0 ? "+" : "") + fmt(pers.offset, 1) +
                    " mmHg vs the population 140"));
    }
    if (ew.score != null) {
      out.push(tile("Anomaly score", fmt(ew.score, 2), "",
                    ew.flagged ? "above the " + fmt(ew.cut, 2) + " cut"
                               : "below the " + fmt(ew.cut, 2) + " cut",
                    ew.flagged ? "critical" : "good"));
    }
    out.push(tile("Readings", fmt(d.n_observations, 0), "",
                  (d.confidence_tier || "").replace("_", " ")));
    host.innerHTML = out.join("");
    host.hidden = !out.length;
  }

  /* One card per signal, because systolic and diastolic are separate forecasters shipping
   * separate families and carrying different evidence: systolic has a fitted conformal band
   * at one horizon, diastolic ships the EWMA baseline and has none. A shared chart implied
   * one model with one confidence. */
  var SIG = {
    sbp: { name: "Systolic", colour: "--series-1", refs: true },
    dbp: { name: "Diastolic", colour: "--series-3", refs: false }
  };

  function renderForecast(d, sig) {
    var spec = SIG[sig];
    var card = $(sig + "-card"), svg = $(sig + "-chart");
    var hist = (d.history || []).filter(function (r) { return r[sig] != null; });
    var per = (d.forecast || {})[sig] || {};
    var fkeys = Object.keys(per).sort().filter(function (k) {
      return per[k] && isFinite(per[k].point);
    });
    card.hidden = hist.length < 2;
    if (card.hidden) return;

    var bt = ((d.backtest || {}).signals || {})[sig] || {};
    var bth = bt.horizons || [];
    var obs = hist.map(function (r) { return r[sig]; });
    var next = fkeys.length ? per[fkeys[0]] : null;
    var lastObs = obs[obs.length - 1];

    // The headline is the number the card exists to deliver, plus its move from the last
    // measured reading. A forecast without that comparison is a number with no direction.
    var delta = next ? next.point - lastObs : null;
    $(sig + "-headline").innerHTML = next
      ? '<div class="hero"><span class="hv">' + fmt(next.point, 0) +
        '</span><span class="hu">mmHg</span></div>' +
        '<div class="hsub">next session, in about ' + fmt(next.days_ahead_est, 0) +
        " days · " + (delta >= 0 ? "+" : "") + fmt(delta, 1) +
        " mmHg from the last reading of " + fmt(lastObs, 0) + "</div>"
      : '<div class="hsub">No forecast was issued for this signal.</div>';

    // Confidence stated from what exists rather than asserted. The interval is fitted at one
    // horizon only, so where it is absent the card says so instead of implying a band.
    var conf = [];
    if (next && next.lo80 != null) {
      conf.push(chip("80% interval", "±" + fmt((next.hi80 - next.lo80) / 2, 0) + " mmHg"));
    } else {
      var banded = fkeys.filter(function (k) { return per[k].lo80 != null; });
      conf.push(chip("80% interval", banded.length
        ? "fitted only at " + banded.length + " of " + fkeys.length + " horizons"
        : '<span class="dim">not fitted for this signal</span>'));
    }
    if (bth.length) {
      conf.push(chip("Typical error", fmt(bth[0].mae, 1) + " mmHg"));
      conf.push(chip("Within 10 mmHg", fmt(bth[0].within_10 * 100, 0) + "%"));
      conf.push(chip("Scored on", bth[0].n + " past sessions"));
    }
    if (bt.family) {
      conf.push(chip("Model", bt.family === "baseline"
        ? "baseline (" + esc(bt.architecture) + ")" : esc(bt.architecture)));
    }
    $(sig + "-conf").innerHTML = conf.join("");

    // ---- chart -----------------------------------------------------------------
    clear(svg); wipeTips(svg);
    var GUT = 62;                       // label gutter, so direct labels never hit the marks
    var W = 940, H = 300, m = { l: 46, r: 16 + GUT, t: 14, b: 30 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);

    var persT = (d.personalisation || {}).threshold;
    var floor = (d.governance || {}).emergency_floor_mmHg;
    var all = obs.concat(fkeys.map(function (k) { return per[k].point; }));
    fkeys.forEach(function (k) {
      if (per[k].lo80 != null) all.push(per[k].lo80, per[k].hi80);
    });
    if (spec.refs) {
      [persT, floor].forEach(function (v) { if (v != null) all.push(v); });
    }
    var dom = C.domain(all, 0.10);
    var yMin = dom[0], yMax = dom[1];

    /* The forecast gets a fixed share of the width. On a plain index scale three predicted
     * points among forty observed ones get 3/43 of the chart -- about sixty pixels for the
     * part the reader came for. The scale break is not hidden: the shaded band marks it. */
    var fShare = fkeys.length ? 0.24 : 0;
    var iwH = iw * (1 - fShare);
    var n = hist.length + fkeys.length;
    function X(i) {
      if (i < hist.length) {
        return m.l + (hist.length <= 1 ? iwH / 2 : (i / (hist.length - 1)) * iwH);
      }
      return m.l + iwH + ((i - hist.length + 1) / fkeys.length) * (iw - iwH);
    }
    var Y = C.axes(svg, m, iw, ih, yMin, yMax, function (v) { return fmt(v, 0); }, 4);

    if (fkeys.length) {
      // Neutral: the band is a region of the x-axis, not a series with an identity.
      C.shadeBand(svg, X(hist.length - 1), X(n - 1), m, ih, C.cssVar("--axis"), "forecast");
    }
    if (spec.refs) {
      C.refLine(svg, Y, floor, m, iw, C.cssVar("--critical"),
                "emergency floor " + fmt(floor, 0), yMin, yMax, "left");
      C.refLine(svg, Y, persT, m, iw, C.cssVar("--ink-muted"),
                "alert threshold " + fmt(persT, 0), yMin, yMax, "left");
    }

    var col = C.cssVar(spec.colour);
    C.line(svg, obs.map(function (v, i) { return [X(i), Y(v)]; }), col, { width: 2 });

    if (fkeys.length) {
      var pts = [[X(hist.length - 1), Y(lastObs)]].concat(
        fkeys.map(function (k, i) { return [X(hist.length + i), Y(per[k].point)]; }));
      // Colour follows the ENTITY: the forecast keeps the signal's hue and is marked
      // predicted by the dash. A separate hue would say these are two different things.
      C.line(svg, pts, col, { width: 2, dash: "6 4" });
      C.dots(svg, pts.slice(1), col, 3.5);
      fkeys.forEach(function (k, i) {
        var nd = per[k];
        if (nd.lo80 == null) return;
        svg.appendChild(C.svgEl("line", {
          x1: X(hist.length + i), x2: X(hist.length + i),
          y1: Y(nd.hi80), y2: Y(nd.lo80), stroke: col, "stroke-width": 6,
          "stroke-linecap": "round", opacity: .22
        }));
      });
      svg.appendChild(C.svgEl("text", {
        x: m.l + iw + 8, y: Y(per[fkeys[fkeys.length - 1]].point) + 4,
        fill: col, "font-size": 11.5, "font-weight": 600
      }, spec.name.toLowerCase()));
    }

    C.xTicks(svg, m, ih, [0, Math.floor(hist.length / 2), hist.length - 1].filter(
      function (v, i, a) { return a.indexOf(v) === i && v >= 0; }),
      X, function (i) { return shortDate(hist[i].ts); });

    C.hoverLayer(svg, svg.parentNode, m, iw, ih, n, X, Y,
      function (i) {
        return i < hist.length ? { y: obs[i] } : { y: per[fkeys[i - hist.length]].point };
      },
      function (i) {
        if (i < hist.length) return shortDate(hist[i].ts) + " — " + fmt(obs[i], 0) + " mmHg";
        var nd = per[fkeys[i - hist.length]];
        return "forecast " + fmt(nd.point, 0) + " mmHg in about " +
               fmt(nd.days_ahead_est, 0) + " days" +
               (nd.lo80 != null ? " (80% " + fmt(nd.lo80, 0) + "–" + fmt(nd.hi80, 0) + ")" : "");
      });

    C.legendInto($(sig + "-legend"), [
      { label: spec.name + ", measured", colour: col },
      { label: spec.name + ", forecast", colour: col, dash: true }
    ]);
    $(sig + "-hint").textContent = hist.length + " sessions";
    $(sig + "-desc").textContent =
      spec.name + " over " + hist.length + " sessions, continuing to the right of the last " +
      "measurement as a dashed forecast" +
      (fkeys.some(function (k) { return per[k].lo80 != null; })
        ? "; the vertical band is the 80% interval, drawn only where one is fitted." : ".");

    // ---- table (the relief rule, and where the interval and checks live) --------
    var tb = $(sig + "-table").querySelector("tbody");
    clear(tb);
    fkeys.forEach(function (k, i) {
      var nd = per[k], co = nd.coherence || {};
      var h = bth[i] || {};
      var tr = document.createElement("tr");
      tr.innerHTML =
        '<td class="num">' + fmt(nd.days_ahead_est, 0) + " d</td>" +
        '<td class="num">' + fmt(nd.point, 1) + " mmHg</td>" +
        '<td class="num">' + (nd.lo80 != null
          ? fmt(nd.lo80, 0) + " – " + fmt(nd.hi80, 0)
          : '<span class="dim">not fitted here</span>') + "</td>" +
        '<td class="num">' + (h.mae != null
          ? fmt(h.mae, 1) + " mmHg" : '<span class="dim">–</span>') + "</td>" +
        "<td>" + (co.ok === undefined ? "" : co.ok
          ? pill("good", "coherent")
          : pill("critical", (co.violations || [])[0] || "check failed")) + "</td>";
      tb.appendChild(tr);
    });
    $(sig + "-note").textContent =
      (bt.family === "baseline"
        ? "This signal ships the " + bt.architecture + " baseline: no learned candidate beat " +
          "it by enough to justify shipping one, and a baseline has no fitted interval. "
        : "") + ((d.backtest || {}).caption || "");
  }

  /* Symptoms, with the Venn-Abers pair as the confidence statement. At a 1% base rate a bare
   * probability of 0.004 looks equally authoritative whether four hundred calibration
   * examples support it or two; the width of the pair is the difference. */
  var AGG = { any: "Any symptom", red_flag: "Priority symptom",
              mech_hypertensive: "Pressure-driven", mech_hypotensive: "Drop-driven",
              mech_volume: "Volume-driven", mech_drug: "Medication-driven" };
  var TOP_N = 8;

  function symptomTabs(d) {
    var tabs = [];
    var now = d.symptom_risk || {};
    if (now.available && (now.items || []).length) {
      tabs.push({ label: "Latest reading", items: now.items, note: now.note || "", basis: null });
    }
    var ch = d.symptom_chained;
    if (ch && ch.available) {
      (ch.sessions_ahead || []).slice().sort(function (a, b) { return a - b; })
        .forEach(function (s) {
          var mine = (ch.items || []).filter(function (it) { return it.sessions_ahead === s; });
          if (!mine.length) return;
          tabs.push({
            label: "Session " + s + " ahead", items: mine,
            note: [ch.cut_note, ch.conditioning_note].filter(Boolean).join(" "),
            basis: (ch.uncertainty_basis || {})[String(s)] || null
          });
        });
    }
    return tabs;
  }

  function renderSymptoms(d) {
    var card = $("symptom-card");
    var tabs = symptomTabs(d);
    card.hidden = !tabs.length;
    if (card.hidden) return;
    if (SYM_TAB >= tabs.length) SYM_TAB = 0;

    var seg = $("symptom-seg");
    clear(seg);
    tabs.forEach(function (t, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "seg-btn" + (i === SYM_TAB ? " on" : "");
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", i === SYM_TAB ? "true" : "false");
      b.textContent = t.label;
      b.addEventListener("click", function () { SYM_TAB = i; renderSymptoms(LAST); });
      seg.appendChild(b);
    });

    var tab = tabs[SYM_TAB];
    /* The two blocks key their items differently: the observed block appends the horizon
     * (`any_h0`, `mech_volume_h0`) while the chained one does not (`any`). Looking up the
     * raw key therefore missed every aggregate on the "Latest reading" tab, and "Any
     * symptom", "Priority symptom" and the four mechanism scores were ranked in the table as
     * though they were symptoms a patient could have. */
    function aggOf(it) { return AGG[String(it.key).replace(/_h\d+$/, "")]; }
    var named = tab.items.filter(function (it) { return !aggOf(it); })
                         .sort(function (a, b) { return b.prob - a.prob; });
    var aggs = tab.items.filter(aggOf)
                        .sort(function (a, b) { return b.prob - a.prob; });

    var host = $("symptom-body");
    clear(host);
    if (aggs.length) {
      var chips = document.createElement("div");
      chips.className = "chips";
      chips.innerHTML = aggs.map(function (it) {
        return '<span class="chip-stat"><span class="cs-k">' +
               esc(aggOf(it)) + '</span><span class="cs-v">' +
               fmt(it.prob * 100, 0) + "%</span></span>";
      }).join("");
      host.appendChild(chips);
    }

    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    var t = document.createElement("table");
    t.innerHTML = "<thead><tr><th>Symptom</th><th>Driver</th>" +
                  '<th class="num">Probability</th><th>Confidence range</th></tr></thead>';
    var body = document.createElement("tbody");
    var top = named.slice(0, TOP_N);
    var scale = top.length ? Math.max(top[0].prob, 0.01) * 1.12 : 1;
    top.forEach(function (it) {
      var lo = it.prob_lo, hi = it.prob_hi, band;
      if (lo != null && hi != null) {
        // The dot is the point estimate; the lighter span behind it is the calibrated
        // interval. Both share one axis, so a wide interval looks wide.
        band = '<span class="pbar">' +
               '<i class="band" style="left:' +
               Math.min(100, 100 * lo / scale).toFixed(1) + "%;width:" +
               Math.max(0.8, Math.min(100, 100 * (hi - lo) / scale)).toFixed(1) + '%"></i>' +
               '<i class="pt" style="left:' +
               Math.min(98.5, 100 * it.prob / scale).toFixed(1) + '%"></i></span>' +
               '<span class="caption">' + fmt(lo * 100, 2) + "–" + fmt(hi * 100, 2) + "%</span>";
      } else {
        band = '<span class="dim">' + esc(it.confidence_basis || "no interval") + "</span>";
      }
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + esc(pretty(it.label || it.key)) +
        (it.red_flag ? " " + pill("critical", "priority") : "") + "</td>" +
        '<td class="caption">' + esc(pretty(it.mechanism)) + "</td>" +
        '<td class="num">' + fmt(it.prob * 100, 1) + "%</td>" +
        "<td>" + band + "</td>";
      body.appendChild(tr);
    });
    t.appendChild(body);
    wrap.appendChild(t);
    host.appendChild(wrap);

    $("symptom-hint").textContent = named.length + " symptoms";
    $("symptom-note").textContent =
      (named.length > TOP_N ? "Showing the " + TOP_N + " highest of " + named.length + ". " : "") +
      "The confidence range is a Venn-Abers pair: a calibrated interval on the probability " +
      "itself, so a number backed by little calibration data reads as uncertain rather than " +
      "merely small. " + (tab.basis || "") + " " + (tab.note || "");
  }

  /* The threshold is a shrinkage estimator under a governance cap, and on most patients the
   * cap is what decides the answer. A single number hides that; this puts every input on one
   * mmHg axis with the permitted band drawn, so a capped result is visibly capped. */
  function renderOffset(d) {
    var card = $("offset-card"), p = d.personalisation || {};
    var g = d.governance || {};
    card.hidden = p.threshold == null;
    if (card.hidden) return;

    var pop = g.population_threshold_mmHg;
    $("offset-headline").innerHTML =
      '<div class="hero"><span class="hv">' + fmt(p.threshold, 0) +
      '</span><span class="hu">mmHg</span></div>' +
      '<div class="hsub">' + (p.offset >= 0 ? "+" : "") + fmt(p.offset, 1) +
      " mmHg against the population " + fmt(pop, 0) +
      (p.capped ? " · held at the governance cap" : "") +
      (p.cohort_key ? " · cohort " + esc(p.cohort_key) : "") + "</div>";

    var svg = $("offset-chart");
    clear(svg); wipeTips(svg);
    var W = 900, H = 176, m = { l: 132, r: 24, t: 12, b: 30 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);

    var capHi = pop + 15, capLo = pop - 25;   // OFFSET_CAP_LOOSEN / TIGHTEN, governance block
    var rows = [
      { k: "Population floor", v: pop, kind: "ref" },
      { k: "Cohort " + (p.cohort_key || ""), v: p.cohort, kind: "input" },
      { k: "This patient", v: p.personal, kind: "input" },
      { k: "Alert threshold", v: p.threshold, kind: "final" }
    ].filter(function (r) { return r.v != null && isFinite(r.v); });

    var dom = C.domain(rows.map(function (r) { return r.v; }).concat([capLo, capHi, pop]), 0.08);
    function X(v) { return m.l + ((v - dom[0]) / (dom[1] - dom[0])) * iw; }

    // Permitted band first, so the marks sit on top of it.
    svg.appendChild(C.svgEl("rect", {
      x: X(capLo), y: m.t, width: Math.max(0, X(capHi) - X(capLo)), height: ih,
      fill: C.cssVar("--good"), opacity: .09
    }));
    [[capLo, "cap −25"], [capHi, "cap +15"]].forEach(function (c) {
      svg.appendChild(C.svgEl("line", { x1: X(c[0]), x2: X(c[0]), y1: m.t, y2: m.t + ih,
                                        stroke: C.cssVar("--good"), "stroke-width": 1.5,
                                        "stroke-dasharray": "4 3", opacity: .85 }));
      svg.appendChild(C.svgEl("text", { x: X(c[0]), y: m.t + ih + 18, "text-anchor": "middle",
                                        fill: C.cssVar("--ink-muted"), "font-size": 10.5 },
                              c[1]));
    });

    var band = ih / rows.length;
    rows.forEach(function (r, i) {
      var y = m.t + band * i + band / 2;
      var colour = r.kind === "final" ? C.cssVar("--series-1")
                 : r.kind === "ref" ? C.cssVar("--ink-muted") : C.cssVar("--series-2");
      svg.appendChild(C.svgEl("line", { x1: m.l, x2: m.l + iw, y1: y, y2: y,
                                        stroke: C.cssVar("--grid"), "stroke-width": 1 }));
      svg.appendChild(C.svgEl("circle", { cx: X(r.v), cy: y, r: r.kind === "final" ? 7 : 5,
                                          fill: colour }));
      svg.appendChild(C.svgEl("text", { x: m.l - 10, y: y + 4, "text-anchor": "end",
                                        fill: C.cssVar("--ink-2"), "font-size": 11.5 }, r.k));
      svg.appendChild(C.svgEl("text", { x: X(r.v), y: y - 11, "text-anchor": "middle",
                                        fill: colour, "font-size": 11.5,
                                        "font-weight": r.kind === "final" ? 700 : 500 },
                              fmt(r.v, 0)));
    });

    $("offset-hint").textContent = p.capped ? "held at the cap" : "within the cap";
    $("offset-desc").textContent =
      "Every input to the personalised threshold on one mmHg axis. The shaded band is the " +
      "range governance permits around the population floor.";
    $("offset-note").textContent =
      "The cohort estimate and this patient's own history are blended" +
      (p.shrinkage_w != null
        ? " with a weight of " + fmt(p.shrinkage_w, 2) + " on the patient (" +
          fmt(p.n_warm, 0) + " warm readings)" : "") +
      ", then clamped to the governance band. " +
      (p.capped
        ? "It clamped here, so the cap and not this patient's own history set the threshold."
        : "It did not clamp, so this threshold is the blended estimate.") +
      " The " + fmt(g.emergency_floor_mmHg, 0) + " mmHg emergency floor is never personalised.";
  }

  /* The detector as a series rather than a single number: one score per settled session with
   * its operating cut drawn across, so a reader can see whether today is unusual for this
   * patient or merely the latest of many. */
  function renderAnomaly(d) {
    var card = $("anomaly-card"), ew = d.early_warning || {}, an = d.anomaly || {};
    card.hidden = ew.score == null;
    if (card.hidden) return;

    $("anomaly-headline").innerHTML =
      '<div class="hero"><span class="hv">' + fmt(ew.score, 2) + "</span></div>" +
      '<div class="hsub">' + (ew.flagged ? "above" : "below") + " the " + fmt(ew.cut, 2) +
      " operating cut" + (ew.est_lead_days != null
        ? " · about " + fmt(ew.est_lead_days, 0) + " days of lead time" : "") + "</div>";

    /* The score is NOT bounded to [0,1] -- d_forecast_level returns 1.08 on the demo. Pinning
     * the bar at full would hide the thing worth seeing: how far past the cut it sits. */
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
      "anomaly score " + fmt(ew.score, 2) + " against a cut of " + fmt(ew.cut, 2));
    $("risk-scale-max").textContent = fmt(hi, 1);
    $("risk-cut-label").textContent = "cut " + fmt(ew.cut, 2);
    $("anomaly-hint").textContent = ew.detector || "";

    var pts = (an.points || []).filter(function (p) { return !p.warmup && isFinite(p.score); });
    var svg = $("anomaly-chart");
    clear(svg); wipeTips(svg);
    if (pts.length < 2) {
      $("anomaly-desc").textContent = "";
      C.legendInto($("anomaly-legend"), []);
      $("anomaly-note").textContent =
        "Not enough settled sessions to draw a history yet — the detector needs " +
        fmt(an.warmup_readings, 0) + " warm-up readings.";
      return;
    }

    var W = 940, H = 230, m = { l: 46, r: 18, t: 14, b: 28 };
    var iw = W - m.l - m.r, ih = H - m.t - m.b;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var dom = C.domain(pts.map(function (p) { return p.score; }).concat([an.cut || 0]), 0.15);
    function X(i) { return m.l + (i / (pts.length - 1)) * iw; }
    var Y = C.axes(svg, m, iw, ih, dom[0], dom[1], function (v) { return fmt(v, 2); }, 4);
    C.refLine(svg, Y, an.cut, m, iw, C.cssVar("--critical"),
              "cut " + fmt(an.cut, 2), dom[0], dom[1], "left");
    C.line(svg, pts.map(function (p, i) { return [X(i), Y(p.score)]; }),
           C.cssVar("--series-1"), { width: 2 });
    var hot = pts.map(function (p, i) { return p.flagged ? [X(i), Y(p.score)] : null; })
                 .filter(Boolean);
    if (hot.length) C.dots(svg, hot, C.cssVar("--critical"), 4.5);
    C.xTicks(svg, m, ih, [0, Math.floor(pts.length / 2), pts.length - 1], X,
             function (i) { return shortDate(pts[i].ts); });
    C.hoverLayer(svg, svg.parentNode, m, iw, ih, pts.length, X, Y,
      function (i) { return { y: pts[i].score }; },
      function (i, v) {
        return shortDate(pts[i].ts) + " — score " + fmt(v.y, 3) +
               (pts[i].flagged ? ", flagged" : "");
      });

    C.legendInto($("anomaly-legend"), [
      { label: "Detector score", colour: C.cssVar("--series-1") },
      { label: "Flagged session", colour: C.cssVar("--critical") }
    ]);
    $("anomaly-desc").textContent =
      "One score per settled session; the dashed line is the operating cut and filled points " +
      "are the sessions that crossed it.";
    $("anomaly-note").textContent =
      an.n_flagged + " of " + an.n_settled + " settled sessions scored above the cut. " +
      (an.event_definition
        ? an.event_definition.charAt(0).toUpperCase() + an.event_definition.slice(1) + ". " : "") +
      "The cut is set so roughly " + fmt(an.budget_pct, 0) + "% of sessions are flagged.";
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
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>" + esc(shortDate(h.ts)) + "</td>" +
                     '<td class="num">' +
                     (r.sbp != null ? fmt(r.sbp, 0) + "/" + fmt(r.dbp, 0) : "–") + "</td>" +
                     "<td>" + pill(h.fired ? tierKind(h.tier, h.is_emergency) : "muted",
                                   tierName(h.tier)) + "</td>" +
                     '<td class="caption">' + (h.rule_id ? esc(pretty(h.rule_id)) : "") + "</td>";
      tb.appendChild(tr);
    });
    $("engine-hint").textContent = eng.fired_count + " of " + hist.length + " sessions fired";
    $("engine-note").textContent = eng.note || "";
  }

  /* How much of the fitted model this request fed. The point is not the percentage -- it is
   * the split between what the caller could still supply and what this product does not
   * collect at all, which is a standing cost rather than an oversight. */
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

    var gaps = (fc.gaps || []).map(function (g) {
      return { what: g.supply, n: g.features, how: g.how, fixable: true };
    }).concat((fc.not_collected || []).map(function (g) {
      return { what: g.measurement, n: g.features,
               how: "not collected — a dialysis measurement", fixable: false };
    }));
    $("coverage-gaps-wrap").hidden = !gaps.length;
    var tb = $("coverage-table").querySelector("tbody");
    clear(tb);
    gaps.forEach(function (g) {
      var tr = document.createElement("tr");
      tr.innerHTML = "<td>" + esc(g.what) + "</td>" +
                     '<td class="num">' + fmt(g.n, 0) + "</td>" +
                     '<td class="caption">' +
                     (g.fixable ? esc(g.how || "").replace(/`([^`]+)`/g, function (_, c) {
                       return "<code>" + c + "</code>";
                     }) : '<span class="dim">' + esc(g.how) + "</span>") + "</td>";
      tb.appendChild(tr);
    });

    $("coverage-note").textContent =
      (fc.needs_more_sessions
        ? fc.needs_more_sessions + " more need a longer history rather than a new field — " +
          "the 30-session windows fill in as sessions accumulate. " : "") +
      (fc.not_collected_features ? fc.not_collected_note + " " : "") + (fc.note || "");
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
      tr.innerHTML = '<th scope="row" class="rowh">' + esc(r[0]) + "</th>" +
                     '<td class="num">' + esc(r[1]) + "</td>" +
                     '<td class="caption">' + esc(r[2]) + "</td>";
      tb.appendChild(tr);
    });
  }

  /* Each section in its own try: one renderer meeting an unexpected shape must not blank the
   * page, and the console keeps the cause. */
  function paint(d) {
    [["banner", renderBanner], ["tiles", renderTiles],
     ["sbp", function (x) { renderForecast(x, "sbp"); }],
     ["dbp", function (x) { renderForecast(x, "dbp"); }],
     ["symptoms", renderSymptoms], ["offset", renderOffset], ["anomaly", renderAnomaly],
     ["outlook", renderOutlook], ["engine", renderEngine],
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
