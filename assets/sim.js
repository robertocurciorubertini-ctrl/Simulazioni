/* ══════════════════════════════════════════════════════════════════════
   fisica.sim — script comune delle simulazioni
   ----------------------------------------------------------------------
   Va incluso in ogni simulazione DOPO il blocco window.SIM_METADATA:

     <link rel="stylesheet" href="../assets/sim.css">
     <script src="../assets/sim.js" defer></script>

   Cosa fa:
     • inserisce la barra superiore (ritorno al catalogo, categoria, titolo,
       livello, simulazione precedente/successiva della stessa categoria);
     • legge simulations.json per colore della categoria e navigazione;
     • espone window.FS con la tavolozza comune e alcune utilità per i canvas.
   Non richiede alcuna modifica al codice della simulazione.
   ══════════════════════════════════════════════════════════════════════ */
(function () {
  "use strict";

  // Radice del sito, ricavata dall'indirizzo di questo stesso file.
  var self = document.currentScript;
  var ROOT = self && self.src ? self.src.replace(/assets\/sim\.js(\?.*)?$/, "") : "./";

  var meta = window.SIM_METADATA || {};
  var norm = function (v) { return String(v == null ? "" : v).trim().toLowerCase(); };
  var esc = function (v) {
    return String(v == null ? "" : v)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  };

  /* ── Tavolozza e utilità per i canvas ─────────────────────────────── */
  var FS = (window.FS = window.FS || {});
  FS.root = ROOT;
  FS.color = {
    ink: "#1b1d22", ink2: "#474c57", ink3: "#7d828c",
    grid: "rgba(27,29,34,0.07)", axis: "rgba(27,29,34,0.45)",
    paper: "#f4f1ea", surface: "#fffdf9",
    blue: "#1f6fd0", red: "#c2491d", green: "#0e8a5a", violet: "#6d4fc0", amber: "#b7791f"
  };
  FS.font = function (px, style) {
    return (style ? style + " " : "") + px + 'px "DM Mono", ui-monospace, monospace';
  };
  /* Adatta il canvas alla densità di pixel dello schermo.
     Restituisce larghezza e altezza in pixel CSS e lascia il contesto già scalato. */
  FS.fitCanvas = function (canvas, cssHeight) {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth || canvas.offsetWidth;
    var h = cssHeight || canvas.clientHeight;
    canvas.style.height = h + "px";
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  };
  /* Passo "tondo" (1, 2, 5 × 10ⁿ) per le tacche di un asse. */
  FS.niceStep = function (range, targetTicks) {
    var raw = range / Math.max(1, targetTicks || 6);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var n = raw / mag;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
  };
  /* Freccia vettoriale da (x0,y0) a (x1,y1). */
  FS.arrow = function (ctx, x0, y0, x1, y1, color, width) {
    var a = Math.atan2(y1 - y0, x1 - x0);
    var len = Math.hypot(x1 - x0, y1 - y0);
    var head = Math.min(9, Math.max(4, len * 0.35));
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = color;
    ctx.lineWidth = width || 1.8;
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1 - Math.cos(a) * head * 0.6, y1 - Math.sin(a) * head * 0.6); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - head * Math.cos(a - 0.4), y1 - head * Math.sin(a - 0.4));
    ctx.lineTo(x1 - head * Math.cos(a + 0.4), y1 - head * Math.sin(a + 0.4));
    ctx.closePath(); ctx.fill();
    ctx.restore();
  };

  /* ── Barra superiore ──────────────────────────────────────────────── */
  function levelLabel(v) {
    var n = norm(v);
    return n === "base" ? "Base" : n === "intermedio" ? "Intermedio" : n === "avanzato" ? "Avanzato" : "";
  }

  function currentFile() {
    var here;
    try { here = decodeURIComponent(location.pathname); } catch (e) { here = location.pathname; }
    return here.replace(/\\/g, "/");
  }

  function buildBar(info) {
    var old = document.querySelector(".fsbar");
    if (old) old.remove();

    var title = meta.title || document.title || "Simulazione";
    var category = info.category || meta.category || "";
    var level = levelLabel(meta.level);
    // stessa regola usata da index.html per gli identificatori delle sezioni
    var slug = norm(info.isWip ? "wip" : category).normalize("NFD").replace(/[̀-ͯ]/g, "")
      .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
    var catHref = ROOT + "index.html" + (slug ? "#" + slug : "");

    var bar = document.createElement("nav");
    bar.className = "fsbar";
    bar.setAttribute("aria-label", "Navigazione del sito");
    bar.innerHTML =
      '<a class="fsbar-home" href="' + ROOT + 'index.html" title="Torna al catalogo">' +
        '<span aria-hidden="true">←</span><span class="fsbar-home-text">fisica.<b>sim</b></span></a>' +
      '<div class="fsbar-crumb">' +
        (category ? '<a class="fsbar-cat" href="' + esc(catHref) + '">' + esc(category) + "</a>" : "") +
        '<span class="fsbar-title">' + esc(title) + "</span>" +
      "</div>" +
      (info.isWip ? '<span class="fsbar-chip wip" title="Simulazione in costruzione">In costruzione</span>' : "") +
      (level ? '<span class="fsbar-chip">' + esc(level) + "</span>" : "") +
      '<div class="fsbar-nav">' +
        (info.prev
          ? '<a href="' + esc(ROOT + encodeURI(info.prev.file)) + '" title="Precedente: ' + esc(info.prev.title) + '" aria-label="Simulazione precedente: ' + esc(info.prev.title) + '">‹</a>'
          : '<span aria-hidden="true">‹</span>') +
        (info.next
          ? '<a href="' + esc(ROOT + encodeURI(info.next.file)) + '" title="Successiva: ' + esc(info.next.title) + '" aria-label="Simulazione successiva: ' + esc(info.next.title) + '">›</a>'
          : '<span aria-hidden="true">›</span>') +
      "</div>";

    if (info.color) document.documentElement.style.setProperty("--fs-cat", info.color);
    document.documentElement.classList.add("fs-has-bar");
    document.body.insertBefore(bar, document.body.firstChild);
  }

  function start() {
    var here = currentFile();
    var wipGuess = /\/WIP\//i.test(here);
    buildBar({ isWip: wipGuess }); // subito, senza attendere il catalogo

    if (meta.title && !/fisica\.sim/.test(document.title)) {
      document.title = meta.title + " · Simulazioni di Fisica";
    }

    fetch(ROOT + "simulations.json", { cache: "no-cache" })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (data) {
        var sims = (data.simulations || []).filter(function (s) { return s && s.file; });
        var me = sims.find(function (s) { return here.endsWith("/" + s.file) || here.endsWith(s.file); });
        if (!me) return;
        var cat = (data.categories || []).find(function (c) { return norm(c.name) === norm(me.category); }) || {};
        var siblings = sims
          .filter(function (s) { return norm(s.category) === norm(me.category); })
          .sort(function (a, b) {
            var d = (Number(a.order) || 0) - (Number(b.order) || 0);
            return d !== 0 ? d : String(a.title).localeCompare(String(b.title), "it");
          });
        var i = siblings.indexOf(me);
        buildBar({
          category: me.category,
          color: cat.color,
          isWip: Boolean(me.isWip) || wipGuess,
          prev: siblings[i - 1],
          next: siblings[i + 1]
        });
      })
      .catch(function () { /* senza catalogo (es. file aperto in locale) resta la barra essenziale */ });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
