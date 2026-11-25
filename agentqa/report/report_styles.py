"""How the report looks, and the little it does.

One file, no network. No CDN, no web font, no framework: the page has to open
from a double-click on a laptop with no internet, and from GitHub Pages, and
look the same both times.

Colours are CSS variables, set once for light mode and again for dark mode
under `prefers-color-scheme`. Headings ask for Space Grotesk and fall back to
the system sans stack, because a downloaded font would be a network call.
"""

from __future__ import annotations

REPORT_CSS = """
:root {
  --page: #faf6f0;
  --card: #fffcf7;
  --ink: #1c1a17;
  --ink-soft: #5c564d;
  --line: #e4dccf;
  --pass: #1f7a4d;
  --pass-soft: #e6f2ea;
  --fail: #c8341a;
  --fail-soft: #fbe8e3;
  --flaky: #b26205;
  --flaky-soft: #fdf0dd;
  --error: #8a6d0b;
  --error-soft: #f9f1d6;
  --accent: #c8341a;
  --heading: "Space Grotesk", "Inter", ui-sans-serif, system-ui, -apple-system,
             "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  --body: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
          "Helvetica Neue", sans-serif;
  --mono: ui-monospace, "Cascadia Code", "Consolas", "SF Mono", Menlo, monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --page: #16150f;
    --card: #1f1d16;
    --ink: #f3ede1;
    --ink-soft: #aaa294;
    --line: #35312a;
    --pass: #6cc08a;
    --pass-soft: #1b2a21;
    --fail: #f08063;
    --fail-soft: #2e1d18;
    --flaky: #e0a550;
    --flaky-soft: #2d2317;
    --error: #d7bd5c;
    --error-soft: #2a2516;
    --accent: #f08063;
  }
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  max-width: 100%;
  overflow-x: hidden;
}

body {
  background: var(--page);
  color: var(--ink);
  font-family: var(--body);
  font-size: 16px;
  line-height: 1.55;
  -webkit-text-size-adjust: 100%;
}

.wrap {
  max-width: 1000px;
  margin: 0 auto;
  padding: 32px 16px 64px;
}

h1, h2, h3 { font-family: var(--heading); font-weight: 700; letter-spacing: -0.01em; }
h1 { font-size: 2rem; margin: 0; }
h2 { font-size: 1.25rem; margin: 40px 0 12px; }
h3 { font-size: 1rem; margin: 0 0 8px; }

a { color: var(--accent); }

.masthead { display: flex; flex-wrap: wrap; align-items: baseline; gap: 12px; }
.masthead .what { color: var(--ink-soft); font-size: 1rem; }

.pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 0; padding: 0; list-style: none; }
.pill {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 0.85rem;
  color: var(--ink-soft);
}

.decision {
  margin-top: 28px;
  background: var(--card);
  border: 1px solid var(--line);
  border-left: 6px solid var(--line);
  border-radius: 12px;
  padding: 20px 24px;
}
.decision.ready { border-left-color: var(--pass); }
.decision.blocked { border-left-color: var(--fail); }
.decision .verdict { font-family: var(--heading); font-size: 2.5rem; line-height: 1.1; }
.decision.ready .verdict { color: var(--pass); }
.decision.blocked .verdict { color: var(--fail); }
.decision .counted { font-size: 1.25rem; font-weight: 600; margin-top: 4px; }
.decision p { margin: 10px 0 0; color: var(--ink-soft); }

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 14px 16px;
}
.card .score { font-family: var(--heading); font-size: 1.6rem; }
.card .note { color: var(--ink-soft); font-size: 0.85rem; margin-top: 4px; }
.card.good .score { color: var(--pass); }
.card.bad .score { color: var(--fail); }

.tag {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-radius: 4px;
  padding: 1px 6px;
  margin-left: 6px;
  background: var(--fail-soft);
  color: var(--fail);
}

.filters { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
.filters button {
  font: inherit;
  font-size: 0.85rem;
  cursor: pointer;
  background: var(--card);
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 4px 14px;
}
.filters button[aria-pressed="true"] { background: var(--ink); color: var(--page); border-color: var(--ink); }

.table-scroll {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--card);
}
table { border-collapse: collapse; width: 100%; min-width: 620px; }
th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--ink-soft); }
tr:last-child td { border-bottom: none; }
td.what { color: var(--ink-soft); font-size: 0.9rem; }

.word { font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.03em; }
.word.stable_pass { color: var(--pass); }
.word.flaky { color: var(--flaky); }
.word.stable_fail { color: var(--fail); }
.word.error { color: var(--error); }

details.test {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  margin-bottom: 10px;
  padding: 0;
}
details.test > summary {
  cursor: pointer;
  padding: 12px 16px;
  font-weight: 600;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: baseline;
}
details.test > summary::marker { color: var(--ink-soft); }
details.test .body { padding: 0 16px 16px; }
.asked { color: var(--ink-soft); font-size: 0.9rem; margin: 0 0 12px; }

.run {
  border-top: 1px solid var(--line);
  padding-top: 12px;
  margin-top: 12px;
}
.run h4 { font-family: var(--heading); font-size: 0.95rem; margin: 0 0 8px; }

.steps { font-family: var(--mono); font-size: 0.8rem; margin: 0; padding: 0; list-style: none; }
.steps li { padding: 3px 0; border-bottom: 1px dotted var(--line); overflow-wrap: anywhere; }
.steps li:last-child { border-bottom: none; }
.steps .kind { display: inline-block; min-width: 92px; color: var(--ink-soft); }
.steps li.blocked .kind, .steps li.tool_error .kind { color: var(--fail); }
.steps li.guard_blocked .kind { color: var(--flaky); }
.guardtag {
  display: inline-block;
  border-radius: 999px;
  padding: 0 8px;
  margin-right: 6px;
  font-family: var(--heading);
  font-size: 0.72rem;
}
.guardtag.stopped { background: var(--pass-soft); color: var(--pass); border: 1px solid var(--pass); }
.guardtag.wrong { background: var(--flaky-soft); color: var(--flaky); border: 1px solid var(--flaky); }

.reply, .claimbox, .judge {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  font-size: 0.9rem;
  overflow-wrap: anywhere;
}
.reply { background: var(--page); border: 1px solid var(--line); white-space: pre-wrap; }
.claimbox { background: var(--fail-soft); border: 1px solid var(--fail); color: var(--ink); }
.judge { background: var(--page); border: 1px dashed var(--line); color: var(--ink-soft); }

.runinfo { width: 100%; min-width: 0; }
.runinfo td:first-child { color: var(--ink-soft); white-space: nowrap; }
.runinfo td:last-child { font-family: var(--mono); font-size: 0.85rem; overflow-wrap: anywhere; }

footer { margin-top: 48px; color: var(--ink-soft); font-size: 0.85rem; text-align: center; }

@media (max-width: 520px) {
  h1 { font-size: 1.5rem; }
  .decision .verdict { font-size: 2rem; }
  .wrap { padding: 20px 16px 48px; }
}
"""

# Without JavaScript every row simply shows. This only hides rows once a
# person presses a filter, so the page is complete before it runs.
REPORT_JS = """
(function () {
  var buttons = document.querySelectorAll('.filters button');
  var rows = document.querySelectorAll('tr[data-stability]');
  function show(which) {
    for (var i = 0; i < rows.length; i++) {
      var stability = rows[i].getAttribute('data-stability');
      var keep = which === 'all'
        || (which === 'failed' && (stability === 'stable_fail' || stability === 'error'))
        || (which === 'flaky' && stability === 'flaky')
        || (which === 'stable' && stability === 'stable_pass');
      rows[i].style.display = keep ? '' : 'none';
    }
    for (var b = 0; b < buttons.length; b++) {
      buttons[b].setAttribute('aria-pressed', buttons[b].dataset.filter === which ? 'true' : 'false');
    }
  }
  for (var b = 0; b < buttons.length; b++) {
    buttons[b].addEventListener('click', function (event) {
      show(event.currentTarget.dataset.filter);
    });
  }
})();
"""
