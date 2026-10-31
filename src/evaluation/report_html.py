"""Turn summary.json into a one-page HTML report with charts (owner: Member 4).

    python -m src.evaluation.report_html results/unit1_all_methods/dev

Writes report.html next to summary.json. Self-contained (no internet needed),
works in light and dark mode, and every chart has a table view below it.
Vectorless methods are blue, embedding methods orange.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

DENSE = {"dense", "hybrid"}
KEY_METHODS = ["bm25", "bm25f+exp", "char_tfidf", "rrf", "dense", "hybrid"]


def family(label: str) -> str:
    return "emb" if label.split("+")[0].split("(")[0] in DENSE else "lex"


def bar_chart(title: str, unit: str, rows: list[tuple[str, float]], fmt: str = "{:.3f}") -> str:
    """Horizontal bars, one per method, value label at the bar end, hover tooltip."""
    vmax = max(v for _, v in rows) or 1.0
    bar_h, gap, left, width = 18, 8, 150, 640
    plot_w = width - left - 70
    height = len(rows) * (bar_h + gap) + 10
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    for i, (label, v) in enumerate(rows):
        y = 5 + i * (bar_h + gap)
        w = max(2.0, v / vmax * plot_w)
        text = fmt.format(v)
        tip = html.escape(f"{label}: {text} {unit}".strip())
        parts.append(
            f'<g class="bar" data-tip="{tip}"><rect class="hit" x="0" y="{y - gap / 2}" width="{width}" '
            f'height="{bar_h + gap}"/><text class="lbl" x="{left - 8}" y="{y + 13}" text-anchor="end">'
            f'{html.escape(label)}</text><path class="m {family(label)}" d="M{left},{y} h{w - 4} '
            f'a4,4 0 0 1 4,4 v{bar_h - 8} a4,4 0 0 1 -4,4 h-{w - 4} z"/><text class="val" x="{left + w + 6}" '
            f'y="{y + 13}">{text}</text></g>'
        )
    parts.append(f'<line class="axis" x1="{left}" y1="0" x2="{left}" y2="{height}"/></svg>')
    table = "".join(f"<tr><td>{html.escape(l)}</td><td>{fmt.format(v)}</td></tr>" for l, v in rows)
    return (f'<figure><figcaption>{html.escape(title)}</figcaption>{"".join(parts)}'
            f'<details><summary>Table view</summary><table><tr><th>method</th><th>{html.escape(unit or "value")}'
            f'</th></tr>{table}</table></details></figure>')


def tile(value: str, label: str, note: str) -> str:
    return f'<div class="tile"><div class="big">{value}</div><div class="tl">{label}</div><div class="note">{note}</div></div>'


def render(summary: dict) -> str:
    meta, res, sig = summary["meta"], summary["results"], summary["significance"]
    primary = meta["config"].get("primary_metric", "nDCG@10")
    order = sorted(res, key=lambda l: -res[l]["mean"][primary])
    key = [m for m in KEY_METHODS if m in res]
    s = sig.get(primary, {}).get("methods", {})
    tiles = ""
    if {"bm25", "dense"} <= res.keys() and "efficiency" in res["bm25"]:
        b, d = res["bm25"], res["dense"]
        be, de = b["efficiency"], d["efficiency"]
        diff = d["mean"][primary] - b["mean"][primary]
        p = s.get("dense", {}).get("p", float("nan"))
        tiles = "".join([
            tile(f"{diff:+.3f}", f"{primary}: embeddings vs BM25",
                 f"p = {p:.2f}: {'significant' if p < 0.05 else 'not significant'} on {meta['n_in_scope']} questions"),
            tile(f"{de['query_ms_median'] / be['query_ms_median']:.0f}×", "slower per question",
                 f"{be['query_ms_median']:.2f} ms vs {de['query_ms_median']:.1f} ms"),
            tile(f"{de['build_ms'] / be['build_ms']:.0f}×", "slower to index a PDF",
                 f"{be['build_ms']:.0f} ms vs {de['build_ms'] / 1000:.1f} s"),
            tile(f"{de['process_peak_mb'] / be['process_peak_mb']:.1f}×", "more peak memory",
                 f"{be['process_peak_mb']:.0f} MB vs {de['process_peak_mb']:.0f} MB, plus a "
                 f"{de['model_bytes'] / 2**20:.0f} MB model file"),
        ])
    charts = [bar_chart(f"Ranking quality: {primary} (higher is better)", "",
                        [(l, res[l]["mean"][primary]) for l in order])]
    budget = max(int(k.split("@")[1]) for k in res[order[0]]["mean"] if k.startswith("allEss@"))
    charts.append(bar_chart(f"Questions with all essential evidence inside {budget} tokens", "share of questions",
                            sorted(((l, res[l]["mean"][f"allEss@{budget}"]) for l in order), key=lambda r: -r[1]),
                            "{:.0%}"))
    groups = [g for g in res[order[0]]["groups"] if g in ("student wording", "book wording")]
    for g in groups:
        n = res[order[0]]["groups"][g]["n"]
        charts.append(bar_chart(f"{primary} on {g} questions (n = {n})", "",
                                sorted(((l, res[l]["groups"][g][primary]) for l in key), key=lambda r: -r[1])))
    if all("efficiency" in res[l] for l in key):
        charts.append(bar_chart("Query time, median (lower is better)", "ms",
                                [(l, res[l]["efficiency"]["query_ms_median"]) for l in key], "{:.2f} ms"))
        charts.append(bar_chart("Index build time for 3 PDFs (lower is better)", "ms",
                                [(l, res[l]["efficiency"]["build_ms"]) for l in key], "{:,.0f} ms"))
        charts.append(bar_chart("Peak process memory (lower is better)", "MB",
                                [(l, res[l]["efficiency"]["process_peak_mb"]) for l in key], "{:.0f} MB"))
    sig_rows = "".join(
        f"<tr><td>{html.escape(l)}</td><td>{r['diff']:+.3f}</td><td>[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]</td>"
        f"<td>{r['wins']}/{r['losses']}</td><td>{r['p']:.3f}</td><td>{r['p_holm']:.3f}</td>"
        f"<td>{'yes' if r['p_holm'] < 0.05 else 'no'}</td></tr>"
        for l in order if l in s for r in [s[l]])
    return PAGE.format(
        title=html.escape(meta["experiment"]), split=meta["split"], n=meta["n_in_scope"],
        paras=meta["n_paragraphs"], date=meta["date_utc"], tok=html.escape(meta["tokenizer_version"]),
        counter=meta["token_counter"], tiles=tiles, charts="".join(charts), primary=primary,
        baseline=sig.get(primary, {}).get("baseline", ""), sig_rows=sig_rows,
        machine=html.escape(res[order[0]].get("efficiency", {}).get("machine", "")),
    )


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Retrieval Evaluation Report</title>
<style>
:root {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#dddcd6; --lex:#2a78d6; --emb:#eb6834; --hover:#f0efec; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --line:#383835; --lex:#3987e5; --emb:#d95926; --hover:#262624; }} }}
:root[data-theme="dark"] {{ --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --line:#383835; --lex:#3987e5; --emb:#d95926; --hover:#262624; }}
body {{ margin:0; background:var(--surface); color:var(--ink); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:900px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:26px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:32px 0 8px; }}
.sub {{ color:var(--ink2); margin:0 0 20px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(190px, 1fr)); gap:12px; }}
.tile {{ border:1px solid var(--line); border-radius:10px; padding:14px; }}
.big {{ font-size:30px; font-weight:650; }} .tl {{ font-weight:600; }} .note {{ color:var(--ink2); font-size:13px; }}
.legend {{ display:flex; gap:18px; color:var(--ink2); font-size:13px; margin:8px 0; }}
.sw {{ display:inline-block; width:12px; height:12px; border-radius:3px; vertical-align:-1px; margin-right:6px; }}
figure {{ margin:22px 0; }} figcaption {{ font-weight:600; margin-bottom:6px; }}
svg {{ width:100%; height:auto; display:block; }} svg text {{ font-size:12px; fill:var(--ink2); }}
.m.lex {{ fill:var(--lex); }} .m.emb {{ fill:var(--emb); }} .hit {{ fill:transparent; }}
.bar:hover .hit {{ fill:var(--hover); }} .val {{ fill:var(--ink); }} .axis {{ stroke:var(--line); }}
details {{ font-size:13px; color:var(--ink2); }} table {{ border-collapse:collapse; margin:6px 0; }}
td, th {{ padding:3px 10px; border-bottom:1px solid var(--line); text-align:left; }}
#tip {{ position:fixed; pointer-events:none; background:var(--ink); color:var(--surface); padding:4px 8px;
  border-radius:6px; font-size:12px; display:none; }}
.scroll {{ overflow-x:auto; }}
</style></head><body><main>
<h1>Vectorless retrieval: how good, how light?</h1>
<p class="sub">Experiment <b>{title}</b>, {split} split: {n} questions over {paras} paragraphs from 3 OS PDFs.
Tokenizer {tok}; token counter {counter}; run {date}.</p>
<div class="tiles">{tiles}</div>
<div class="legend"><span><span class="sw" style="background:var(--lex)"></span>Vectorless (lexical, no model)</span>
<span><span class="sw" style="background:var(--emb)"></span>Embedding model (bge-small, 64 MB)</span></div>
{charts}
<h2>Are the differences real? ({primary} vs {baseline})</h2>
<p class="sub">Paired randomization test, Holm-corrected across all methods. "Significant" means Holm p &lt; 0.05.</p>
<div class="scroll"><table><tr><th>method</th><th>diff</th><th>95% CI</th><th>wins/losses</th><th>p</th><th>Holm p</th>
<th>significant</th></tr>{sig_rows}</table></div>
<p class="sub">Machine: {machine}</p>
</main><div id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('.bar').forEach(g => {{
  g.addEventListener('mousemove', e => {{ tip.textContent = g.dataset.tip; tip.style.display = 'block';
    tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; }});
  g.addEventListener('mouseleave', () => tip.style.display = 'none');
}});
</script></body></html>
"""


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/unit1_all_methods/dev")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    (out_dir / "report.html").write_text(render(summary), encoding="utf-8")
    print(f"Wrote {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
