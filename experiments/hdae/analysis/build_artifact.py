"""Generate the HDAE readers + control artifact from the result JSONs.

Tabs are emitted from data so the page can be rebuilt and republished unchanged when the
g-sweep adds a third tab.
"""
import json, os, sys, numpy as np
P = "experiments/hdae/outputs/attr_predictors_celebahq"
SP = "/tmp/claude-1001/-home-exouser/ee943fc6-c5ef-4405-b557-1b557434dfe9/scratchpad"
fe = json.load(open(f"{P}/full_eval.json"))
ctrl = json.load(open("experiments/hdae/outputs/nullctrl/nullctrl.json"))
meas = json.load(open(f"{SP}/bigmeasure.json"))
_gp = "experiments/hdae/outputs/nullctrl/gsweep_attrs.json"
raw = json.load(open(_gp)) if os.path.exists(_gp) else None
# collapse the per-(attr,g) cells into a best-g pick per attribute, by CF1 against FC_unobs
_ap = "experiments/hdae/outputs/nullctrl/ablation_null4.json"
abl = json.load(open(_ap)) if os.path.exists(_ap) else {}
gsw = None
if raw:
    by = {}
    for k, v in raw.items():
        by.setdefault(v["attr"], []).append(v)
    gsw = {a: max(c, key=lambda x: x["CF1"]) for a, c in by.items() if c}
    gsw_all = by

ok = {a: v for a, v in fe.items() if v.get("recon_tuned") and v.get("test_tuned")}
THIN = [a for a, v in ok.items() if v["n_pos_recon"] < 70]
NOISY = [a for a, v in ok.items() if v["recon_tuned"]["bal"] < 0.75 and v["n_pos_recon"] >= 70]
KEEP = sorted([a for a in ok if a not in THIN and a not in NOISY])
UNSCORED = [a for a in fe if a not in ok]
PRETTY = {"null1_target": "{Bald}", "null2_target_beard": "{Bald, Beard}", "null4_all": "{all 4}"}

def esc(s): return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---------- tab 1 ----------
rows = sorted(ok.items(), key=lambda x: -x[1]["recon_tuned"]["bal"])
t1 = []
for a, v in rows:
    rt, ru, tt, tu = v["recon_tuned"], v["recon_untuned"], v["test_tuned"], v["test_untuned"]
    cls = "cut-thin" if a in THIN else ("cut-noisy" if a in NOISY else "keep")
    tag = "thin" if a in THIN else ("noisy" if a in NOISY else "")
    t1.append(
      f'<tr class="{cls}"><td>{esc(a)}{" <b class=obs>obs</b>" if v["observed"] else ""}</td>'
      f'<td class="n">{v["n_pos_recon"]}</td>'
      f'<td class="n">{tu["bal"]:.3f}</td><td class="n">{tt["bal"]:.3f}</td>'
      f'<td class="n hi">{ru["bal"]:.3f}</td><td class="n">{rt["bal"]:.3f}</td>'
      f'<td class="n">{rt["f1"]:.3f}</td><td class="n">{rt["prec"]:.2f}</td><td class="n">{rt["rec"]:.2f}</td>'
      f'<td class="n">{rt["pred_rate"]:.2f}/{rt["base"]:.2f}</td>'
      f'<td class="tag">{tag}</td></tr>')
means = {k: np.mean([v[k]["bal"] for v in ok.values()]) for k in
         ("test_untuned", "test_tuned", "recon_untuned", "recon_tuned")}
helps = sum(v["recon_tuned"]["bal"] > v["recon_untuned"]["bal"] for v in ok.values())
keep_bal = np.mean([ok[a]["recon_tuned"]["bal"] for a in KEEP])
keep_f1 = np.mean([ok[a]["recon_tuned"]["f1"] for a in KEEP])

# ---------- tab 2 ----------
t2, floor_rows = [], []
for k in meas:
    if k not in ctrl: continue
    s, g = k.split("|")
    mo, fo = meas[k]["FC_obs"], ctrl[k]["FC_obs"]
    mu, fu = meas[k]["FC_unobs_reliable"], ctrl[k]["FC_unobs_reliable"]
    t2.append(f'<tr><td>{PRETTY[s]}</td><td class="n">{g[1:]}</td>'
              f'<td class="n">{mo:.4f}</td><td class="n">{fo:.4f}</td><td class="n hi">{mo/fo:.4f}</td>'
              f'<td class="n">{mu:.4f}</td><td class="n">{fu:.4f}</td><td class="n hi">{mu/fu:.4f}</td></tr>')
for k in sorted(ctrl):
    s, g = k.split("|")
    floor_rows.append(f'<tr><td>{PRETTY[s]}</td><td class="n">{g[1:]}</td>'
                      f'<td class="n">{ctrl[k]["CC"]:.4f}</td><td class="n">{ctrl[k]["FC_obs"]:.4f}</td>'
                      f'<td class="n">{ctrl[k]["FC_unobs_reliable"]:.4f}</td></tr>')
n_ctrl = len(ctrl)

tab3_nav = '<button class="tb" data-t="3">Best g per attribute</button>' if gsw else ''
tab3 = ""
if gsw:
    gr = "".join(f'<tr><td>{esc(a)}</td><td class="n hi">{v["g"]:g}</td><td class="n">{v["CC"]:.4f}</td>'
                 f'<td class="n">{v["FC_obs"]:.4f}</td><td class="n">{v["FC_unobs"]:.4f}</td>'
                 f'<td class="n hi">{v["CF1"]:.4f}</td></tr>' for a, v in sorted(gsw.items()))
    ab = ""
    for k, v in sorted(abl.items()):
        a = v["attr"]
        o = next((x for x in raw.values() if x["attr"] == a and x["g"] == v["g"]), None)
        if not o: continue
        win = "target-only" if o["CF1"] > v["CF1"] else "all-4"
        ab += (f'<tr><td>{esc(a)}</td><td class="n">{v["g"]:g}</td>'
               f'<td class="n hi">{o["CC"]:.4f}</td><td class="n">{v["CC"]:.4f}</td><td class="n">{v["CC"]-o["CC"]:+.4f}</td>'
               f'<td class="n hi">{o["FC_unobs"]:.4f}</td><td class="n">{v["FC_unobs"]:.4f}</td>'
               f'<td class="n hi">{o["CF1"]:.4f}</td><td class="n">{v["CF1"]:.4f}</td>'
               f'<td class="n">{v["CF1"]-o["CF1"]:+.4f}</td><td class="tag">{win}</td></tr>')
    ABL = ""
    if ab:
        done = "all four attributes" if len(abl) >= 4 else f"{len(abl)} of 4 attributes so far"
        ABL = (f'<h3 style="margin-top:30px">Null-set ablation &mdash; {done}</h3>'
               '<p class="sub">The same cell re-run with the unconditional branch nulling ALL FOUR '
               'attributes instead of one. Matched cohort, x<sub>T</sub>, readers and g, so the null '
               'set is the only variable.</p>'
               '<div class="tw"><table><thead><tr><th>attribute</th><th>g</th>'
               '<th>CC target</th><th>CC all-4</th><th>&Delta;CC</th>'
               '<th>FCu target</th><th>FCu all-4</th>'
               '<th>CF1 target</th><th>CF1 all-4</th><th>&Delta;CF1</th><th>winner</th>'
               f'</tr></thead><tbody>{ab}</tbody></table></div>'
               '<div class="note"><h3>FC favours all-4, and that is an artifact</h3>'
               '<p>Nulling all four makes guidance amplify the other attributes toward their '
               '<em>conditioned</em> values, which are unchanged &mdash; so a binary reader still '
               'answers &ldquo;same&rdquo; and scores them preserved. It cannot see an attribute '
               'intensified <em>within</em> its class, which is exactly the fuller-beard effect the '
               'qualitative grids showed. CF1, which also weighs whether the edit landed, favours '
               'target-only.</p></div>')
    full = ""
    for a in sorted(gsw_all):
        for v in sorted(gsw_all[a], key=lambda x: x["g"]):
            star = ' class="hi"' if v["g"] == gsw[a]["g"] else ""
            full += (f'<tr><td>{esc(a)}</td><td class="n"{star}>{v["g"]:g}</td>'
                     f'<td class="n">{v["CC"]:.4f}</td><td class="n">{v["FC_obs"]:.4f}</td>'
                     f'<td class="n">{v["FC_unobs"]:.4f}</td><td class="n">{v["CF1"]:.4f}</td></tr>')
    tab3 = f'''<section id="t3" hidden><h2>Best guidance per attribute</h2>
<p class="sub">Target-only nulling, conditional x<sub>T</sub>, per-attribute readers. Each attribute's
guidance chosen by CF1.</p>
<div class="tw"><table><thead><tr><th>attribute</th><th>best g</th><th>CC</th><th>FC_obs</th>
<th>FC_unobs</th><th>CF1</th></tr></thead><tbody>{gr}</tbody></table></div>
<div class="note"><h3>Thresholds differ by role, on purpose</h3>
<p>The attribute being intervened on is read at its <b>F1-optimal</b> threshold, because CC counts
positive readings and a permissive reader inflates it — the Bald reader ran precision 0.44 against a
6% base rate under balanced tuning. Every FC reader is read <b>untuned</b>, because tuning on
photographs transfers badly to generated images (it helped only 15 of 37 attributes).</p></div>
{ABL}
<h3 style="margin-top:26px">Full grid</h3>
<p class="sub">Every cell. The highlighted g is the CF1 pick for that attribute.</p>
<div class="tw"><table><thead><tr><th>attribute</th><th>g</th><th>CC</th><th>FC_obs</th>
<th>FC_unobs</th><th>CF1</th></tr></thead><tbody>{full}</tbody></table></div></section>'''

html = f'''<title>Readers and the FC Floor</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{{color-scheme:light;--bg:#F6F8F9;--surface:#FFF;--ink:#14171C;--muted:#5F6874;--line:#E1E6EA;
--line2:#EDF1F3;--accent:#0F7A87;--accent-soft:#E2F1F3;--good:#137A5C;--good-soft:#E1F2EC;
--bad:#AF432B;--bad-soft:#F8E7E2;--warn-soft:#FBF0DC;--warn:#8A6520;
--sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,monospace;}}
@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;--bg:#0F1217;
--surface:#161A20;--ink:#E7ECF1;--muted:#98A2B0;--line:#252B34;--line2:#1D222A;--accent:#45B8C6;
--accent-soft:#123034;--good:#43BB91;--good-soft:#0F2E25;--bad:#E0765B;--bad-soft:#341A14;
--warn-soft:#332814;--warn:#D9AC63;}}}}
:root[data-theme=dark]{{color-scheme:dark;--bg:#0F1217;--surface:#161A20;--ink:#E7ECF1;--muted:#98A2B0;
--line:#252B34;--line2:#1D222A;--accent:#45B8C6;--accent-soft:#123034;--good:#43BB91;--good-soft:#0F2E25;
--bad:#E0765B;--bad-soft:#341A14;--warn-soft:#332814;--warn:#D9AC63;}}
body{{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6;
padding-block:0 64px;padding-left:16px;padding-right:16px;}}
.wrap{{max-width:1080px;margin:0 auto}}
.hero{{padding-block:44px 22px}}
.eyebrow{{font-family:var(--mono);font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--accent);margin:0 0 12px}}
h1{{font-size:clamp(30px,5vw,42px);line-height:1.1;letter-spacing:-.02em;font-weight:600;margin:0 0 14px;text-wrap:balance}}
.lede{{font-size:17.5px;color:var(--muted);max-width:66ch;margin:0}}
h2{{font-size:21px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em}}
h3{{font-size:14.5px;font-weight:600;margin:0 0 7px}}
.sub{{color:var(--muted);font-size:14px;margin:0 0 18px;max-width:70ch}}
p{{max-width:70ch}}
nav{{display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-top:8px;
position:sticky;top:env(safe-area-inset-top,0px);background:var(--bg);z-index:5}}
.tb{{font-family:var(--sans);font-size:14px;font-weight:500;color:var(--muted);background:none;
border:none;border-bottom:2px solid transparent;padding:10px 14px;cursor:pointer}}
.tb[aria-selected=true]{{color:var(--accent);border-bottom-color:var(--accent)}}
.tb:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
section{{padding-block:28px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}}
.tile{{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:15px 17px}}
.tile .k{{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}}
.tile .v{{font-family:var(--mono);font-size:25px;font-weight:600;line-height:1;font-variant-numeric:tabular-nums}}
.tile .n{{font-size:12.5px;color:var(--muted);margin:8px 0 0;line-height:1.45}}
.up{{color:var(--good)}}.dn{{color:var(--bad)}}
.tw{{overflow-x:auto;margin:16px 0;border:1px solid var(--line);border-radius:9px;background:var(--surface)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:8px 12px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line2)}}
th:first-child,td:first-child{{text-align:left}}
thead th{{font-family:var(--mono);font-size:10px;letter-spacing:.07em;text-transform:uppercase;
color:var(--muted);font-weight:500;background:var(--line2);position:sticky;top:0}}
tbody tr:last-child td{{border-bottom:none}}
td.n{{font-family:var(--mono);font-variant-numeric:tabular-nums}}
td.hi{{background:var(--accent-soft);font-weight:600}}
tr.cut-thin td{{background:var(--warn-soft)}}
tr.cut-noisy td{{background:var(--bad-soft)}}
td.tag{{font-family:var(--mono);font-size:10px;color:var(--muted);text-transform:uppercase}}
b.obs{{font-family:var(--mono);font-size:9.5px;color:var(--accent);background:var(--accent-soft);
padding:1px 4px;border-radius:3px;font-weight:600;margin-left:4px}}
.note{{background:var(--surface);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:0 9px 9px 0;padding:15px 17px;margin:16px 0}}
.note.warn{{border-left-color:var(--bad)}}
.note p{{margin:0;font-size:13.5px;color:var(--muted)}}
.note p+p{{margin-top:8px}}
code{{font-family:var(--mono);font-size:.88em;background:var(--line2);padding:1.5px 5px;border-radius:4px}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);margin:10px 0}}
.sw{{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:5px}}
.foot{{color:var(--muted);font-size:12px;font-family:var(--mono);padding-top:22px}}
</style>
<div class="wrap">
<header class="hero">
<p class="eyebrow">HDAE · CelebA-HQ 256 · cd015r</p>
<h1>Readers and the FC floor</h1>
<p class="lede">Forty per-attribute readers scored on the distribution they are actually used in,
and the first measurement of how much of the counterfactual metric was instrument noise.</p>
</header>
<nav role="tablist">
<button class="tb" data-t="1" aria-selected="true">Predictor evaluation</button>
<button class="tb" data-t="2" aria-selected="false">Null-intervention control</button>
{tab3_nav}
</nav>

<section id="t1">
<h2>Forty readers, four ways</h2>
<p class="sub">One ConvNeXt-Small per attribute (50.2M, native 256px, degradation augmentation,
EMA weights), fit on 90% of partition 0 and selected on the remaining 10% — partitions 1 and 2
are untouched. Scored on the real test split and on 2,048 model reconstructions. Thresholds are
tuned on the real split and applied unchanged to reconstructions, never fitted to them.</p>
<div class="tiles">
<div class="tile"><p class="k">Mean BAL · real test</p><p class="v">{means["test_tuned"]:.3f}</p><p class="n">tuned; untuned {means["test_untuned"]:.3f}</p></div>
<div class="tile"><p class="k">Mean BAL · reconstructions</p><p class="v">{means["recon_untuned"]:.3f}</p><p class="n">untuned. Tuned is <em>worse</em>: {means["recon_tuned"]:.3f}</p></div>
<div class="tile"><p class="k">Tuning helps on recon</p><p class="v dn">{helps}/{len(ok)}</p><p class="n">thresholds fitted on photos do not transfer</p></div>
<div class="tile"><p class="k">Recommended keep</p><p class="v up">{len(KEEP)}</p><p class="n">mean BAL {keep_bal:.3f}, F1 {keep_f1:.3f}</p></div>
</div>
<div class="note warn"><h3>Threshold tuning on real images is counterproductive here</h3>
<p>It raises balanced accuracy on photographs ({means["test_untuned"]:.4f} → {means["test_tuned"]:.4f})
and <em>lowers</em> it on reconstructions ({means["recon_untuned"]:.4f} → {means["recon_tuned"]:.4f}),
helping only {helps} of {len(ok)} attributes. Every reading in the counterfactual pipeline comes off
a generated image, so the untuned threshold is the better default and the tuning step should be dropped.</p></div>
<div class="legend">
<span><i class="sw" style="background:var(--warn-soft);border:1px solid var(--line)"></i>too few positives (n⁺&lt;70)</span>
<span><i class="sw" style="background:var(--bad-soft);border:1px solid var(--line)"></i>noisy on reconstructions (BAL&lt;0.75)</span>
<span><b class="obs">obs</b> conditioned on by the model</span></div>
<div class="tw"><table><thead><tr><th>attribute</th><th>n⁺</th>
<th>test untuned</th><th>test tuned</th><th>recon untuned</th><th>recon tuned</th>
<th>F1 recon</th><th>prec</th><th>rec</th><th>rate/base</th><th></th></tr></thead>
<tbody>{"".join(t1)}</tbody></table></div>
<div class="note warn"><h3>Correction: there is no domain gap</h3>
<p>An earlier version of this page reported that the readers lose about 5 points of balanced
accuracy on generated images. That was an <b>error</b>. It compared the 3,000-image mixed-sex test
split against the 2,048 all-male cohort and attributed the difference to real-vs-generated, when
it was almost entirely cohort composition &mdash; attribute base rates differ sharply between men
and the general population.</p>
<p>Measured correctly &mdash; the <em>same</em> 2,048 subjects, their source photographs against
their reconstructions, across the 35 attributes with at least 30 positives and 30 negatives:
<b>source 0.8473, reconstruction 0.8462, mean gap &minus;0.0010</b>. <b>Zero</b> of 35 attributes
lose more than 5 points; the worst loses 0.022 and several gain. The readers transfer to generated
images essentially perfectly.</p>
<p>The columns below still compare the test split to the cohort, so differences between them
reflect cohort difficulty, not generation.</p></div>
<div class="note"><h3>Two ways a reader corrupts FC, in opposite directions</h3>
<p><b>Noisy</b> readers flip on near-identical images and <em>manufacture</em> drift the model never
caused. <b>Saturated</b> readers never change their answer and <em>hide</em> drift that it did —
invisible in balanced accuracy, which is why prediction rate is shown against base rate. No reader
in this set is saturated; that failure was v1's Blurry head (precision 0.024, recall 1.000).</p>
<p>Unscorable in an all-male cohort: {", ".join(f"<code>{esc(a)}</code>" for a in UNSCORED)}.</p></div>
</section>

<section id="t2" hidden>
<h2>How much of the metric was noise</h2>
<p class="sub">FC counts how often a reader gives the same answer on the reconstruction and on the
counterfactual. But a reader disagrees with itself on two near-identical images, and guidance
perturbs the image even when the requested attribute value is the one already present. The control
runs the identical pipeline with <code>do(Bald = its original value)</code> — same cohort, same
cached x<sub>T</sub>, same readers, nothing intervened on. Whatever falls below 1.0 is floor.</p>
<div class="tiles">
<div class="tile"><p class="k">Control CC (sanity)</p><p class="v up">{min(v["CC"] for v in ctrl.values()):.3f}+</p><p class="n">requesting the existing value reads back correctly</p></div>
<div class="tile"><p class="k">FC floor at g=5</p><p class="v">{ctrl.get("null1_target|g5",{}).get("FC_obs",float("nan")):.3f}</p><p class="n">{{Bald}} null set, no intervention</p></div>
<div class="tile"><p class="k">Predicted floor</p><p class="v dn">0.654</p><p class="n">my estimate from reader error alone — too pessimistic by ~0.27</p></div>
<div class="tile"><p class="k">Configs measured</p><p class="v">{n_ctrl}/9</p><p class="n">2,048 subjects each</p></div>
</div>
<h3 style="margin-top:26px">Normalised FC</h3>
<p class="sub"><code>normalised = measured / floor</code> — the share of <em>achievable</em> preservation
retained. Not <code>(m−f)/(1−f)</code>, which I proposed first and which goes negative here because
the floor is a ceiling on preservation, not a baseline to subtract.</p>
<div class="tw"><table><thead><tr><th>null set</th><th>g</th>
<th>FC_obs</th><th>floor</th><th>normalised</th>
<th>FC_unobs</th><th>floor</th><th>normalised</th></tr></thead>
<tbody>{"".join(t2)}</tbody></table></div>
<h3 style="margin-top:26px">The floor itself</h3>
<p class="sub">Nothing is intervened on in any of these rows. Everything below 1.0 is the pipeline
disturbing the image on its own.</p>
<div class="tw"><table><thead><tr><th>null set</th><th>g</th><th>CC (sanity)</th>
<th>FC_obs floor</th><th>FC_unobs floor</th></tr></thead><tbody>{"".join(floor_rows)}</tbody></table></div>
<div class="note"><h3>Guidance damages the image with no intervention at all</h3>
<p>For the {{Bald}} null set the floor falls monotonically with guidance — 0.9627 at g=3, 0.9435 at
g=4, 0.9232 at g=5. That is pure guidance cost, separated from the edit for the first time.</p></div>
<div class="note"><h3>The null set perturbs attributes even when nothing is intervened on</h3>
<p>At the same g=3 the floor is 0.9627 for {{Bald}} but 0.9194 for {{Bald, Beard}}. Adding Beard to
the null set costs 4.3 points of preservation <em>without any Bald flip</em> — the amplification
mechanism, measured rather than observed.</p></div>
</section>
{tab3}
<p class="foot">2,048 held-out males · conditional x<sub>T</sub> · T=50 DDIM · readers: ConvNeXt-Small per attribute</p>
</div>
<script>
const tabs=[...document.querySelectorAll('.tb')];
function show(t){{tabs.forEach(b=>b.setAttribute('aria-selected',b.dataset.t===t));
['1','2','3'].forEach(i=>{{const s=document.getElementById('t'+i); if(s) s.hidden=(i!==t);}});}}
tabs.forEach(b=>b.addEventListener('click',()=>show(b.dataset.t)));
</script>'''
open(f"{SP}/readers_control.html","w").write(html)
print(f"wrote readers_control.html  ({len(html)} bytes)  tabs={'3' if gsw else '2'}  control configs={n_ctrl}")
print(f"KEEP {len(KEEP)} | THIN {len(THIN)} | NOISY {len(NOISY)} | unscored {len(UNSCORED)}")
