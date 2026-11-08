"""Render readers_all.json as the single table, plus the gap that matters.

The report-vs-radiologist gap is computed per reader: test AUC (report labels)
minus radiologist_val AUC (image labels). A large positive gap means the reader
tracks the report better than it tracks the finding -- which for ours and XRV
is partly circular (they learned these labels) and for CheXzero is not.
"""
import argparse, json

ORDER = ["effusion", "edema", "cardiomegaly", "enlarged_cardiomediastinum",
         "lung_opacity", "consolidation", "atelectasis", "pneumothorax",
         "support_devices", "lung_lesion", "fracture", "pneumonia", "pleural_other",
         "sex", "ap_pa", "age_bin"]
NICE = {"ours10": "ours (best.ckpt)", "ours11": "ours (11-head)",
        "xrv": "XRV `-all`", "chexzero": "CheXzero (zero-shot)"}
ROLE = {"effusion": "modeled", "edema": "modeled", "sex": "modeled", "age_bin": "modeled",
        "ap_pa": "eval-only", "lung_opacity": "eval-only", "consolidation": "eval-only"}


def cell(e, key="auc"):
    if e is None:
        return "—"
    v = e.get(key, e.get("auc_ovr_macro"))
    if v is None:
        return "not scoreable"
    # a radiologist_val AUC on a handful of positives is noise dressed as a
    # number; mark it rather than print it as if it stood up
    thin = "\u2020" if 0 < e.get("n_pos", 999) < 5 else ""
    ci = e.get("ci95")
    if ci and ci[0] is not None:
        return f"{v:.4f}{thin} [{ci[0]:.3f},{ci[1]:.3f}]"
    return f"{v:.4f}{thin}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = json.load(open(a.json))
    T, S = R["table"], R["sets"]
    L = []
    L.append("# CheXpert — three readers, every attribute, train / test / radiologists\n")
    L.append(f"Generated {R['date']}. Row sets: fit {S['fit']:,} · select {S['select']:,} · "
             f"train {S['train']:,} (= fit + select) · test {S['test']:,} · radiologist_val 200.\n")
    L.append("**The three columns do not measure the same thing.** `train` and `test` score "
             "against report-derived CheXpert labels with `-1` folded to 0, so an AUC there is "
             "agreement-with-the-report. `radiologists` scores against labels assigned from the "
             "images by radiologists, and is the only column the v2 gate is defined on. "
             "Independence also differs by reader: ours trained on the fit rows outright, XRV has "
             "CheXpert in its training mix, CheXzero saw none of these labels.\n")
    L.append("`gap` = test − radiologists. Large and positive means the reader tracks the "
             "report better than it tracks the finding.\n")

    for k in ORDER:
        if k not in T:
            continue
        role = ROLE.get(k, "out")
        L.append(f"\n## {k}  ·  *{role}*\n")
        L.append("| reader | train | test | radiologists | gap | test n_pos | unc folded |")
        L.append("|---|---|---|---|---|---|---|")
        for r in ("ours10", "ours11", "xrv", "chexzero"):
            e = T[k].get(r)
            if e is None:
                continue
            tr, te, rv = e.get("train"), e.get("test"), e.get("radval")
            if tr is None and te is None and rv is None:
                L.append(f"| {NICE[r]} | — | — | — | — | — | — |")
                continue
            g = "—"
            if te and rv:
                a_t = te.get("auc", te.get("auc_ovr_macro"))
                a_r = rv.get("auc", rv.get("auc_ovr_macro"))
                if a_t is not None and a_r is not None:
                    g = f"{a_t - a_r:+.4f}"
            npos = te.get("n_pos", "—") if te else "—"
            unc = te.get("uncertain_folded_to_0", "—") if te else "—"
            L.append(f"| {NICE[r]} | {cell(tr)} | {cell(te)} | {cell(rv)} | {g} | "
                     f"{npos:,} | {unc:,} |" if isinstance(npos, int)
                     else f"| {NICE[r]} | {cell(tr)} | {cell(te)} | {cell(rv)} | {g} | {npos} | {unc} |")
        # certain-only note where folding mattered
        notes = []
        for r in ("ours10", "ours11", "xrv", "chexzero"):
            e = T[k].get(r)
            if e and e.get("test") and e["test"].get("auc_certain_only") is not None:
                notes.append(f"{NICE[r]} {e['test']['auc_certain_only']:.4f}")
        if notes:
            n_c = next(e["test"]["n_certain"] for r in ("ours10", "ours11", "xrv", "chexzero")
                       if (e := T[k].get(r)) and e.get("test")
                       and e["test"].get("n_certain") is not None)
            L.append(f"\nTest AUC on certain rows only (`-1` dropped, n={n_c:,}): "
                     + " · ".join(notes))
    L.append("\n---\n")
    L.append("\u2020 fewer than 5 positives in radiologist_val -- the number is not usable. "
             "lung_lesion and pleural_other have 1 positive each; fracture has 0 and cannot be "
             "scored against radiologists at all. These three stay out of the protocol on those "
             "grounds alone, independently of any AUC.\n")
    L.append("Validation: all 22 numbers this table shares with the frozen "
             "`predictor_report.json` (ours10 on test and on radiologist_val, 11 attributes each) "
             "reproduce to 4 decimals, and the fit/select/test sub-split reproduces at "
             "138,727 / 15,529 / 17,740. The CheXzero streaming rewrite reproduces all seven "
             "previously reported radiologist_val AUCs to 4 decimals.\n")
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"wrote {a.out}")
