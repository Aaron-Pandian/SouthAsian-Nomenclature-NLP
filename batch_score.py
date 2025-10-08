# batch_score.py
# Score a CSV of names with the calibrated ensemble and print a concise summary.
# Blank-name rows are INCLUDED in the output CSV but EXCLUDED from all totals/percentages.
#
# Usage:
#   python batch_score.py --in input.csv --out scored.csv --name-col "Provider Name"
# Optional:
#   --threshold 0.59 --abstain-low 0.49 --abstain-high 0.69 --summary-out summary.json
#   --include-params-in-output   # include threshold/abstain params per-row in the output CSV (default: OFF)

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import joblib
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

# Paths (adjust if needed)
ROOT      = Path(".")
SGD_DIR   = ROOT / "artifacts_tuned_sgd"
NB_DIR    = ROOT / "artifacts_namebert_sa"
ENS_DIR   = ROOT / "artifacts_ensemble"

class EnsemblePredictor:
    def __init__(self):
        # Load SGD (first/last TF-IDF + SGD + isotonic)
        self.vec_first = joblib.load(SGD_DIR / "vec_first_tfidf.joblib")
        self.vec_last  = joblib.load(SGD_DIR / "vec_last_tfidf.joblib")
        self.sgd_first = joblib.load(SGD_DIR / "sgd_first_tuned.joblib")
        self.sgd_last  = joblib.load(SGD_DIR / "sgd_last_tuned.joblib")
        self.cal_first = joblib.load(SGD_DIR / "cal_first_isotonic.joblib")
        self.cal_last  = joblib.load(SGD_DIR / "cal_last_isotonic.joblib")

        # Load NameBERT-SA
        self.tok = PreTrainedTokenizerFast.from_pretrained(str(NB_DIR / "tokenizer_hf"))
        self.MAX_LEN = 48
        cfg = RobertaConfig(
            vocab_size=len(self.tok), max_position_embeddings=self.MAX_LEN + 2,
            hidden_size=384, num_hidden_layers=6,
            num_attention_heads=6, intermediate_size=768,
            hidden_act="gelu", attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1, type_vocab_size=1,
            pad_token_id=self.tok.pad_token_id,
            bos_token_id=self.tok.bos_token_id,
            eos_token_id=self.tok.eos_token_id,
            num_labels=2, problem_type="single_label_classification",
            id2label={0: "non_sa", 1: "sa"},
            label2id={"non_sa": 0, "sa": 1},
        )
        self.nb_model = RobertaForSequenceClassification(cfg)
        state = torch.load(NB_DIR / "cls_manual" / "roberta_cls_state.pt", map_location="cpu")
        self.nb_model.load_state_dict(state, strict=True)
        self.nb_model.eval()
        self.iso_nb = joblib.load(NB_DIR / "cal_isotonic_cls.joblib")

        # Ensemble config (threshold, abstain band, weights)
        cfg_js = json.load(open(ENS_DIR / "inference_config.json"))
        self.default_thr = float(cfg_js["threshold"])
        band = cfg_js["abstain_band"]
        self.abstain_l, self.abstain_h = float(band[0]), float(band[1])
        self.w_sgd = float(cfg_js.get("w_sgd", 1.0))
        self.w_nb  = float(cfg_js.get("w_nb", 1.0))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nb_model.to(self.device)

    @staticmethod
    def _split(name: str):
        s = (name or "").strip().lower()
        parts = s.split()
        if len(parts) == 0: return "", ""
        if len(parts) == 1: return parts[0], ""
        return parts[0], parts[-1]

    def _sgd_prob(self, name: str) -> float:
        first, last = self._split(name)
        Xfi = self.vec_first.transform([first])
        Xla = self.vec_last.transform([last])
        p_first = self.sgd_first.predict_proba(Xfi)[:, 1]
        p_last  = self.sgd_last.predict_proba(Xla)[:, 1]
        cp_first = np.clip(self.cal_first.predict(p_first), 0, 1)[0]
        cp_last  = np.clip(self.cal_last.predict(p_last),  0, 1)[0]
        return float(1.0 - (1.0 - cp_first) * (1.0 - cp_last))  # probabilistic OR

    def _nb_prob(self, name: str) -> float:
        first, last = self._split(name)
        text = f"<FIRST> {first} <SEP> <LAST> {last}"
        enc = self.tok(text, truncation=True, padding="max_length",
                       max_length=self.MAX_LEN, return_tensors="pt")
        with torch.no_grad():
            logits = self.nb_model(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device)
            ).logits
            p = torch.softmax(logits, dim=-1)[:, 1].item()
        return float(np.clip(self.iso_nb.predict([p])[0], 0, 1))

    def predict(self, name: str, threshold: float, abstain_l: float, abstain_h: float):
        # NOTE: caller will handle blanks; this function assumes a non-blank name.
        ps = self._sgd_prob(name)
        pn = self._nb_prob(name)
        p  = (self.w_sgd * ps + self.w_nb * pn) / (self.w_sgd + self.w_nb)

        if p < abstain_l: decision = "non_sa"
        elif p > abstain_h: decision = "sa"
        else: decision = "abstain"
        hard = "sa" if p >= threshold else "non_sa"
        return p, ps, pn, decision, hard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV path")
    ap.add_argument("--name-col", dest="name_col", default="name", help="Column with full name")
    ap.add_argument("--threshold", type=float, default=None, help="Override decision threshold (0..1)")
    ap.add_argument("--abstain-low", type=float, default=None, help="Override abstain low")
    ap.add_argument("--abstain-high", type=float, default=None, help="Override abstain high")
    ap.add_argument("--summary-out", type=str, default=None, help="Optional path to write a JSON summary")
    ap.add_argument("--include-params-in-output", dest="include_params_in_output",
                    action="store_true",
                    help="Include threshold/abstain params per-row in the output CSV (default: not included)")
    args = ap.parse_args()

    pred = EnsemblePredictor()
    thr   = pred.default_thr if args.threshold is None else float(args.threshold)
    a_low = pred.abstain_l   if args.abstain_low is None else float(args.abstain_low)
    a_high= pred.abstain_h   if args.abstain_high is None else float(args.abstain_high)

    # Read
    with open(args.inp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if args.name_col not in reader.fieldnames:
            raise ValueError(f"Column '{args.name_col}' not found. Available: {reader.fieldnames}")
        input_rows = list(reader)

    # Score
    scored_rows = []
    n_total_rows = len(input_rows)
    n_skipped_blank = 0
    n_hard_sa = 0

    # Decision counts over NON-BLANK rows only
    decision_counts = {"sa": 0, "non_sa": 0, "abstain": 0}

    for r in input_rows:
        name = r.get(args.name_col, "")
        is_blank = (not name) or (not str(name).strip())

        if is_blank:
            n_skipped_blank += 1
            # Preserve the row in output but leave prob fields empty and mark decision/hard label as blank
            r_out = dict(r)
            r_out.update({
                "prob_ensemble": "",
                "prob_sgd": "",
                "prob_namebert": "",
                "decision_abstain_band": "",
                "hard_label": "",
            })
            if args.include_params_in_output:
                r_out.update({
                    "threshold_used": thr,
                    "abstain_low": a_low,
                    "abstain_high": a_high,
                })
            scored_rows.append(r_out)
            continue  # do NOT count toward totals/percentages

        # Non-blank -> score
        p, ps, pn, decision, hard = pred.predict(name, thr, a_low, a_high)
        if hard == "sa":
            n_hard_sa += 1
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

        # Row output
        r_out = dict(r)
        r_out.update({
            "prob_ensemble": f"{p:.6f}",
            "prob_sgd": f"{ps:.6f}",
            "prob_namebert": f"{pn:.6f}",
            "decision_abstain_band": decision,
            "hard_label": hard,
        })
        if args.include_params_in_output:
            r_out.update({
                "threshold_used": thr,
                "abstain_low": a_low,
                "abstain_high": a_high,
            })
        scored_rows.append(r_out)

    # Write CSV
    if len(scored_rows) > 0:
        out_fields = list(scored_rows[0].keys())
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(scored_rows)

    # Summary over NON-BLANK rows only
    n_effective = n_total_rows - n_skipped_blank
    pct = (lambda k: (decision_counts[k] / n_effective * 100.0) if n_effective else 0.0)

    summary = {
        "input_rows": n_total_rows,
        "non_blank_rows_scored": n_effective,
        "skipped_blank_names": n_skipped_blank,
        "operating_params": {
            "threshold": thr,
            "abstain_low": a_low,
            "abstain_high": a_high
        },
        "decision_breakdown": {
            "sa": {"count": decision_counts["sa"], "percent": round(pct("sa"), 3)},
            "non_sa": {"count": decision_counts["non_sa"], "percent": round(pct("non_sa"), 3)},
            "abstain": {"count": decision_counts["abstain"], "percent": round(pct("abstain"), 3)},
        },
        "hard_label_sa": {
            "count": n_hard_sa,
            "of_non_blank": n_effective,
            "percent": round((n_hard_sa / n_effective * 100.0) if n_effective else 0.0, 3)
        },
        "output_csv": str(Path(args.out).resolve())
    }

    # Print summary
    print("\n=== South Asian Name Classification — Run Summary ===")
    print(f"Input rows (total):    {summary['input_rows']}")
    print(f"Non-blank scored:      {summary['non_blank_rows_scored']}")
    print(f"Skipped (blank names): {summary['skipped_blank_names']}")
    print(f"Operating params:      threshold={thr:.4f}, abstain_low={a_low:.4f}, abstain_high={a_high:.4f}")
    print("Decision breakdown (non-blank only):")
    print(f"  SA:       {summary['decision_breakdown']['sa']['count']:>6}  "
          f"({summary['decision_breakdown']['sa']['percent']:.2f}%)")
    print(f"  non-SA:   {summary['decision_breakdown']['non_sa']['count']:>6}  "
          f"({summary['decision_breakdown']['non_sa']['percent']:.2f}%)")
    print(f"  abstain:  {summary['decision_breakdown']['abstain']['count']:>6}  "
          f"({summary['decision_breakdown']['abstain']['percent']:.2f}%)")
    print(f"hard_label == 'sa': {summary['hard_label_sa']['count']} / {summary['hard_label_sa']['of_non_blank']} "
          f"({summary['hard_label_sa']['percent']:.2f}%)")
    print(f"Output CSV: {summary['output_csv']}\n")

    # Optional: JSON summary
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, indent=2)
        print(f"Summary JSON written to: {Path(args.summary_out).resolve()}")

if __name__ == "__main__":
    main()