
import json, torch, numpy as np, joblib
from pathlib import Path
from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

class EnsemblePredictor:
    def __init__(self, db_path=None):
        # Paths
        self.SGD_DIR = Path("./artifacts_tuned_sgd")
        self.NB_DIR  = Path("./artifacts_namebert_sa")
        self.ENS_DIR = Path("./artifacts_ensemble")

        # Load SGD vecs/models/calibrators
        self.vec_first = joblib.load(self.SGD_DIR / "vec_first_tfidf.joblib")
        self.vec_last  = joblib.load(self.SGD_DIR / "vec_last_tfidf.joblib")
        self.sgd_first = joblib.load(self.SGD_DIR / "sgd_first_tuned.joblib")
        self.sgd_last  = joblib.load(self.SGD_DIR / "sgd_last_tuned.joblib")
        self.cal_first = joblib.load(self.SGD_DIR / "cal_first_isotonic.joblib")
        self.cal_last  = joblib.load(self.SGD_DIR / "cal_last_isotonic.joblib")

        # Load NameBERT-SA
        self.tok = PreTrainedTokenizerFast.from_pretrained(str(self.NB_DIR / "tokenizer_hf"))
        cfg = RobertaConfig(
            vocab_size=len(self.tok), max_position_embeddings=50,
            hidden_size=384, num_hidden_layers=6, num_attention_heads=6, intermediate_size=768,
            hidden_act="gelu", attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1,
            type_vocab_size=1, pad_token_id=self.tok.pad_token_id, bos_token_id=self.tok.bos_token_id, eos_token_id=self.tok.eos_token_id,
            num_labels=2, problem_type="single_label_classification", id2label={0:"non_sa",1:"sa"}, label2id={"non_sa":0,"sa":1}
        )
        self.nb_model = RobertaForSequenceClassification(cfg)
        state = torch.load(self.NB_DIR / "cls_manual/roberta_cls_state.pt", map_location="cpu")
        self.nb_model.load_state_dict(state, strict=True)
        self.nb_model.eval()
        self.iso_nb = joblib.load(self.NB_DIR / "cal_isotonic_cls.joblib")

        # Ensemble config
        self.cfg = json.load(open(self.ENS_DIR / "inference_config.json"))
        self.THR = self.cfg["threshold"]
        self.ABSTAIN_L, self.ABSTAIN_H = self.cfg["abstain_band"]
        self.w_sgd = self.cfg.get("w_sgd", 1.0)
        self.w_nb  = self.cfg.get("w_nb", 1.0)

    @staticmethod
    def _split(name: str):
        name = (name or "").strip().lower()
        parts = name.split()
        if len(parts)==0: return "", ""
        if len(parts)==1: return parts[0], ""
        return parts[0], parts[-1]

    def _sgd_prob(self, name: str):
        first, last = self._split(name)
        Xfi = self.vec_first.transform([first])
        Xla = self.vec_last.transform([last])
        p_first = self.sgd_first.predict_proba(Xfi)[:,1]
        p_last  = self.sgd_last.predict_proba(Xla)[:,1]
        cp_first = np.clip(self.cal_first.predict(p_first), 0, 1)[0]
        cp_last  = np.clip(self.cal_last.predict(p_last),  0, 1)[0]
        return 1.0 - (1.0 - cp_first) * (1.0 - cp_last)

    def _nb_prob(self, name: str):
        first, last = self._split(name)
        text = f"<FIRST> {first} <SEP> <LAST> {last}"
        enc = self.tok(text, truncation=True, padding="max_length", max_length=48, return_tensors="pt")
        with torch.no_grad():
            logits = self.nb_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            p = torch.softmax(logits, dim=-1)[:,1].item()
        return float(np.clip(self.iso_nb.predict([p])[0], 0, 1))

    def predict(self, name: str):
        ps = self._sgd_prob(name)
        pn = self._nb_prob(name)
        p  = (self.w_sgd*ps + self.w_nb*pn) / (self.w_sgd + self.w_nb)

        if p < self.ABSTAIN_L: decision = "non_sa"
        elif p > self.ABSTAIN_H: decision = "sa"
        else: decision = "abstain"
        hard = "sa" if p >= self.THR else "non_sa"

        return {"name": name, "prob_sgd": ps, "prob_nb": pn, "prob_ens": p,
                "decision_abstain_band": decision,
                "hard_decision_at_threshold": {"threshold": self.THR, "label": hard}}
