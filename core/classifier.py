import torch
import spaces
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from collections import OrderedDict
import os
from core.model.load_model import RFModel
from core.utils import render_result_to_html, clean_and_segment_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


class AIContentClassifier:
    def __init__(self, device=DEVICE, model_id="gpt2"):
        self.device = device
        self.model_id = model_id
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.ml_model = RFModel()

    def get_result(self, result):
        likelihood_score = result["likelihood_score"]
        return (
            ("AI-generated", 1) if likelihood_score >= 0.5 else ("Human-generated", 0)
        )

    @spaces.GPU
    def get_pplx_map(self, lines):
        pplx_map = OrderedDict()
        for line in lines:
            ppl = self.get_ppl(line)
            if ppl == -1:
                continue
            pplx_map[line] = ppl
        result = {
            "pplx_map": pplx_map,
            "burstiness": max(pplx_map.values()),
            "average_pplx": sum(pplx_map.values()) / len(pplx_map),
        }
        return result

    def has_three_consecutive_low_pplx(self, pplx_list, threshold=40, count=3):
        consecutive = 0
        for pplx in pplx_list:
            if pplx < threshold:
                consecutive += 1
                if consecutive >= count:
                    return 1
            else:
                consecutive = 0
        return 0

    def get_likelihood(self, result: dict):
        low_pplx_flag = self.has_three_consecutive_low_pplx(
            list(result["pplx_map"].values())
        )
        features = [
            result["average_pplx"],
            result["burstiness"],
            len(result["pplx_map"]),
            low_pplx_flag,
        ]
        likelihood_score = self.ml_model.predict(features)
        return round(likelihood_score, 2)

    def classify(self, sentence):
        lines = clean_and_segment_text(sentence)
        if len(lines) < 5 or len(" ".join(lines)) < 100:
            return {
                "render_result_to_html": f"""
<h1>Your Detailed Report</h1>

<h2>Summary</h2>
More text is needed.

<h2>Highlighted Text</h2>
We are confident that the <span style="background-color: rgb(79,70,229,0.5)">highlighted text</span> is AI Generated.<br><br>
{sentence}
""",
                "description": "Invalid input",
                "label": 0,
                "likelihood_score": 0,
                "average_pplx": 0,
                "burstiness": 0,
                "pplx_map": {},
            }
        result = self.get_pplx_map(lines)
        result["likelihood_score"] = self.get_likelihood(result)
        description, label = self.get_result(result)
        result["label"] = label
        result["description"] = description
        result["render_result_to_html"] = render_result_to_html(result)
        return result

    def get_ppl(self, sentence):
        try:
            encodings = self.tokenizer(sentence, return_tensors="pt")
            seq_len = encodings.input_ids.size(1)
            max_length = self.model.config.n_positions
            stride = 512

            nlls = []
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = self.model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len
                    nlls.append(neg_log_likelihood)

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            ppl = int(torch.exp(torch.stack(nlls).sum() / end_loc))
            return ppl
        except Exception:
            return -1
