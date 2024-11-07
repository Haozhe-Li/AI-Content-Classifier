import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from collections import OrderedDict
import os
from core.utils import render_sentence, clean_and_segment_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AIContentClassifier:
    def __init__(self, device="cpu", model_id="gpt2"):
        self.device = device
        self.model_id = model_id
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.max_length = self.model.config.n_positions
        self.stride = 512

    def get_result(self, result):
        likelihood_score = result["likelihood_score"]

        if likelihood_score > 0.8:
            label = 1
            return "We are highly confident that this text is AI-generated.", label
        elif likelihood_score >= 0.5 and likelihood_score <= 0.8:
            label = 1
            return (
                "This text may be AI-generated. In most cases, AI-generated content could be mixed with human-written content.",
                label,
            )
        elif likelihood_score > 0.2 and likelihood_score < 0.5:
            label = 0
            return "This text may be human-written, but we are not confident with the result.", label
        else:
            label = 0
            return "We are confident that this text is human-written.", label

    async def get_pplx_map(self, lines):
        pplx_map = OrderedDict()
        for line in lines:
            ppl = await self.get_ppl(line)
            if ppl == -1:
                continue
            pplx_map[line] = ppl
        result = {
            "pplx_map": pplx_map,
            "burstiness": max(pplx_map.values()),
            "average_pplx": sum(pplx_map.values()) / len(pplx_map),
        }
        return result

    async def get_likelihood(self, result: dict):
        pplx_map = result["pplx_map"]
        average_pplx = result["average_pplx"]
        burstiness = result["burstiness"]
        likelihood_score = 0
        consecutive_count = 0

        for pplx in pplx_map.values():
            if pplx < 40:
                likelihood_score += 1
                consecutive_count += 1
            else:
                consecutive_count = 0

            if consecutive_count >= 3:
                likelihood_score += 1

        if average_pplx < 60:
            likelihood_score += 1
        elif average_pplx < 80:
            likelihood_score += 0.5

        likelihood_score = likelihood_score / len(pplx_map)

        factor = 2
        if likelihood_score >= 0.5:
            return round(min(1, likelihood_score ** (1/factor)), 2)
        else:
            return round(max(0, likelihood_score ** factor), 2)

    async def classify(self, sentence):
        lines = await clean_and_segment_text(sentence)
        if len(lines) < 5:
            return {
                "render_sentence": f"""
<h1>Your Detailed Report</h1>

<h2>Summary</h2>
More text is needed.

<h2>Highlighted Text</h2>
We are confident that the <span style="background-color: rgb(79,70,229,0.5)">highlighted text</span> is AI Generated.<br><br>
{sentence}
""",
                "description": "Please enter a longer text with at least 100 characters.",
                "label": 0,
                "likelihood_score": 0,
                "average_pplx": 0,
                "burstiness": 0,
                "pplx_map": {},
            }
        result = await self.get_pplx_map(lines)
        result["likelihood_score"] = await self.get_likelihood(result)
        description, label = self.get_result(result)
        result["label"] = label
        result["description"] = description
        result["render_sentence"] = render_sentence(result)
        return result

    async def get_ppl(self, sentence):
        try:
            encodings = self.tokenizer(sentence, return_tensors="pt")
            seq_len = encodings.input_ids.size(1)

            nlls = []
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, self.stride):
                end_loc = min(begin_loc + self.max_length, seq_len)
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
