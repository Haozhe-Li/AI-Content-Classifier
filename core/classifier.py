import torch
import re
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from collections import OrderedDict
import asyncio
import json
from openai import OpenAI
import os
from core.utils import render_sentence

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AIContentClassifier:
    def __init__(self, device="cpu", model_id="gpt2"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        self.device = device
        self.model_id = model_id
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.max_length = self.model.config.n_positions
        self.stride = 512

    def get_results(self, results):
        pplx_maps = results["pplx_maps"]
        threshold = results["average_pplx"]
        likelihood_score = results["likelihood_score"]

        consecutive_count = 0
        for pplx in pplx_maps.values():
            if pplx < 40:
                consecutive_count += 1
                if consecutive_count >= 3:
                    label = 1
                    return (
                        "This text will be classified as a mix of AI-generated and human-written. The highlighted parts in the detailed report below indicate which sentences are most likely AI-generated.",
                        label,
                    )
            else:
                consecutive_count = 0

        if threshold < 60 and likelihood_score > 0.5:
            label = 1
            return "This text has been classified as AI generated.", label
        elif (threshold < 80 and likelihood_score > 0.5) or likelihood_score > 0.7:
            label = 1
            return (
                "This text will be classified as a mix of AI-generated and human-written. The highlighted parts in the detailed report below indicate which sentences are most likely AI-generated.",
                label,
            )
        else:
            label = 0
            return "This text has been classified as human-written.", label

    async def pplx_method(self, lines):
        offset = ""
        pplx_maps = OrderedDict()
        for line in lines:
            if not re.search("[a-zA-Z0-9]+", line):
                continue
            if offset:
                line = offset + line
                offset = ""
            line = line.strip()
            if line[-1] in ["[", "("]:
                offset = line[-1]
                line = line[:-1]
            ppl = await self.get_ppl(line)
            if ppl == -1:
                continue
            pplx_maps[line] = ppl
        results = {
            "pplx_maps": pplx_maps,
            "burstiness": max(pplx_maps.values()),
            "average_pplx": sum(pplx_maps.values()) / len(pplx_maps),
        }
        return results

    async def openai_method(self, sentence):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": 'Please analyze the given text for characteristics indicating AI-generated content, focusing on sentence structure, formatting, and depth.\n\nThe analysis should consider the following features:\n\n- **Sentence Length and Uniformity**: Determine if the text uses consistently similar sentence lengths and lacks variations in structure.\n  \n- **Content Depth**: Examine whether the content lacks depth or specificity, reflecting a superficial level of engagement with the topic.\n\n- **Use of Formatting**: Note any repeated use of bullet points, lists, or overly organized formats that lack complexity.\n\n- **Variable Information**: Check for any indication of varied or redundant information to assess the fluidity of human-like writing.\n\n# Steps\n\n1. Analyze **sentence length** consistency and check for repetitive patterns.\n2. Evaluate **formatting** choices, especially seeing if bullet points are overused.\n3. Assess **content depth**, if the text goes into detail or remains shallow.\n4. Identify **lack of variation or redundancy** in the content, typical of AI-generated text.\n\n# Output Format\n\nRespond in a JSON format including two fields:\n1. `"likelihood_score"`: A numerical likelihood score from 0 to 1 indicating the probability of the content being AI-generated.\n2. `"description"`: A detailed analysis explaining why this score was given, referencing at least 2-3 of the key features (e.g., sentence length similarity, lack of depth, formatting).\n\n```json\n{\n  "likelihood_score": [score between 0 and 1],\n  "description": "[Explanation of the score, noting aspects such as sentence length similarity, superficial treatment of content, and repeated formatting]."\n```\n\n# Examples\n\n**Example 1:**\n\n**Input:**\n"Climate change has likely led to the decline of some of Scotland\'s mountain plants, according to new research.\n\nScientists said many of the species relied on snow cover remaining high on hills until late spring and even summer to ensure a moist environment.\n\nThey also said plants that thrived on lower ground in warmer conditions were spreading to mountain habitats.\n\nSpecies found to be in decline include snow pearlwort, alpine lady-fern and alpine speedwell.\n\nThe research by the Botanical Society of Britain and Ireland (BSBI) has taken 20 years to complete and has been published in the new Plant Atlas.\n\nData used to produce the report included more than three million plant records of 2,555 species collected by hundreds of botanists across Scotland.\n\nClimate change, habitat loss and the spread of non-native species were found to key threats to the health of British and Irish native plants.\n\nBSBI said devastating losses of species in Scotland were among the findings.\n\nAlmost the entire British population of snow pearlwort is found on Ben Lawers, but half of the Perthshire mountain\'s known colonies have disappeared over the last 40 years.\n"\n\n**Output:**\n```json\n{\n  "likelihood_score": 0.2,\n  "description": "The text is concise and factual, with a straightforward journalistic style rather than a structured essay format. This style is less typical of AI-generated content, which often organizes information into clear sections or bullet points. The text includes specific details such as the names of declining plant species ("snow pearlwort, alpine lady-fern, and alpine speedwell") and locations like "Ben Lawers" in Perthshire. The mention of particular species, locations, and statistics (such as “half of the known colonies” disappearing) suggests a level of specificity more typical of human-written content based on direct research findings. Unlike AI-generated text, which may balance pros and cons or delve into an explanatory tone, this text sticks to reporting factual outcomes from the research without additional, generalized context. This text is more likely to be human-written, likely extracted from a research summary or news article."\n}\n```\n\n**Example 2:**\n\n**Input:**\n"The advent of electric vehicles has been touted as a cornerstone in the transition towards sustainable transportation. With global efforts to reduce carbon emissions and mitigate climate change, EVs have gained substantial attention and investment. This paper aims to dissect the multifaceted nature of EVs, weighing their advantages against the inherent limitations.\n\nBenefits of Electric Vehicles:\n\nEnvironmental Impact:\nEVs offer a substantial reduction in greenhouse gas emissions, especially when powered by renewable energy sources. The absence of tailpipe emissions contributes significantly to improving air quality in urban areas.\n\nEnergy Efficiency:\nElectric vehicles are inherently more efficient than their internal combustion engine counterparts. The direct conversion of electrical energy to power provides a higher efficiency rate, reducing overall energy consumption.\n\nPerformance Advantages:\nElectric motors can provide instant torque, offering a smooth and swift acceleration. This feature, combined with low center of gravity designs, often results in superior handling and an enhanced driving experience.\n\nDrawbacks of Electric Vehicles:\n\nLimited Range and Range Anxiety:\nDespite advancements, most EVs still offer a limited range compared to gasoline vehicles. This limitation, coupled with long charging times, contributes to range anxiety among potential consumers.\n\nCharging Infrastructure:\nThe lack of widespread and uniform charging infrastructure remains a significant barrier. The variability in charging station availability, especially in rural or underserved urban areas, poses a challenge for long-distance travel.\n\nConclusion:\nElectric vehicles present a promising avenue towards a more sustainable future in personal transportation. They offer substantial environmental benefits, improved efficiency, and performance advantages. However, challenges like limited range, underdeveloped charging infrastructure, battery issues, high upfront costs, and the potential impact on the power grid remain substantial hurdles. Addressing these challenges is crucial for the widespread adoption and long-term Done! of electric vehicles. As technology advances and infrastructure improves, the balance of these factors may shift further in favor of EVs, solidifying their position as a viable alternative to traditional combustion engines.\n"\n\n**Output:**\n```json\n{\n  "likelihood_score": 0.9,\n  "description": "The text is organized in a highly structured, almost academic style.  This type of organized flow is common in AI-generated content, where information is often broken down methodically. The language is consistently formal and neutral, with no subjective or personal insights, which are more typical of human writing. The text covers multiple points (environmental impact, energy efficiency, performance, range anxiety, and infrastructure) with balanced, concise explanations. AI models like ChatGPT are good at generating this type of exhaustive summary, aiming to include a variety of relevant points to provide a thorough overview. There’s an absence of concrete data, such as specific percentages, statistics, or citations that a human writer might include to support arguments.  Based on these characteristics, it’s highly likely this text was generated by an AI language model like ChatGPT."\n}\n```\n\n# Notes\n\n- Avoid directly concluding that a text is AI-generated without strong reasoning. Highlight specifically why a score was given.\n- The depth and richness of a human-authored text can vary; thus, low variability or point-wise lists alone may not always definitively indicate AI generation but will contribute to a higher likelihood score.',
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": sentence,
                        }
                    ],
                },
            ],
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={"type": "json_object"},
        )
        return float(
            json.loads(response.choices[0].message.content)["likelihood_score"]
        )

    async def classify(self, sentence):
        """
        Takes in a sentence split by full stop
        and print the perplexity of the total sentence

        split the lines based on full stop and find the perplexity of each sentence and print
        average perplexity

        Burstiness is the max perplexity of each sentence
        """
        total_valid_char = sum(len(x) for x in re.findall("[a-zA-Z0-9]+", sentence))
        sentence = re.sub(r"[^a-zA-Z0-9\s\.\?\!\,\:\'\"\(\)\[\]]", "", sentence)
        lines = re.split(r"(?<=[.?!][ \[\(])|(?<=\n)\s*", sentence)
        lines = list(filter(lambda x: x and len(x) > 0, lines))

        if len(lines) < 5 or total_valid_char < 100:
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
                "pplx_maps": {},
            }
        task1 = asyncio.create_task(self.pplx_method(lines))
        task2 = asyncio.create_task(self.openai_method(sentence))
        output, likelihood = await asyncio.gather(task1, task2)

        results = output
        results["likelihood_score"] = likelihood
        description, label = self.get_results(results)
        results["label"] = label
        results["description"] = description
        results["render_sentence"] = render_sentence(results)
        return results

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
