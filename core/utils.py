from sentencex import segment
from strip_markdown import strip_markdown

async def clean_and_segment_text(text: str) -> list:
    text = strip_markdown(text)
    lines = list(segment(language="", text=text))
    lines = [line.replace("\n", " ") for line in lines]
    return lines

def render_sentence(results) -> str:
    pplx_map = results["pplx_map"]
    label = results["label"]
    likelihood_score = results["likelihood_score"]
    average_pplx = results["average_pplx"] if results["average_pplx"] < 40 else 40
    burstiness = results["burstiness"]
    description = results["description"]
    result = f"""
<h1>Your Detailed Report</h1>

<h2>Summary</h2>
<p>{description} The likelihood score that the text is AI-generated is <strong>{likelihood_score}</strong>.</p>

<h2>Highlighted Text</h2>
We are confident that the <span style="background-color: rgb(79,70,229,0.5)">highlighted text</span> is AI Generated.<br><br>
"""
    for line, pplx in pplx_map.items():
        if (pplx < average_pplx and label != 0) or likelihood_score > 0.95 :
            result += (
                f"""<span style="background-color: rgb(79,70,229,0.5)">{line}</span> """
            )
        else:
            result += f"{line} "
    return result.strip()
