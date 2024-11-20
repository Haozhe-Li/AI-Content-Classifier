from sentencex import segment
from strip_markdown import strip_markdown
import fitz
from docx import Document


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text


def extract_text_from_docx(file_path: str) -> str:
    text = ""
    doc = Document(file_path)
    for para in doc.paragraphs:
        text += para.text
    return text


async def clean_and_segment_text(text: str) -> list:
    text = strip_markdown(text)
    lines = list(segment(language="", text=text))
    lines = [line.replace("\n", " ") for line in lines]
    return lines


def render_result_to_html(results) -> str:
    pplx_map = results["pplx_map"]
    label = results["label"]
    likelihood_score = results["likelihood_score"]
    display_threshould = results["average_pplx"] if results["average_pplx"] < 40 else 40
    burstiness = results["burstiness"]
    summary = """"""
    if likelihood_score >= 0.8:
        summary += f"<b>We are highly confident that the entire text is AI Generated.</b> The likelihood that it is AI Generated is <b>{likelihood_score}</b>, and the likelihood that it is Human Generated is only <b>{round(1 - likelihood_score, 2)}</b>. This text shows a low level of variation in text complexity, which is a common characteristic of AI generated text."
    elif likelihood_score >= 0.5:
        summary += f"<b>We are uncertain about the result.</b> The likelihood that it is AI Generated is <b>{likelihood_score}</b>, and the likelihood that it is Human Generated is <b>{round(1 - likelihood_score, 2)}</b>. This text shows a moderate-high level of variation in text complexity, so we are uncertain about the origin of the text. In most cases, this indicates that it is a mix of AI and human-written text. If we have to make a decision, we would say that the text is <b>AI Generated</b>."
    elif likelihood_score >= 0.3:
        summary += f"<b>We are uncertain about the result.</b> The likelihood that it is AI Generated is <b>{likelihood_score}</b>, and the likelihood that it is Human Generated is <b>{round(1 - likelihood_score, 2)}</b>. This text shows a moderate-high level of variation in text complexity, which is a common characteristic of human-generated text. In most cases, this indicates that it is a mix of AI and human-written text. If we have to make a decision, we would say that the text is <b>Human Generated</b>."
    else:
        summary += f"<b>We are highly confident that the entire text is Human Generated.</b> The likelihood that it is AI Generated is only <b>{likelihood_score}</b>, and the likelihood that it is Human Generated is <b>{round(1 - likelihood_score, 2)}</b>. This text shows a high level of variation in text complexity, which is a common characteristic of human-generated text."
    result = f"""
<h1>Your Detailed Report</h1>

<h2>Summary</h2>
<p>{summary}</p>

<h2>Highlighted Text</h2>
We are confident that the <span style="background-color: rgb(79,70,229,0.5)">highlighted text</span> is AI Generated.<br><br>
"""
    for line, pplx in pplx_map.items():
        if (pplx < display_threshould and label != 0) or likelihood_score >= 0.90:
            result += (
                f"""<span style="background-color: rgb(79,70,229,0.5)">{line}</span> """
            )
        else:
            result += f"{line} "
    result = result.replace(
        """</span> <span style="background-color: rgb(79,70,229,0.5)">""", ""
    )
    return result.strip()
