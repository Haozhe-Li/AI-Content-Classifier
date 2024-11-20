from core.classifier import AIContentClassifier
from core.settings import llama_text, gpt_text, human_text, html_placeholder
from core.utils import extract_text_from_pdf, extract_text_from_docx
import gradio as gr

classifier = AIContentClassifier()


async def main(input_text):
    result = await classifier.classify(input_text)
    description = result["description"]
    render_result_to_html = result["render_result_to_html"]
    if description == "Invalid input":
        gr.Warning("Invalid input, more text is needed.", title="Error!", duration=5)
        return description, render_result_to_html
    gr.Info("Your result is ready below.", duration=5, title="Done!")
    gr.Warning(
        "Please noted that result may not be accurate, further investigation is needed.",
        duration=10,
        title="Disclaimer",
    )
    return description, render_result_to_html


async def load_gpt_text():
    gr.Info("GPT-4o Text is loaded", duration=2, title="Done!")
    return gpt_text


async def load_llama_text():
    gr.Info("LLaMa 3 70b Text is loaded", duration=2, title="Done!")
    return llama_text


async def load_human_text():
    gr.Info("Human Text is loaded", duration=2, title="Done!")
    return human_text


def clear_all():
    gr.Info("All fields are cleared", title="Done!", duration=2)
    return "", "Your result will appear here", html_placeholder

def parse_file(file):
    try:
        file_path = file.name
        if file_path.endswith(".pdf"):
            gr.Info("PDF file is loaded", duration=2, title="Done!")
            return extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            gr.Info("DOCX file is loaded", duration=2, title="Done!")
            return extract_text_from_docx(file_path)
        else:
            gr.Warning("Invalid file format, only PDF and DOCX are supported.", title="Error!", duration=5)
            return ""
    except Exception as e:
        return ""


with gr.Blocks(title="AI Content Classifier", theme=gr.themes.Soft()) as demo:
    input_text = gr.Textbox(
        label="Input Text",
        lines=10,
        placeholder="Please enter your text here, minimum 100 characters",
    )
    file_input = gr.File(label="Or upload a file", type="filepath")
    with gr.Row():
        gpt_text_btn = gr.Button("ChatGPT")
        gpt_text_btn.click(load_gpt_text, inputs=[], outputs=[input_text])
        llama_text_btn = gr.Button("LLaMa")
        llama_text_btn.click(load_llama_text, inputs=[], outputs=[input_text])
        human_text_btn = gr.Button("Human")
        human_text_btn.click(load_human_text, inputs=[], outputs=[input_text])
    detect_button = gr.Button("Detect", variant="primary")
    clear_button = gr.Button("Clear", variant="stop")
    output_text = gr.Textbox(label="Classifier", value="Your result will appear here")
    output_html = gr.HTML(label="Detailed Output", value=html_placeholder)
    detect_button.click(main, inputs=input_text, outputs=[output_text, output_html])
    clear_button.click(clear_all, outputs=[input_text, output_text, output_html])
    file_input.change(parse_file, inputs=file_input, outputs=[input_text])

demo.launch()
