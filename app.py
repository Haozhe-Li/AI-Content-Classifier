from core.classifier import AIContentClassifier
from core.settings import llama_text, gpt_text, human_text, mixed_text, html_placeholder
import gradio as gr

classifier = AIContentClassifier()


async def main(input_text):
    result = await classifier.classify(input_text)
    description = result["description"]
    render_result_to_html = result["render_result_to_html"]
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


with gr.Blocks(title="AI Content Classifier", theme=gr.themes.Soft()) as demo:
    input_text = gr.Textbox(
        label="Input Text",
        lines=10,
        placeholder="Please enter your text here, minimum 100 characters",
    )
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

demo.launch()
