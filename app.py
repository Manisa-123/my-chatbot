import gradio as gr
from chatbot_ui import chatbot
iface = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="Insurance AI Chatbot")
if __name__ == "__main__":
    iface.launch(share=False)
