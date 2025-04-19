import gradio as gr
from chatbot_ui import chatbot
from document_processor import process_documents
iface = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="Insurance AI Chatbot")

if __name__ == "__main__":
    process_documents()
    iface.launch(share=False)
