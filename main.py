import gradio as gr

def greet(name):
    return "Hello " + name + "!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")

with gr.Blocks() as demo:
    gr.Markdown('# <center>INTELLIGENT ANALYSIS OF WEBSITE DRIVEN QA CHATBOT</center>')
    
if __name__ == "__main__":
    demo.queue().launch(debug = True,server_name="0.0.0.0", server_port=8000)