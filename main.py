# from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from llama_index import LLMPredictor
# import torch
# from langchain.llms.base import LLM
# from transformers import pipeline
import gradio as gr
import pandas as pd
import pytesseract
from PIL import Image
import fitz
# from llama_index import Document
import os
# from langchain.document_loaders import CSVLoader
# import moviepy.editor as mp
import json
import pickle

def read_text_file(file):
    with open(file.name, "r") as f:
        text = f.read()
        # build_the_bot(text)
    return text

def read_image_file(file):
    print(file.name)
    text = pytesseract.image_to_string(Image.open(file.name))
    # build_the_bot(text)
    return text

def read_pdf(file):
  doc = fitz.open(file.name)
  text = ""
  for page in doc:
    text+=page.get_text()
#   build_the_bot(text)
  return text

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
def find_file_type(input,file):
    if input:
      print(1)
    #   return build_the_bot(input)
    
    if file:
        filename, file_extension = os.path.splitext(file.name)
        if file_extension.lower()=='.txt':
          text=read_text_file(file)
        elif file_extension.lower() in ['.png','.jpg','.jpeg']:
          text=read_image_file(file)
        elif file_extension.lower()=='.pdf':
          text=read_pdf(file)
        # elif file_extension.lower() in ['.mp3','.wav','.m4a']:
        #   text = transcribe2(file.name)
        # elif file_extension=='.mp4':
        #   text = video(file.name)
        else:
          text='Try with Text, Image and PDF Document only.'  
        return text
    
with gr.Blocks() as demo:
    gr.Markdown('# <center>INTELLIGENT ANALYSIS OF WEBSITE DRIVEN QA CHATBOT</center>')
    with gr.Column(variant="panel"):
      gr.Markdown("## Feed the Bot ")
      with gr.Row(variant="compact"):    
          input=gr.Textbox(
              label='Enter the Content',
              placeholder='Prompt engineering is a concept in artificial intelligence (AI), particularly natural language processing (NLP). In prompt engineering, the description of the task that the AI is supposed to accomplish is embedded in the input, e.g. as a question, instead of it being implicitly given. Prompt engineering typically works by converting one or more tasks to a prompt-based dataset and training a language model with what has been called "prompt-based learning" or just "prompt learning".'
          )
          with gr.Column(variant="panel"): 
            gr.Markdown("## <center>Or Upload your Document <center>")
            with gr.Row(variant="compact"):
              files = gr.File()
      text_output = gr.Textbox(
          label="Verify the content",
          placeholder="Start Building the Bot to view the content")
      text_button = gr.Button("Build the Bot!!!")
      text_button.click(find_file_type, [input,files], text_output)

if __name__ == "__main__":
    demo.queue().launch(
       debug = True,
    #    server_name="0.0.0.0", server_port=8000
    )