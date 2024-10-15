from flask import Flask, render_template, request, jsonify
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain
import os

app = Flask(__name__)

# Hugging Face API Token and Model Repo
HUGGINGFACEHUB_API_TOKEN = "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"
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=256,
    temperature=0.7,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Prompt Template
template = """Question : {question}
Answer: Let's think step by step..."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question")
    
    # Call the LLMChain with the question
    try:
        answer = llm_chain.run(question)
    except Exception as e:
        answer = "Sorry, I couldn't generate an answer."

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
