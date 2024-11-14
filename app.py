from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.schema import BaseRetriever
from typing import Any, List
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')



embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name="medical-chatbot"

#Loading the index
docsearch = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings  # Use the same embeddings function as before
)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


# qa=RetrievalQA.from_chain_type(
#     llm=llm, 
#     chain_type="stuff", 
#     retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True, 
#     chain_type_kwargs=chain_type_kwargs)





# First create the custom retriever class
class CustomRetriever(BaseRetriever):
    vectorstore: Any
    
    def __init__(self, vectorstore):
        super().__init__()
        self.vectorstore = vectorstore
    
    def get_relevant_documents(self, query: str) -> List:
        return self.vectorstore.similarity_search(query, k=2)
    
    async def aget_relevant_documents(self, query: str) -> List:
        return await self.vectorstore.asimilarity_search(query, k=2)

# Create retriever from your docsearch
custom_retriever = CustomRetriever(docsearch)

# Create the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=custom_retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


