from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import openai

# location of the pdf file/files. 
pdf_path = input("Please enter the path of the PDF file (must end with .pdf): \n")
reader = PdfReader(pdf_path)

# openAI API
os.environ["OPENAI_API_KEY"] = " "

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)


embeddings = OpenAIEmbeddings()

# take the text chunk and find the corresponding embeddings
docsearch = FAISS.from_texts(texts, embeddings)

# Here you can pass on different models that you want
# default = 'text-davinci-003' model
# This will create a chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")


def ask_question_and_get_answer():
    while True:
        # Ask for user's input
        user_input = input("Enter your question (or type 'exit' to quit): ")

        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            break

        # Pass the input as a query
        query = user_input
        docs = docsearch.similarity_search(query)

        # Get the answer and print it
        answer = chain.run(input_documents=docs, question=query, max_tokens=200)
        print("Answer:", answer, "\n")



ask_question_and_get_answer()