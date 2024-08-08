from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from werkzeug.utils import secure_filename
import fitz, os, torch, logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG,  # Set the logging level to DEBUG
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Add a file handler to save logs to a file
file_handler=logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
app.logger.addHandler(file_handler)

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
app.config['UPLOAD_FOLDER']='uploads'  # Folder to store uploaded files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create folder if not exists
ALLOWED_EXTENSIONS={'pdf'}

def load_llm():
    checkpoint="MBZUAI/LaMini-T5-738M"
    tokenizer=AutoTokenizer.from_pretrained(checkpoint)
    base_model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint,torch_dtype=torch.float32)
    pipe=pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    llm=HuggingFacePipeline(pipeline=pipe)
    return llm

#creating prompt template using langchain
def load_prompt():
    prompt=""" You need to answer the question in the sentence as same as in the  pdf content.
    Given below is the context and question of the user.
    context={context}
    question={question}
    If the answer is not in the pdf content, output "I do not know what the hell you are asking about"
    """
    prompt=ChatPromptTemplate.from_template(prompt)
    return prompt

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    doc=fitz.open(stream=pdf_file.read(), filetype="pdf")
    text=""
    for page in doc:
        text+=page.get_text()
    return text

def vectorize_document(filepath):
    DB_FAISS_PATH='vectorstore/db_faiss'
    loader=PyPDFLoader(filepath)
    docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits=text_splitter.split_documents(docs)
    vectorstore=FAISS.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    vectorstore.save_local(DB_FAISS_PATH)

def load_knowledgeBase():
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    DB_FAISS_PATH='vectorstore/db_faiss'
    vectorstore=FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize the question-answering pipeline
qa_pipeline=pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file=request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename=secure_filename(file.filename)
        filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        file.seek(0)
        vectorize_document(filepath)
        content=extract_text_from_pdf(file)
        return jsonify({"content": content})

    return jsonify({"error": "Invalid file format. Only PDF files are allowed."}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data=request.json
    question=data.get('question')
    context=data.get('context')

    if not question or not context:
        return jsonify({"error": "Invalid input"}), 400

    #app.logger.debug("--Line 140")
    #app.logger.debug(question)
    #app.logger.debug(context)

    vectorstore=load_knowledgeBase()
    llm=load_llm()
    prompt=load_prompt()

    app.logger.debug("")
    #getting only the chunks that are similar to the query for llm to produce the output
    similar_embeddings=FAISS.from_documents(documents=vectorstore.similarity_search(question), 
                                            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
                
    #creating the chain for integrating llm,prompt,stroutputparser
    retriever=similar_embeddings.as_retriever(search_kwargs={"k": 3})
    rag_chain=(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response=rag_chain.invoke(question) 
    result={'answer': response}
    #result = qa_pipeline(question=question, context=context)

    app.logger.debug(result) 
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
