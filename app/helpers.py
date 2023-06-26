import streamlit as st
import pickle


import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS


from dotenv import find_dotenv, load_dotenv
import os



def create_embeddings(df):
    load_dotenv(find_dotenv())
    def embed(text):
        return OpenAIEmbeddings().embed_query(text)
    df['text'] = df['definition'] + ' ' + df['tags']
    df['embeddings'] = df['text'].apply(embed)
    return df

 

# def create_embeddings(term, definition):
#     data_path = "data/embeddings"
#     if os.path.exists(f"{data_path}{term}.pkl"):
#         with open(f"{data_path}/glossary_embeddings.txt", 'rb') as f:
#             VectorStore=pickle.load(f)
#         st.write('Embeddings loaded from disk')
#     else:
#         embeddings = OpenAIEmbeddings()
#         print(OPENAI_API_KEY)
#         print(embeddings)
#         VectorStore = FAISS.from_texts(definition, embedding=embeddings)
#         with open(f"{data_path}/glossary_embeddings.txt", 'wb') as f:
#             pickle.dump(VectorStore, f)
#         st.write('Embeddings computation Completed and stored to Disk')
    
    
def extract_tags(term, definition):
    load_dotenv(find_dotenv())
    query = 'list up to 7 tags for this definition. The tags must be short and together help recreate the definition and help discriminate this definition from others that can be semantically close. the output should be a comma separated list '
    st.write(f'query: {query}')
    text = f"term: {term} definition:{definition}"
    st.write(f'text: {text}')
    llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo", max_tokens=200)
    st.write(llm)
    tags = llm(query + text)
    # chain=load_qa_chain(llm=llm, chain_type="stuff")
    # tags = chain.run(input_documents=text, question=query)
    return tags

def generate_definition( term, keywords):
    query= f'Suggest a definition for this term: {term} and these keywords {keywords} making sure the definition is concise, clear and elaborate enough to discriminate ahainst other terms from this domain.'
    text = "You are a seasoned data steward with vast knowledge in many business areas and able to write in a very clear, precise and concise english"
    llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo", max_tokens=200)
    definition = llm(text + query)
    return definition

def verify_term_vs_keywords(term, keywords):
    text = f'You are a seasoned data steward with vast knowledge in many business areas and specially in various domains and able to evaluate how relevant a business term is'
    query = f"""
    Evaluate in a critical way the term {term} in relation for the topics: {keywords} supplied it is part of 
    and suggest a better term name that discriminate better against other similar business terms and that is more descrptive and follw better best practices.
    Provide answe in pint format using relevant emojis to make it easier to understand
    Provide criticisms only if necessary. If the term is consistent with the name entered, mention that is looks good enough based on context provided """
    llm = OpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=200)
    verification = llm(text + query)
    return verification
    