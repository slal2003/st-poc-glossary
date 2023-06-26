import streamlit as st
import pickle


import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

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
      
def extract_tags(term, definition):
    load_dotenv(find_dotenv())
    query = 'list up to 7 tags for this definition. The tags must be short and together help recreate the definition and help discriminate this definition from others that can be semantically close. the output should be a comma separated list '
    st.write(f'query: {query}')
    text = f"term: {term} definition:{definition}"
    st.write(f'text: {text}')
    llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo", max_tokens=200)
    st.write(llm)
    tags = llm(query + text)
    # chain=load_qa_chain(llm=llm, chain_type="stuff")
    # tags = chain.run(input_documents=text, question=query)
    return tags

def generate_definition( term, domain, keywords):
    load_dotenv(find_dotenv())
    messages = [
        SystemMessage(content=f'You are a seasoned Data Stewart, expert in data analytics and business glossaries'),
        HumanMessage(content=f'Suggest a definition for this term: {term} of this {domain} and these keywords {keywords} making sure the definition is concise, clear and elaborate enough to discriminate against other terms from this domain. return the answer in markdown format. It needs to be ready to populate the glossary. makle sure only the definition is output and no t the rules or anything else')
    ]
    llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo", max_tokens=200)
    definition = llm(messages)
    return definition

def generate_evaluation(term, domain, keywords, definition):
        load_dotenv(find_dotenv())
        messages = [
            SystemMessage(content=f'You are a seasoned Data Stewart, expert in data analytics and business glossaries'),
            HumanMessage(content=f'Evaluate the adequation of this business/data analytics {term} from this {domain} its keywords {keywords} and its GPT generated {definition} and suggest better term to avoid confusion and ambiguity. It is important to limit yourself to the evaluation only')
        ]
        llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo", max_tokens=200)
        evaluation = llm(messages)
        return evaluation
