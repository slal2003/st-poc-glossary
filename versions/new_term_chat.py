from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

def generate_definition(term, domain, keywords):
    query= f'Suggest a definition for this term: {term} and these keywords {keywords} making sure the definition is concise, clear and elaborate enough to discriminate ahainst other terms from this domain.'
    text = "You are a seasoned data steward with vast knowledge in many business areas and able to write in a very clear, precise and concise english"
    llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo", max_tokens=200)
    definition = llm(text + query)
    return definition