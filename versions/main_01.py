import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.no_default_selectbox import selectbox
import pandas as pd
import dotenv
import time

from helpers import create_openai_embeddings, create_azure_embeddings, generate_definition,generate_openai_definition, generate_openai_evaluation, generate_azure_evaluation
from clusters import cluster_and_find_duplicate_clusters, plot_clusters, cluster_terms

def initialize_session_state():
    """Initialize session state variables."""
    default_values = {
        "shared": True,
        "new_term": "",
        "username": "",
        "password": "",
        "button_pressed": False,
        "distance": 0.01,
        "option": '1. Load Glossary'
    }
    for key, val in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = val

def authentified_user(username, password, user_list):
    return username in user_list and password == 'abc'

def load_glossary():
    """Load the glossary."""
    st.title('Load Glossary')
    col1, col2 = st.columns(2, gap="medium")
    uploaded_file = col1.file_uploader('Drag and Drop Glossary')
    
    if uploaded_file:
        glossary = pd.read_csv(uploaded_file)
        st.session_state.glossary_file = uploaded_file.name
        valid_users = glossary['owner'].unique()
        
        st.session_state.username = col2.text_input('Enter your username')
        st.session_state.password = col2.text_input('Enter your password', type='password')
        
        if st.session_state.username and st.session_state.password:
            if authentified_user(st.session_state.username, st.session_state.password, valid_users):
                col2.info(f'user: {st.session_state.username} is valid')
                st.subheader(f'Glossary Loaded')
                st.dataframe(glossary)
                st.session_state.loaded_glossary = glossary
            else:
                col2.warning(f'unknown user')


def analyse_glossary(df):
    # If there's no state or df has changed, do the expensive computation
    if 'df_embeddings' in st.session_state: # or st.session_state.df != df:
        print('df embedding exists')
        with st.spinner('In progress'):
            st.info('embedding exists...')
            st.session_state.df = df
            st.info('Finding possible duplicates...')
            st.session_state.df_embeddings, st.session_state.clusters, st.session_state.dupes = cluster_terms(st.session_state.df_embeddings, st.session_state.distance)
            st.write(f'there are {st.session_state.clusters} clusters for {len(st.session_state.df_embeddings.index)} definitions')
            fig = plot_clusters(st.session_state.df_embeddings, 'embeddings')
            st.plotly_chart(fig)
            if st.session_state.dupes > 0:
                print(st.session_state.dupes)
                df_clusters = df[['cluster', 'term', 'owner','definition']].sort_values(['cluster', 'term','owner', 'definition'])
                df_clusters['dupes_in_cluster'] = df_clusters.groupby('cluster')['cluster'].transform('count')
                df_clusters = df_clusters[df_clusters['dupes_in_cluster']>1]
                
                st.markdown("""
                            ## Terms with duplicates to solve
                            """)
                st.dataframe(df_clusters)
    # Now this will only rerun when selected_cluster changes
    if 'df_embeddings' not in st.session_state: # or st.session_state.df != df:
        print('1er if')
        with st.spinner('In progress'):
            st.info('Create embeddings...')
            st.session_state.df_embeddings = create_openai_embeddings(df)
            st.session_state.df = df
            st.info('Finding possible duplicates...')
            
            st.session_state.df_embeddings, st.session_state.clusters, st.session_state.dupes = cluster_terms(st.session_state.df_embeddings, st.session_state.distance)
            st.write(f'there are {st.session_state.clusters} clusters for {len(st.session_state.df_embeddings.index)} definitions')
            fig = plot_clusters(st.session_state.df_embeddings, 'embeddings')
            st.plotly_chart(fig)
            if st.session_state.dupes > 0:
                print(st.session_state.dupes)
                df_clusters = df[['cluster', 'term', 'owner','definition']].sort_values(['cluster', 'term','owner', 'definition'])
                df_clusters['dupes_in_cluster'] = df_clusters.groupby('cluster')['cluster'].transform('count')
                df_clusters = df_clusters[df_clusters['dupes_in_cluster']>1]
                
                st.markdown("""
                            ## Terms with duplicates to solve
                            """)
                st.dataframe(df_clusters)



def manage():
    st.title('Manage Glossary')
    st.info(f'''
        username = {st.session_state.username} \n
        Glossary Loaded
        ''')
    st.dataframe(st.session_state.loaded_glossary)
    st.session_state.button_pressed = st.button('start analysis')
    if st.session_state.button_pressed:
        st.write('button pressed')
        analyse_glossary(st.session_state.loaded_glossary)
        
def add_term():
    """Add a new term."""
    st.title('Add new terms')
    data_entry, result = st.columns([1,2])
    
    new_term = data_entry.text_input('Enter new term', value=st.session_state.new_term)
    if new_term != '':
        # deal with capitalisation
        lower_new_term = new_term.lower()
        existing_lower_terms = [t.lower() for t in st.session_state.df['term'].to_list()]
        
        # display a warning
        if lower_new_term in existing_lower_terms:
            st.warning(f" ⚠️ ***'{new_term}'*** already exists ")

            # Retrieve the term using the lowercase match
            found_term = st.session_state.df[st.session_state.df['term'].str.lower() == lower_new_term]
            
            print(f'found {found_term} ')
            st.markdown(f"""
                        #### {found_term['term'].values[0]}
                        - {found_term['definition'].values[0]}
                        """)
            
            ## Offer options
            option = st.selectbox('select your option:', ('I will use this term', 'I will enter a new term'))
            
            if option == 'I will enter a new term':
                st.session_state.new_term = ""
                st.info('enter a new term')
                st.session_state.new_term = ''
                
            else: 
                pass
        else: 
            st.write("item doesn't exists")
            with st.form('new_item'):
                domain = st.text_input('enter domain')
                keywords = st.text_input('enter some keyword for the definition')
                submitted = st.form_submit_button('Submit')
                if submitted:
                    print(f'a new term was submitted: it is {new_term}')
                    with st.container():
                        definition = generate_openai_definition(new_term, domain, keywords)
                        st.markdown('## Definition')
                        st.info(definition.content)
                        st.divider()
                        st.markdown('## Evaluation')
                        evaluation = generate_openai_evaluation(new_term, domain, keywords, definition)
                        st.info(evaluation.content)

def add_sidebar():
    """Add sidebar elements."""
    with st.sidebar:
        st.subheader('Clustering Distance')
        st.session_state.distance = st.slider('select distance', 0.00, 0.20, st.session_state.distance)
        add_vertical_space(5)
        st.subheader('STEPS')
        st.session_state.option = st.radio('Select the step', options=['1. Load Glossary', '2. Manage Glossary', '3. Add new item to glossary'])
        add_vertical_space(5)
        "st.session_state object: ", st.session_state

def main():
    """Main function."""
    initialize_session_state()
    
    st.set_page_config(
        page_title="Glossary POC",
        page_icon=":book:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    add_sidebar()
    
    if st.session_state.option == '1. Load Glossary':
        load_glossary()
    elif st.session_state.option == '2. Manage Glossary':
        manage()
    elif st.session_state.option == '3. Add new item to glossary':
        add_term()

if __name__ == '__main__':
    main()
