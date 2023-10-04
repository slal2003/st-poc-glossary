import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.no_default_selectbox import selectbox
import pandas as pd
import dotenv
import time

from helpers import create_openai_embeddings, create_azure_embeddings, generate_definition,generate_openai_definition, generate_openai_evaluation, generate_azure_evaluation
from clusters import cluster_and_find_duplicate_clusters, plot_clusters, cluster_terms


def authentified_user(username, password, user_list):
    invalid_credentials = True
    if username not in user_list or password != 'abc':
        return False
    else:
        return True
    
def load_glossary():
    st.title('Load Glossary')
    glossary = ''        
    
    try:
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            glossary = pd.read_csv(st.file_uploader('Drag and Drop Glossary')) 
            st.session_state.glossary_file = glossary.Name()
    except:
        print('file not dragged yet')
    if len(glossary) >0:
        with col1:
            st.info('File loaded. Thank you')
            valid_users = glossary['owner'].unique()
            with col2:
                st.session_state.username = st.text_input('Enter your username')
                st.session_state.password = st.text_input('Enter your password', type='password')
                valid_user = authentified_user(st.session_state.username, st.session_state.password, valid_users)
                if not valid_user:
                    st.warning(f'unknown user')
                else:
                    with col2:
                        st.info(f'user: {st.session_state.username} is valid')
        add_vertical_space(5)
        if valid_user:
            st.subheader(f'Glossary Loaded')
            st.dataframe(glossary)
            st.session_state.loaded_glossary = glossary
            # st.session_state.loaded_glossary     
    else:
        with col1:
            st.warning('upload a file')
            
def embeddings_exist(df):
    return False

def get_embeddings(df):
    if embeddings_exist(df):
        print('embedding exists')
    else:
        df=  create_openai_embeddings(df)
        print(df['embeddings'].head(2))

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
    st.title('Add new terms')
    data_entry, result = st.columns([1,2])
        
    with data_entry:
        if 'new_term' not in st.session_state:
            st.session_state.new_term = ""
        new_term = st.text_input('Enter new term', value=st.session_state.new_term) 
        
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
    with st.sidebar:
        st.subheader('Clustering Distance')
        st.session_state.distance = st.slider('select distance', 0.00, 0.20, 0.01)
        add_vertical_space(5)
        st.subheader('STEPS')
        st.session_state.option = st.radio(
            'Select the step',
            options = [
                '1. Load Glossary',
                '2. Manage Glossary',
                '3. Add new item to glossary']            
        )
        add_vertical_space(5)
        "st.session_state object: ", st.session_state
    return st.session_state.option, st.session_state.distance


def main():    
    if "shared" not in st.session_state:
            st.session_state["shared"] = True

    st.set_page_config(
        page_title="Glossary POC",
        page_icon=":book:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    placeholder = st.empty()
    add_sidebar()
        
    if st.session_state.option == '1. Load Glossary':
        load_glossary()
        
    if st.session_state.option == '2. Manage Glossary':
        try: 
            print(st.session_state.username)
            manage()
        except:
            print('no valid user 1')
        
    if st.session_state.option == '3. Add new item to glossary':
        try:
            print(st.session_state.username)
            add_term()
        except:
            print('no valid user 2')
            

if __name__ == '__main__':
    main()