import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.no_default_selectbox import selectbox
import pandas as pd
import dotenv
import time

from helpers import create_embeddings
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
        df=  create_embeddings(df)
        print(df['embeddings'].head(2))
        
# def process_glossary_1(df):
#     with st.spinner('In progress'):
#         st.info('Create embeddings...')
#         df = create_embeddings(df)
#         st.info('Finding possible duplicates...')
#         df, clusters, dupes = cluster_terms(df, 1)
#         st.write(f'there are {clusters}')
#         fig = plot_clusters(df, 'embeddings')
#         st.plotly_chart(fig)
#         print(clusters)
#         if clusters > 0:
#             duplicate_clusters = df['cluster'].value_counts()
#             duplicate_clusters = duplicate_clusters[duplicate_clusters >1]
#             dupe_list = duplicate_clusters.index.to_list()
#         # duplicates_df = df[df['cluster'].isin(duplicate_clusters)]
#         st.session_state.selected_cluster = selectbox(
#             'select the cluster to fix',
#             dupe_list,
#             )
#         cont = st.button('Continue')
#         if cont:
#             st.write('hello')
#             selected_df = df[df['cluster'] == st.session_state.selected_cluster][['cluster', 'term', 'definition']]
#             st.dataframe(selected_df)
   
    
    
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
                st.dataframe(df_clusters)
                
    # Now this will only rerun when selected_cluster changes
    if 'df_embeddings' not in st.session_state: # or st.session_state.df != df:
        print('1er if')
        with st.spinner('In progress'):
            st.info('Create embeddings...')
            st.session_state.df_embeddings = create_embeddings(df)
            st.session_state.df = df
            st.info('Finding possible duplicates...')
            
            st.session_state.df_embeddings, st.session_state.clusters, st.session_state.dupes = cluster_terms(st.session_state.df_embeddings, st.session_state.distance)
            st.write(f'there are {st.session_state.clusters} clusters for {len(st.session_state.df_embeddings.index)} definitions')
            fig = plot_clusters(st.session_state.df_embeddings, 'embeddings')
            st.plotly_chart(fig)
            if st.session_state.dupes > 0:
                print(st.session_state.dupes)
                df_clusters = df[['cluster', 'term', 'owner','definition']].sort_values(['cluster', 'term','owner', 'definition'])
                st.dataframe(df_clusters)
                # duplicate_clusters = st.session_state.df_embeddings['cluster'].value_counts()
                # duplicate_clusters = duplicate_clusters[duplicate_clusters >1]
                # st.session_state.dupe_list = duplicate_clusters.index.to_list()
                

                  
        
            



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

def add_terms():
    st.title('Add new terms')

def add_sidebar():
    with st.sidebar:
        st.session_state.option = st.radio(
            'Menu',
            options = [
                '1. Load Glossary',
                '2. Manage Glossary',
                '3. Add new item to glossary']            
        )
        st.session_state.distance = st.slider('select distance', 0.0, 1.0, 0.01)
        add_vertical_space(30)
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
    add_sidebar()
        
    if st.session_state.option == '1. Load Glossary':
        load_glossary()
        
    if st.session_state.option == '2. Manage Glossary':
        try: 
            print(st.session_state.username)
            manage()
        except:
            print('no valid user 1')
        
    if st.session_state.option == '3. process glossary':
        try:
            print(st.session_state.username)
        except:
            print('no valid user 2')
            
    if st.session_state.option == '4. Add new item to glossary':
        try:
            st.session_state.username
            add_terms()
        except:
            print('no valid user 4')

if __name__ == '__main__':
    main()