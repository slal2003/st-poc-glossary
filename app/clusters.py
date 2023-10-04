# import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
import umap
import plotly.express as px
import pandas as pd


def cluster_and_find_duplicate_clusters(df, threshold):
    # clustering using HDBSCAN 
    embeddings = np.stack(df['embeddings'].values)
    distance_matrix = cosine_distances(embeddings)
    clusterer = hdbscan.HDBSCAN(metric='precomputed', cluster_selection_epsilon=1-threshold)
    # Fit the model and predict clusters
    df['cluster'] = clusterer.fit_predict(distance_matrix)
    # Calculate the number of duplicate clusters
    # Duplicate clusters are those that contain more than one term
    num_duplicate_clusters = df['cluster'].value_counts()[df['cluster'].value_counts() > 1].count()
    return df, num_duplicate_clusters

def cluster_terms(df, threshold):
    # new method 
    # Calculate distance matrix
    distance_matrix = pairwise_distances(df['embeddings'].tolist(), metric='cosine')
    # Perform Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=threshold)
    df['cluster'] = clustering.fit_predict(distance_matrix)
    # Count the number of clusters and duplicates
    num_clusters = df['cluster'].nunique()
    num_duplicates = (df['cluster'].value_counts() > 1).sum()
    return df, num_clusters, num_duplicates

def plot_clusters(df, embedding_column):
    reducer = umap.UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(list(df[embedding_column]))
    # Create a DataFrame for the 2D embeddings
    embeddings_df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    # Add the cluster assignments
    embeddings_df['cluster'] = df['cluster']
    # Add term names
    embeddings_df['term'] = df['term']
    # Plot the clusters
    fig = px.scatter(embeddings_df, x='x', y='y', color="cluster" , hover_data=['term'])
    # fig = px.scatter(embeddings_df, x='x', y='y', color='cluster', hover_data=['term'])
    return fig




