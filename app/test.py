import pandas as pd
from helpers import create_embeddings

# create a dictionary of lists
data = {
    'definition': ['The quick brown fox', 'jumps over the lazy dog'], 
    'tags': ['', '']
}

df = pd.DataFrame(data)

print(df)

df = create_embeddings(df)

print(df)
