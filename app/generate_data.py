import pandas as pd
import openai
import datetime
from dotenv import find_dotenv, load_dotenv

# Your OpenAI key
openai.api_key = "YOUR_OPENAI_KEY"

load_dotenv(find_dotenv())

# Your existing data
data = {
    'term': ['Translation Log', 'Carbon offset'],
    'definition': ["The category or type of an item or product. This might refer to broad categories like 'electronics' or 'clothing'.",
                   "The type of units used for a particular class of items. This could refer to physical units (like 'pieces', 'pairs', 'kilograms'), or it could refer to abstract units of measurement (like 'licenses' for software products)."],
    'domain': ['Core Range Information - Glossary', 'Climate Positive - Glossary'],
    'owner': ['slim@acme.com', 'slim@acme.com'],
    'tags': ['Business Term', 'Business Term'],
    'other_domains': ['logistics| digital', 'logistics| digital'],
    'inserted_ts': [datetime.datetime(2023, 6, 15, 9, 38, 3), datetime.datetime(2023, 6, 15, 9, 38, 3)],
    'updated_ts': [datetime.datetime(2023, 6, 19, 9, 38, 3), datetime.datetime(2023, 6, 19, 9, 38, 3)],
    'len_def': [114, 221],
    'len_tags': [13, 13]
}

# Create a DataFrame
df = pd.DataFrame(data)

# List to store new data
new_data = []

# Generate 300 new rows
for _ in range(300):
    # Generate a new term and its definition
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0613",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate a new business term and its definition."},
        ]
    )
    
    # Split the response into term and definition
    term_def = response['choices'][0]['message']['content'].strip().split(':')
    term = term_def[0].strip()
    definition = term_def[1].strip()
    
    # Generate a related domain
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-0613",
      messages=[
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": f"Generate a related domain for the term '{term}'."},
        ]
    )
    
    domain = response['choices'][0]['message']['content'].strip()
    
    row = {
        'term': term,
        'definition': definition,
        'domain': domain,
        'owner': 'slim@acme.com',  # assuming the owner remains the same
        'tags': 'Business Term',  # assuming the tags remain the same
        'other_domains': 'logistics| digital',  # assuming the other domains remain the same
        'inserted_ts': datetime.datetime(2023, 6, 15, 9, 38, 3),  # assuming the timestamps remain the same
        'updated_ts': datetime.datetime(2023, 6, 19, 9, 38, 3),  # assuming the timestamps remain the same
        'len_def': len(definition),
        'len_tags': len('Business Term')
    }
    new_data.append(row)

# Create a new DataFrame
new_df = pd.DataFrame(new_data)

# Append new data to the existing DataFrame
df = df.append(new_df)

print(df)
