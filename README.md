# st-poc-glossary-poc

## Glossary cleaner

This part handles existing glossaries making sure there are no duplicates with existing terms (lexically or semantically).
This is attained by grouping by similarity and offering the glossary manager tools to resolve posssible conflicts.

### 1. Data Uploading and processing

- Create a UI that allow users to load the existing glossary by uploading a csv file, or by connecting to the glossary db or via API call
- Load the data into a df
- Embed the list of terms and append the embedding columns to the df

### 2. Flag possibe conflicts and collusions

- Build a mechanism to flag possible duplicates.
- Use GPT to suggest : merging terms, discriminate terms

### 3. Conflict resolution

- Create UX to resolve ownership conflict resolution using communication tools e.g.SLack /Slack API, suggested conversations etc

## Adding a new term
