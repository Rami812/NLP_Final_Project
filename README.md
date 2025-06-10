# NLP_Final_Project
In this project we explore how GDP is impacted by a government’s central bank speech by creating features via NLP techniques like Word2Vec embedding, Part of Speech (POS) tagging and predict final trends by utilizing logistic regression to predict GDP trends. 

We have the following datasets:
Global Economic Indicators Dataset (2010-2023): This contains the economic metrics such as GDP growth rate, inflation rates, unemployment rates and stock index values across different countries. 
Speeches Dataset: Contains the transcripts of speeches from central banks across countries, some being translated along with details of who and on which date the respective speeches were given.
ISO3 Country Codes: Contains a mapping of 3-digit alpha country codes to country names for data translation
Alpha Vantage: We have additional news data across different countries relating to GDP, economic development, unemployment etc. I had to use an API key for this. 
FRED: We acquired GDP data of different countries from Federal Reseve Economic Data 

Phases of the Project:
Data Processing: Merged speeches and economic indicators dataset based on common country and date column, standardized country codes based on ISO-3 codes and then created a classification column for whether or not GDP increased. 
Feature Engineering:
Tokenized speech texts and trained Word2Vec models (CBOW and Skip-Gram) to generate word embeddings.
Aggregated word embeddings to create document-level embeddings for each speech.
Results:
Logistic regression resulted in a score of 0.80 in ROC_AUC with key features being GDP increase and Word2Vec embeddings and GDP growth rate.
Dependencies for the project:
Python 3.11
Libraries: 
Pandas
Numpy
Gensim
Spacy
Scikit-learn
Matplotlib and seaborna


Dataset for speeches data: https://cbspeeches.com/
Dataset for ISO: This file has been added. The file has been added to the repo. 

An updated Google Colab can be found here-https://colab.research.google.com/drive/16POFVYD2v6IQ6rdDpq3zKusjatsJtyO8?usp=sharing 
