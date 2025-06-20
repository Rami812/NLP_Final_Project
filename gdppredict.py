import streamlit

import ast

import sklearn
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.utils import resample

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import StandardScaler

import requests
from io import BytesIO
st.markdown("### Utilizing Central Bank Speeches and News Data to Predict GDP Trends Across Countries")
st.markdown("---")

# Introduction section
st.markdown("""
**Introduction and Background:**

In this project, we try to utilize central bank speeches and news data across different countries to predict whether or not the Gross Domestic Indicator (GDP) increases post news/speech. Political speeches have often acted as summaries, especially ones from central banks who have mandates relating to ensuring price stability and in the case of the US, also ensuring that there are high levels of employment. There has always been debate on how much to disclose and communicate about strategic initiatives like monetary policies in combating inflationary/deflationary rates for the common public. Therefore, with the help of modern NLP techniques, this project is meant to answer the following questions:
- Are there notable differences in speeches by central banks between countries which can be studied to ensure a better methodology for delivering such speeches across countries? Answered in the Histogram Analysis Tab
- With every country being connected in a global economy, are there common trends in their GDP values across time? Answered in the GDP Analysis Tab
- Would machine learning models like Logistic Regression or pre-trained models like FinBERT fare well in being able to speculate if GDP is to increase/decrease? Answered in the Confusion Matrix Tab
With central bank speeches oftentimes, being summaries of more complicated econometric and financial analysis meant to provide insights to the common public in a more bite sized manner, there is a growing importance of ensuring such communication is executed effectively and Natural Language Processing techniques are assisting in analyzing these further.We picked GDP as the variable for increment/decrement as this is often times calculated as the average of common goods across nations and these prices are indicative of the market factors of demand and supply being present in the economy. 

""")

st.markdown("---")
def load_excel_from_github_raw(github_url):
    """
    Load Excel file directly from GitHub raw URL
    Works best for public repositories
    """
    try:
        response = requests.get(github_url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Read Excel from bytes
        excel_data = BytesIO(response.content)
        df = pd.read_excel(excel_data)
        return df
    except Exception as e:
        st.error(f"Error loading Excel from GitHub: {str(e)}")
        return pd.DataFrame()
# Configure page

st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Interactive Data Analysis Dashboard")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Histogram Analysis", "Confusion Matrix Analysis","GDP Trend Analysis"]
)
import ast
#while figuring out the streamlit version, can look into how to better fine tune BERT for this
def plot_gdp_trend_over_time(df_merged, country="All"):
    """
    Plot GDP trends over time for a specific country or all countries
    """
    if country != "All":
        df_filtered = df_merged[df_merged["Country"] == country].copy()
        title = f'GDP Trend Over Time - {country}'
    else:
        df_filtered = df_merged.copy()
        title = 'GDP Trend Over Time - All Countries'
    
    # Convert numeric dates back to datetime for plotting
    df_filtered['Date_readable'] = pd.to_datetime(df_filtered['Date_y'], unit='ns')
    
    # Sort by date for proper line plotting
    df_filtered = df_filtered.sort_values('Date_readable')
    
    if country == "All":
        # For all countries, create separate lines for each country
        fig = px.line(df_filtered, 
                     x='Date_readable', 
                     y='GDP Growth Rate (%)', 
                     color='Country',
                     title=title,
                     labels={'Date_readable': 'Date', 'GDP Growth Rate (%)': 'GDP Growth Rate (%)'})
    else:
        # For single country
        fig = px.line(df_filtered, 
                     x='Date_readable', 
                     y='GDP Growth Rate (%)', 
                     title=title,
                     labels={'Date_readable': 'Date', 'GDP Growth Rate (%)': 'GDP Growth Rate (%)'},
                     color_discrete_sequence=['#1f77b4'])
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='GDP Growth Rate (%)',
        hovermode='x unified'
    )
    
    return fig

def plot_gdp_increase_over_time(df_merged, country="All"):
    """
    Plot GDP_increase patterns over time with different visualizations
    """
    if country != "All":
        df_filtered = df_merged[df_merged["Country"] == country].copy()
        title_suffix = f' - {country}'
    else:
        df_filtered = df_merged.copy()
        title_suffix = ' - All Countries'
    
    # Convert numeric dates back to datetime
    df_filtered['Date_readable'] = pd.to_datetime(df_filtered['Date_y'], unit='ns')
    df_filtered = df_filtered.sort_values('Date_readable')
    
    # Create subplot with multiple visualizations
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'GDP Increase Over Time{title_suffix}',
            f'GDP Increase Distribution{title_suffix}',
            f'GDP vs GDP Increase{title_suffix}',
            f'GDP Distribution{title_suffix}',
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    if country == "All":
        # Plot 1: Time series of GDP_increase by country
        for country_name in df_filtered['Country'].unique():
            country_data = df_filtered[df_filtered['Country'] == country_name]
            fig.add_trace(
                go.Scatter(x=country_data['Date_readable'], 
                          y=country_data['GDP_Increase'],
                          mode='lines+markers',
                          name=country_name,
                          showlegend=True),
                row=1, col=1
            )
    else:
        # Plot 1: Single country time series
        fig.add_trace(
            go.Scatter(x=df_filtered['Date_readable'], 
                      y=df_filtered['GDP_Increase'],
                      mode='lines+markers',
                      name='GDP Increase',
                      line=dict(color='#1f77b4'),
                      showlegend=False),
            row=1, col=1
        )
    
    # Plot 2: Distribution of GDP_increase
    fig.add_trace(
        go.Histogram(x=df_filtered['GDP_Increase'], 
                    name='Distribution',
                    showlegend=False,
                    marker_color='lightblue'),
        row=1, col=2
    )
    
    # Plot 3: GDP vs GDP_Increase scatter
    if country == "All":
        fig.add_trace(
            go.Scatter(x=df_filtered['GDP Growth Rate (%)'], 
                      y=df_filtered['GDP_Increase'],
                      mode='markers',
                      text=df_filtered['Country'],
                      name='GDP vs Increase',
                      showlegend=False,
                      marker=dict(size=8, opacity=0.7)),
            row=2, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=df_filtered['GDP Growth Rate (%)'], 
                      y=df_filtered['GDP_Increase'],
                      mode='markers',
                      name='GDP vs Increase',
                      showlegend=False,
                      marker=dict(color='orange', size=8)),
            row=2, col=1
        )
    # Plot 2: Distribution of GDP_increase
    fig.add_trace(
        go.Histogram(x=df_filtered['GDP Growth Rate (%)'], 
                    name='Distribution',
                    showlegend=False,
                    marker_color='lightblue'),
        row=2, col=2
    )
    

    
    # Update layout
    fig.update_layout(height=800, 
                     title_text=f"GDP Increase Analysis{title_suffix}",
                     showlegend=True)
    
    # Update axis labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="GDP Increase", row=1, col=1)
    
    fig.update_xaxes(title_text="GDP Increase", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text="GDP", row=2, col=1)
    fig.update_yaxes(title_text="GDP Increase", row=2, col=1)

    fig.update_xaxes(title_text="GDP Value", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    
    return fig
def parts_of_speech_plot(df_merged,country):
    df_merged_filtered=df_merged[df_merged["Country"]==country]
    df_merged_filtered.reset_index(drop=True, inplace=True)
    #get total count for a specific country for each POS
    country_pos_val={}
    for i in range(len(df_merged_filtered)):
        text_dict=df_merged_filtered.loc[i,"pos_tag_weighted"]
        if text_dict.startswith('Counter('):
            inner_dict_str = text_dict[8:-1]  # Remove 'Counter(' and ')'

        # Use ast.literal_eval to safely evaluate the dictionary string
        pos_count= ast.literal_eval(inner_dict_str)
        for key in pos_count.keys():
            if key not in country_pos_val.keys():
                country_pos_val.update({key:pos_count[key]})
            else:
                country_pos_val[key]+=pos_count[key]
    pos_mapping = {
    'INTJ': 'Interjection',
    'SYM': 'Symbol',
    'SPACE': 'Space',
    'X': 'Other',
    'SCONJ': 'Subordinating Conjunction',
    'PART': 'Particle',
    'NUM': 'Numeral',
    'CCONJ': 'Coordinating Conjunction',
    'ADV': 'Adverb',
    'PRON': 'Pronoun',
    'AUX': 'Auxiliary Verb',
    'PROPN': 'Proper Noun',
    'VERB': 'Verb',
    'DET': 'Determiner',
    'ADJ': 'Adjective',
    'PUNCT': 'Punctuation',
    'ADP': 'Adposition',
    'NOUN': 'Noun'}
    POS_Tag=list(country_pos_val.keys())
    POS_Tag_Full_Form = [pos_mapping[tag] for tag in POS_Tag]
    print("POS_Tag",POS_Tag)
    relative_frequencies=list(country_pos_val.values())
    temp_df=pd.DataFrame()
    temp_df["Parts of Speech Tags"]=POS_Tag_Full_Form
    temp_df["Relative Frequency"]=relative_frequencies
    print(temp_df.head(5))
    fig=px.bar(temp_df,
        x="Parts of Speech Tags",y="Relative Frequency",
        title=f'Histogram of Parts of Speech for {country}',
        color_discrete_sequence=['skyblue']
    )
    fig.update_layout(
        xaxis_title=country,
        yaxis_title=f'Relative Frequency to all {country} speeches/news',
        showlegend=False
    )
    return fig
def plot_histogram_plotly(data, column, bins=30):
    """Create interactive histogram using plotly"""
    fig = px.histogram(
        data, 
        x=column, 
        nbins=bins,
        title=f'Interactive Histogram of {column}',
        color_discrete_sequence=['skyblue']
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title='Frequency',
        showlegend=False
    )
    return fig
def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create plotly heatmap
    if labels is None:
        labels = [f'Class {i}' for i in range(len(np.unique(y_true)))]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=600,
        height=600
    )
    
    return fig, cm

def load_from_google_drive(file_id):
    """
    Load dataset from Google Drive using file ID
    Useful when data is stored in Google Drive
    """
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading from Google Drive: {str(e)}")
        return pd.DataFrame()

def save_dataset_to_session(df, name='custom_dataset'):
    """Save dataset to session state for persistence"""
    st.session_state[name] = df
    st.success(f"Dataset '{name}' saved to session!")


    categories=list(country_pos_val.keys())
    values=list(country_pos_val.values())
    plt.bar(categories, values)
    plt.title(f'POS Counts for {country}')
    plt.xlabel('POS Tag')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
df_merged=pd.read_excel("df_merged.xlsx") 

df_merged['Date_y'] = pd.to_numeric(pd.to_datetime(df_merged['Date_y']))

X = df_merged.select_dtypes(include=['float',"int"]).drop(columns=["GDP_Increase"])
y = df_merged["GDP_Increase"]

test_size = 0.2

# Use chronological split instead to preserve time-series relation
split_index = int(len(X) * (1 - test_size))
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
scaler = StandardScaler()
scaler.fit(X_train)
# Transform both training and test data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#'classifier__C': np.float64(10.0), 'classifier__max_iter': 1000, 'classifier__penalty': 'l1', 'classifier__solver': 'liblinear'

model = LogisticRegression(penalty="l1",C=10.0,max_iter=1000,solver="liblinear")
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
y_true, y_pred = y_test,y_pred
# Main app logic
if app_mode == "Histogram Analysis":
    st.header("📈 Histogram Analysis")
    st.markdown("---")
    # Methodology section
    st.markdown("""
**Methodology and Data Results**

Methodology and Data Results:

At first we merge two different datasets- one containing economic indicators like unemployment level, inflation, GDP etc from that has been compiled by MIT from sources like the Word Bank and another- containing speeches from different central banks compiled by a group of research to complement the original source made by the Bank of International settlement. As there were few matches between the dataset, we also utilized news updates across countries through the AlphaVantage API to filter for news/economic data. As there were low matches between exact dates across the datasets, we used the year as our merging condition for the record of GDP as well as the textual data either speech or news. 

Creating Target Variable:
We first merge the two datasets based on common columns like Date and Country to create our main dataset. On it we then create our target column-”GDP Increase” to contain a “Yes” if the previous year in our dataset had lower value than the current one for a specific country. 

Word2Vec Embedding:
We then apply the pretrained model-Word2Vec- to get an indicative speech level embedding by going through each of the sentences for a speech. The model that comes with an input, hidden and an output layer gets trained on each of the words in the vocabulary in order. We use it to create two versions of embeddings- Continuous Bag of Words (CBOW) and Skip-Gram(Sgram). With the former we try to predict the current word being looked at based on the surrounding words and with the latter we try to predict surrounding words based on the current word being looked at. Both versions provide a better understanding of semantic relations so we apply both to create word embeddings for every speech. As we will be using logistic regression in our model, we first calculate an average vector based on the vectors in  sentence and then take the average value of components of this vector to represent a final scalar value from the word2vec models which we utilize as features. 

We then evaluate the effectiveness of our models by comparing the cosine similarity our model generates between words compared to the amount generated from a human annotated dataset of SimLex-999 which has a similarity score between words as well. 


The results showed that almost 80% of the word pairs had at least one word not found in our model’s dictionary also not found in our dictionary. This was gained even after adjusting hyperparameters of the model indicating how the wordset in the dataset might be more novel compared to the one in SimLex. 

Parts of Speech Tagging:

We utilized the en_core_web_sm model from the spaCy natural language processing library to process the grammar in every speech. We get pairing of words and their respective grammar like (word, Noun) across the vocab covering objects like Noun, Verb and Adj. We then perform a total count of the grammar objects across the dataset to then do a relative count to the total for each speech. This relative Part of Speech frequency is then added as features to the dataset to be used in our model. 

We analyze the data further by checking if the relative frequency of grammar objects is different across countries and plotting relative frequencies across different countries. 

With different parts of speeches being present across countries and concentration of objects like Interjections  being different in speeches across countries like for the USA and Japan, there seems to be a clear indication for different methods of communication. 

Below you can play around with the dataset to filter it and see some of the distributions of our features:""")
    st.markdown("---")

    # Display data info
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df_merged))
    with col2:
        st.metric("Columns", len(df_merged.columns))
    with col3:
        st.metric("Numeric Columns", len(df_merged.select_dtypes(include=[np.number]).columns))
    
    # Show sample data
    if st.checkbox("Show sample data"):
        st.dataframe(df_merged.head())
    
    # Column selection
    numeric_columns = df_merged.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_columns:
        st.error("No numeric columns found in the dataset!")
        st.stop()
    
    selected_column = st.selectbox("Select column for histogram:", numeric_columns)
    selected_country= st.selectbox("Choose Country to filter the dataset by:",['USA', 'Japan', 'France', 'Canada', 'Australia',"All"])
 
    # Histogram customization
    st.subheader("Histogram Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        bins = st.slider("Number of bins:", min_value=5, max_value=100, value=30)
    
    with col2:
        plot_type = st.radio("Plot type:", ["Interactive (Plotly)"])
    
    # Generate histogram
    if st.button("Generate Histogram"):
        st.subheader(f"Histogram of {selected_column}")
        if selected_country!="All":
          df=df_merged[df_merged["Country"]==selected_country]
        else:
          df=df_merged
        fig = plot_histogram_plotly(df_merged, selected_column, bins)
        st.plotly_chart(fig, use_container_width=True,key=1)
        
        fig2=parts_of_speech_plot(df,selected_country)
        st.plotly_chart(fig2,use_container_width=True,key=2)
        # Display statistics
        st.subheader("Statistics of all the countries")
        stats_df = df_merged.drop(columns=["Date_y"]).describe()
        st.dataframe(stats_df)
elif app_mode == "Confusion Matrix Analysis":
    st.info("Using Logistic Regression Hypertuned via Grid Search we found the following results")
    if st.button("Generate Confusion Matrix"):
        fig, cm = plot_confusion_matrix(y_true, y_pred)
        st.plotly_chart(fig, use_container_width=True)

        # Display metrics
        st.subheader("Classification Metrics")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        col1, col2, col3, col4,col5 = st.columns(5)
        with col1:
          st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
        with col2:
          st.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.3f}")
        with col3:
          st.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.3f}")
        with col4:
          st.metric("F1-Score", f"{f1_score(y_true, y_pred, average='weighted'):.3f}")
        with col5:
            st.metric("ROC", f"{roc_auc_score(y_true, y_pred, average='weighted'):.3f}")
                st.markdown("---")
        st.markdown("""
**Logistic Regression Analysis **
Before we apply the logistic regression model, we ensure the data is well prepared for the model. We maintain chronological order of the GDP by ordering the dataset by date, this is to ensure that sequence with time dependent data is maintained and perform Time Series cross validation with 5 folds to get the best hyperparameters for our model’s performance. By 5 folds, we ensure that it iteratively trains and tests itself by separating chunks of the training data as validation data with each trying a different combination of parameters to see which one leads to the best result overall in training. We use roc_auc as our main metric for comparison as this indicates how well our model can distinguish between our two target classes. The best hyperparameter combination for our model that we get is the following:

-Regularization Strength (C=10): This is the inverse regularization indicator, showing that our model benefits from moderate regularization of historical data and higher values can lead to overfitting to our training data. 
-Penalty Regularization (L1): L1 regularization(Lasso) indicates that our model performs better by letting some of our model features become irrelevant with a zero coefficient. 
-Solver Algorithm(liblinear): This hyperparameter indicates the method through which the model can find the correct weight in terms of importance in features for better predictions for our model and in this case by trying different values at a time for each feature leads to a better result. 
-Maximum Iterations(1000): This indicates the optimal number of steps required for the solver to find out the correct weight to give to each feature in our dataset

Below is the resulting confusion matrix that we get from our model detailing our predictions:


Based on the confusion matrix we realize that the model has the following metrics:
-Precision: When making the prediction of 1(Increase in GDP), the model is correct by 90% of the time. 
-Accuracy: When the model makes predictions, it makes the correct one 90% of the time. 
-ROC_AUC: The model’s ability to differentiate between increase and decrease in GDP cases is 90%

Additionally in terms of feature importance the following were the most important in predictions:
-GDP Growth Rate-30%
-speech_embedding_CBOW-16%
-speech_embedding_Sgram-8%

We also trained a FIN-BERT model which is a pre-trained NLP model that has been trained on financial data for sentiment analysis, our results were similar much worse compared to that of our hyper-parameter tuned Logistic Regression model with an auc_roc score of 0.5. This indicates in order to check if a pre-trained model can perform better, we would need to finetune it to our dataset and also acquire more data for further improved training
""")
        st.markdown("---")


elif app_mode == "GDP Trend Analysis":
    st.header("📈 GDP Trend Analysis")
    st.markdown("---")
    # Data Visuals section
    st.markdown("""
**Data Vissual Anlaysis**
This section is meant to check for common trends in GDP values and GDP increases across countries via data visuals. Based on looking at figures for countries like Canada and France there does not seem to be a consistent distribution or trend over time between countries. Below is what the data visuals you generate is meant to show:

-GDP Trend Over Time: A trendline meant to showcase the rate of increase in GDP values for countries (In general due to inflation there is an overall increasing trend with a few dips across the years for each country)
-GDP Increase Over Time: Showcases if GDP increased or decreased for consecutive periods of time with a darker shade representing that GDP increased/decreased more frequently in timer horizon
-GDP Increase Distribution: Shows the count of 0s and 1s for a specific country in our a dataset 
-GDP Values Distribution: Shows the count of GDP values for the country being referenced in our dataset

""")
    st.markdown("---")   
    # Country selection
    selected_country = st.selectbox(
        "Choose Country for GDP Analysis:",
        ['All', 'USA', 'Japan', 'France', 'Canada', 'Australia']
    )
    
    # Analysis type selection
    analysis_type = st.radio(
        "Select Analysis Type:",
        ["GDP Trend Over Time", "GDP Increase Analysis", "Both"]
    )
    
    if st.button("Generate GDP Analysis"):
        if analysis_type in ["GDP Trend Over Time", "Both"]:
            st.subheader("GDP Trend Over Time")
            fig1 = plot_gdp_trend_over_time(df_merged, selected_country)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Display summary statistics
            if selected_country != "All":
                country_data = df_merged[df_merged["Country"] == selected_country]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average GDP", f"{country_data['GDP Growth Rate (%)'].mean():.2f}")
                with col2:
                    st.metric("Max GDP", f"{country_data['GDP Growth Rate (%)'].max():.2f}")
                with col3:
                    st.metric("Min GDP", f"{country_data['GDP Growth Rate (%)'].min():.2f}")
                with col4:
                    st.metric("GDP Growth Rate", f"{((country_data['GDP Growth Rate (%)'].iloc[-1] / country_data['GDP Growth Rate (%)'].iloc[0]) - 1) * 100:.2f}%")
        
        if analysis_type in ["GDP Increase Analysis", "Both"]:
            st.subheader("GDP Increase Analysis")
            fig2 = plot_gdp_increase_over_time(df_merged, selected_country)
            st.plotly_chart(fig2, use_container_width=True)
            
            # GDP Increase statistics
            if selected_country != "All":
                country_data = df_merged[df_merged["Country"] == selected_country]
            else:
                country_data = df_merged
                
            increase_count = (country_data['GDP_Increase'] == 1).sum()
            total_count = len(country_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("GDP Increase Rate", f"{(increase_count/total_count*100):.1f}%")
            with col2:
                st.metric("Periods with GDP Increase", f"{increase_count}")
            with col3:
                st.metric("Total Periods", f"{total_count}")




            

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit")

# Instructions for Google Colab
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Run in Google Colab")


# Colab setup code
colab_setup = '''
# Run this in Google Colab to set up ngrok tunnel
from pyngrok import ngrok
import subprocess
import time

# Kill any existing streamlit processes
subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)

# Start streamlit in background
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])

# Wait for streamlit to start
time.sleep(10)

# Create ngrok tunnel
public_url = ngrok.connect(8501)
print(f"Streamlit app is available at: {public_url}")
'''

if st.sidebar.button("Show Colab Setup Code"):
    st.sidebar.code(colab_setup, language="python")
