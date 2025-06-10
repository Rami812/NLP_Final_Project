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

import requests
from io import BytesIO

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
    ["Histogram Analysis", "Confusion Matrix Analysis"]
)
import ast
#while figuring out the streamlit version, can look into how to better fine tune BERT for this
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
    POS_Tag=list(country_pos_val.keys())
    relative_frequencies=list(country_pos_val.values())
    temp_df=pd.DataFrame()
    temp_df["Parts of Speech Tags"]=POS_Tag
    temp_df["Relative Frequencies"]=relative_frequencies
    print(temp_df.head(5))
    fig=px.bar(temp_df,
        x="Parts of Speech Tags",y="Relative Frequencies", 
        title=f'Interactive Histogram of Parts of Speech for {country}',
        color_discrete_sequence=['skyblue']
    )
    fig.update_layout(
        xaxis_title=country,
        yaxis_title='Frequency',
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
df_merged.drop(columns=["Unnamed: 0","Unnamed: 0.1"],inplace=True)

df_merged['Date_x'] = pd.to_numeric(pd.to_datetime(df_merged['Date_x']))
df_merged['Date_y'] = pd.to_numeric(pd.to_datetime(df_merged['Date_y']))
X = df_merged.select_dtypes(include=['float'])
y = df_merged["GDP_Increase"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
y_true, y_pred = y_test,y_pred
# Main app logic
if app_mode == "Histogram Analysis":
    st.header("📈 Histogram Analysis")


    # Display data info
    st.subheader("Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df_merged))
    with col2:
        st.metric("Columns", len(df_merged.columns))
    with col3:
        st.metric("Numeric Columns", len(df_merged.drop(columns=["index","Date_x","Date_y"]).select_dtypes(include=[np.number]).columns))
    
    # Show sample data
    if st.checkbox("Show sample data"):
        st.dataframe(df_merged.head())
    
    # Column selection
    numeric_columns = df_merged.drop(columns=["index","Date_x","Date_y"]).select_dtypes(include=[np.number]).columns.tolist()
    
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
        st.subheader("Statistics of the all countries")
        stats_df = df_merged.drop(columns=["Date_x","Date_y","index"]).describe()
        st.dataframe(stats_df)
elif app_mode == "Confusion Matrix Analysis":
    st.header("🎯 Confusion Matrix Analysis")
    
 

    st.info("Using Logistic Regression  model with 2 classes")        
    if st.button("Generate Confusion Matrix"):
        fig, cm = plot_confusion_matrix(y_true, y_pred)
        st.plotly_chart(fig, use_container_width=True)
            
        # Display metrics
        st.subheader("Classification Metrics")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
        col1, col2, col3, col4 = st.columns(4)
        with col1:
          st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
        with col2:
          st.metric("Precision", f"{precision_score(y_true, y_pred, average='weighted'):.3f}")
        with col3:
          st.metric("Recall", f"{recall_score(y_true, y_pred, average='weighted'):.3f}")
        with col4:
          st.metric("F1-Score", f"{f1_score(y_true, y_pred, average='weighted'):.3f}")


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
