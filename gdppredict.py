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
            f'Monthly GDP Increase Pattern{title_suffix}'
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
    
    # Plot 4: Monthly pattern (extract month from date)
    df_filtered['Month'] = df_filtered['Date_readable'].dt.month
    monthly_avg = df_filtered.groupby('Month')['GDP_Increase'].mean().reset_index()
    
    fig.add_trace(
        go.Bar(x=monthly_avg['Month'], 
               y=monthly_avg['GDP_Increase'],
               name='Monthly Average',
               showlegend=False,
               marker_color='green'),
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
    
    fig.update_xaxes(title_text="Month", row=2, col=2)
    fig.update_yaxes(title_text="Avg GDP Increase", row=2, col=2)
    
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
        st.dataframe(df_merged.drop(columns=["index","Date_x","Date_y"]).head())
    
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
        st.subheader("Statistics of all the countries")
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

elif app_mode == "GDP Trend Analysis":
    st.header("📈 GDP Trend Analysis")
    
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
                    st.metric("Average GDP", f"{country_data['GDP'].mean():.2f}")
                with col2:
                    st.metric("Max GDP", f"{country_data['GDP'].max():.2f}")
                with col3:
                    st.metric("Min GDP", f"{country_data['GDP'].min():.2f}")
                with col4:
                    st.metric("GDP Growth Rate", f"{((country_data['GDP'].iloc[-1] / country_data['GDP'].iloc[0]) - 1) * 100:.2f}%")
        
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
