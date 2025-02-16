import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.text_analyzer import TextAnalyzer
from src.document_processor import DocumentProcessor
from src.language_tools import LanguageTools
import tempfile
import os
from pathlib import Path

# Initialize the NLP components
@st.cache_resource
def load_nlp_components():
    return {
        'analyzer': TextAnalyzer(),
        'processor': DocumentProcessor(),
        'language_tools': LanguageTools()
    }

nlp = load_nlp_components()

# Set page config
st.set_page_config(
    page_title="Advanced NLP Toolkit",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTextArea textarea {
        height: 200px;
    }
    .stButton>button {
        width: 100%;
    }
    .plot-container {
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🔤 NLP Toolkit")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type",
    ["Text Analysis", "Document Processing", "Language Tools"]
)

def plot_sentiment_distribution(sentiment):
    """Create a sentiment distribution plot."""
    fig = go.Figure(data=[
        go.Bar(
            x=['Negative', 'Neutral', 'Positive'],
            y=[sentiment['negative'], sentiment['neutral'], sentiment['positive']],
            marker_color=['#ff9999', '#ffcc99', '#99ff99']
        )
    ])
    fig.update_layout(
        title="Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Score",
        showlegend=False
    )
    st.plotly_chart(fig)

def plot_key_phrases(phrases):
    """Create a horizontal bar chart for key phrases."""
    if phrases:
        df = pd.DataFrame(phrases)
        fig = px.bar(
            df,
            x='score',
            y='phrase',
            orientation='h',
            title="Key Phrases"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig)

def plot_pos_distribution(pos_tags):
    """Create a pie chart for POS tag distribution."""
    if pos_tags:
        fig = px.pie(
            values=list(pos_tags.values()),
            names=list(pos_tags.keys()),
            title="Parts of Speech Distribution"
        )
        st.plotly_chart(fig)

def plot_readability_metrics(readability):
    """Create a radar chart for readability metrics."""
    if readability:
        metrics = ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'smog_index']
        values = [readability[m] for m in metrics]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2]
                )),
            showlegend=False,
            title="Readability Metrics"
        )
        st.plotly_chart(fig)

def display_text_analysis():
    """Display text analysis interface."""
    st.title("Text Analysis")
    
    text_input = st.text_area(
        "Enter text to analyze",
        "Enter your text here..."
    )
    
    if st.button("Analyze Text"):
        if text_input and text_input != "Enter your text here...":
            with st.spinner("Analyzing text..."):
                results = nlp['analyzer'].analyze(text_input)
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Sentiment Analysis")
                    plot_sentiment_distribution(results['sentiment'])
                    
                    st.subheader("Key Phrases")
                    plot_key_phrases(results['key_phrases'])
                
                with col2:
                    st.subheader("Parts of Speech")
                    plot_pos_distribution(results['pos_tags'])
                    
                    st.subheader("Readability")
                    plot_readability_metrics(results['readability'])
                
                # Display entities
                st.subheader("Named Entities")
                if results['entities']:
                    df = pd.DataFrame(results['entities'])
                    st.dataframe(df)
                else:
                    st.info("No named entities found.")
                
                # Display summary
                st.subheader("Summary")
                st.write(results['summary'])
        else:
            st.warning("Please enter some text to analyze.")

def display_document_processing():
    """Display document processing interface."""
    st.title("Document Processing")
    
    uploaded_file = st.file_uploader("Upload a document", type=['txt', 'pdf', 'docx'])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                results = nlp['processor'].process_document(tmp_path)
                
                # Display results
                st.subheader("Document Analysis")
                
                # Topics
                st.write("**Topics:**")
                for topic in results['topics']:
                    st.write(f"- {topic}")
                
                # Structure
                st.write("**Document Structure:**")
                st.json(results['structure'])
                
                # Statistics
                st.write("**Document Statistics:**")
                st.json(results['statistics'])
                
                # Summary
                st.subheader("Document Summary")
                st.write(results['summary'])
                
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
    else:
        st.info("Please upload a document to process.")

def display_language_tools():
    """Display language tools interface."""
    st.title("Language Tools")
    
    tool_type = st.radio(
        "Select Tool",
        ["Language Detection", "Translation", "Grammar Check"]
    )
    
    if tool_type == "Language Detection":
        text = st.text_area("Enter text to detect language", "")
        
        if st.button("Detect Language"):
            if text:
                with st.spinner("Detecting language..."):
                    result = nlp['language_tools'].detect_language(text)
                    st.success(f"Detected Language: {result['language']}")
                    if 'confidence' in result:
                        st.info(f"Confidence: {result['confidence']:.2f}")
            else:
                st.warning("Please enter text to detect language.")
    
    elif tool_type == "Translation":
        col1, col2 = st.columns(2)
        
        with col1:
            source_lang = st.selectbox(
                "Source Language",
                ["auto", "en", "es", "fr", "de", "it", "pt", "nl", "ru", "ja", "ko", "zh"]
            )
        
        with col2:
            target_lang = st.selectbox(
                "Target Language",
                ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "ja", "ko", "zh"]
            )
        
        text = st.text_area("Enter text to translate", "")
        
        if st.button("Translate"):
            if text:
                with st.spinner("Translating..."):
                    result = nlp['language_tools'].translate(
                        text,
                        target_lang=target_lang,
                        source_lang=source_lang
                    )
                    
                    if 'translated_text' in result:
                        st.subheader("Translation")
                        st.write(result['translated_text'])
                    else:
                        st.error("Translation failed. Please try again.")
            else:
                st.warning("Please enter text to translate.")
    
    else:  # Grammar Check
        text = st.text_area("Enter text to check grammar", "")
        
        if st.button("Check Grammar"):
            if text:
                with st.spinner("Checking grammar..."):
                    result = nlp['language_tools'].check_grammar(text)
                    
                    if result['is_correct']:
                        st.success("No grammar errors found!")
                    else:
                        st.warning("Grammar issues found:")
                        for suggestion in result['suggestions']:
                            st.write(f"- {suggestion}")
            else:
                st.warning("Please enter text to check grammar.")

# Main app logic
if analysis_type == "Text Analysis":
    display_text_analysis()
elif analysis_type == "Document Processing":
    display_document_processing()
else:  # Language Tools
    display_language_tools()

# Footer
st.markdown("---")
st.markdown(
    "Created by [Abhishek Banerjee](mailto:abhishekninja@yahoo.com) | "
    "[GitHub](https://github.com/newloverofcode2025/portfolio-projects)"
)
