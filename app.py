import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from train import create_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

class ImagePreprocessor:
    @staticmethod
    def adjust_brightness(image, factor):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def adjust_contrast(image, factor):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def adjust_sharpness(image, factor):
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    @staticmethod
    def apply_clahe(image):
        lab = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final)

# Model-related functions
def create_model(num_classes):
    import torchvision.models as models
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model

def load_model(model_path, num_classes, device):
    try:
        model = create_model(num_classes)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_image(image, model, label_mapping, device):
    if model is None:
        st.error("Model not loaded properly")
        return None
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            confidences, predictions = torch.topk(probabilities, min(3, len(label_mapping)))
            confidences = confidences[0].cpu().numpy()
            predictions = predictions[0].cpu().numpy()
            
            results = []
            idx_to_class = {v: k for k, v in label_mapping.items()}
            for pred, conf in zip(predictions, confidences):
                results.append((idx_to_class[pred], float(conf)))
            
            return results
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None
    

# Disease Information Database
DISEASE_INFO = {
    # Cashew Diseases
    'Cashew anthracnose': {
        'symptoms': [
            'Dark brown to black spots on leaves and fruits',
            'Sunken lesions on fruits',
            'Wilting and death of young shoots',
            'Leaf spots with dark margins',
            'Discoloration of young stems',
            'Premature leaf fall',
            'Dieback of twigs and branches'
        ],
        'treatment': [
            'Remove and destroy infected plant parts',
            'Maintain good air circulation through proper spacing',
            'Apply fungicides containing copper or mancozeb',
            'Prune infected branches during dry weather',
            'Use biological control agents like Trichoderma',
            'Ensure proper sanitation of tools',
            'Apply protective fungicides before rainy season'
        ],
        'pesticides': [
            'Copper oxychloride (2-3g/L)',
            'Carbendazim (1g/L)',
            'Mancozeb (2.5g/L)',
            'Thiophanate-methyl (1g/L)',
            'Chlorothalonil (2g/L)',
            'Propineb (2g/L)',
            'Hexaconazole (1ml/L)'
        ],
        'prevention': [
            'Use disease-free planting material',
            'Maintain proper drainage',
            'Avoid overhead irrigation',
            'Regular monitoring of trees',
            'Apply balanced fertilization',
            'Maintain proper tree spacing',
            'Control weeds around trees'
        ],
        'environmental_conditions': {
            'temperature': '20-28°C',
            'humidity': '80-95%',
            'rainfall': 'High rainfall promotes disease spread',
            'season': 'Most severe during rainy season'
        },
        'economic_impact': 'Can cause 30-40% yield loss if not managed properly'
    },
    'Cashew healthy': {
        'characteristics': [
            'Deep green leaves',
            'Uniform leaf color',
            'No spots or lesions',
            'Vigorous growth',
            'Normal fruit development',
            'Good canopy development',
            'Healthy root system'
        ],
        'maintenance': [
            'Regular irrigation',
            'Balanced fertilization',
            'Proper pruning',
            'Weed management',
            'Regular monitoring',
            'Soil health maintenance',
            'Integrated pest management'
        ],
        'optimal_conditions': {
            'temperature': '20-35°C',
            'rainfall': '1000-2000mm annually',
            'soil_pH': '5.5-6.5',
            'spacing': '7-8m between trees',
            'sunlight': 'Full sun exposure'
        },
        'best_practices': [
            'Regular health monitoring',
            'Timely pruning',
            'Proper irrigation schedule',
            'Soil testing',
            'Nutrient management'
        ]
    },
    # Cassava Diseases
    'Cassava mosaic': {
        'symptoms': [
            'Yellow or pale green mosaic pattern on leaves',
            'Leaf distortion and reduced size',
            'Stunted plant growth',
            'Reduced tuber size and yield',
            'Chlorotic patches between leaf veins'
        ],
        'treatment': [
            'Remove infected plants',
            'Use virus-free planting materials',
            'Control whitefly populations',
            'Plant resistant varieties',
            'Practice crop rotation'
        ],
        'pesticides': [
            'Imidacloprid (0.5ml/L) for whitefly control',
            'Thiamethoxam (0.2g/L)',
            'Acetamiprid (0.4g/L)',
            'Neem-based insecticides',
            'Yellow sticky traps for monitoring'
        ],
        'prevention': [
            'Use certified disease-free cuttings',
            'Early detection and roguing',
            'Maintain field sanitation',
            'Plant during low whitefly seasons',
            'Use barrier crops'
        ],
        'environmental_conditions': {
            'temperature': '25-30°C',
            'humidity': '70-80%',
            'vectors': 'Transmitted by whiteflies'
        },
        'economic_impact': 'Can cause 20-95% yield loss in susceptible varieties'
    },
    'Cassava healthy': {
        'characteristics': [
            'Dark green leaves',
            'Proper branching',
            'Good stem development',
            'Vigorous root system',
            'Normal leaf size',
            'Uniform growth'
        ],
        'maintenance': [
            'Proper spacing',
            'Weed control',
            'Adequate irrigation',
            'Nutrient management',
            'Regular monitoring'
        ],
        'optimal_conditions': {
            'temperature': '25-30°C',
            'rainfall': '1000-1500mm annually',
            'soil_pH': '5.5-7.0',
            'spacing': '1m x 1m'
        },
        'best_practices': [
            'Quality planting material',
            'Proper land preparation',
            'Timely harvesting',
            'Crop rotation'
        ]
    },
    # Add more diseases as needed...
}

# UI Helper Functions for Disease Information
def get_disease_category(disease_name):
    """
    Returns the crop category for a given disease
    """
    if 'Cashew' in disease_name:
        return 'Cashew'
    elif 'Cassava' in disease_name:
        return 'Cassava'
    elif 'Maize' in disease_name:
        return 'Maize'
    elif 'Tomato' in disease_name:
        return 'Tomato'
    return 'Unknown'

def is_healthy_state(disease_name):
    """
    Checks if the disease name represents a healthy state
    """
    return 'healthy' in disease_name.lower()

def get_disease_color(disease_name):
    """
    Returns appropriate color coding for disease states
    """
    if is_healthy_state(disease_name):
        return '#4CAF50'  # Green for healthy
    else:
        return '#f44336'  # Red for disease
    

def display_disease_info(disease_name, disease_info):
    """
    Enhanced disease information display with modern UI components
    """
    # Disease header with category-based styling
    category = get_disease_category(disease_name)
    st.markdown(f"""
    <div class="info-card">
        <div class="card-header">{disease_name}</div>
""", unsafe_allow_html=True)

    # Create tabbed interface for disease information
    if is_healthy_state(disease_name):
        tabs = st.tabs(["🌱 Characteristics", "🔧 Maintenance", "🌍 Optimal Conditions", "✅ Best Practices"])
        
        with tabs[0]:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            for char in disease_info['characteristics']:
                st.markdown(f"• {char}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tabs[1]:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            for maint in disease_info['maintenance']:
                st.markdown(f"• {maint}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tabs[2]:
            display_conditions(disease_info['optimal_conditions'])
            
        with tabs[3]:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            for practice in disease_info['best_practices']:
                st.markdown(f"• {practice}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        tabs = st.tabs(["🔍 Symptoms", "💊 Treatment", "🛡️ Prevention", "🌡️ Conditions", "📊 Impact"])
        
        with tabs[0]:
            st.markdown('<div class="info-card symptoms-card">', unsafe_allow_html=True)
            for symptom in disease_info['symptoms']:
                st.markdown(f"• {symptom}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="info-card treatment-card">', unsafe_allow_html=True)
                st.subheader("Treatment Methods")
                for treatment in disease_info['treatment']:
                    st.markdown(f"• {treatment}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="info-card pesticides-card">', unsafe_allow_html=True)
                st.subheader("Recommended Pesticides")
                for pesticide in disease_info['pesticides']:
                    st.markdown(f"• {pesticide}")
                st.markdown('</div>', unsafe_allow_html=True)
                
        with tabs[2]:
            st.markdown('<div class="info-card prevention-card">', unsafe_allow_html=True)
            for prevention in disease_info['prevention']:
                st.markdown(f"• {prevention}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with tabs[3]:
            display_conditions(disease_info['environmental_conditions'])
            
        with tabs[4]:
            display_impact(disease_info['economic_impact'])

def display_conditions(conditions):
    """
    Display environmental conditions in a grid of metric cards
    """
    cols = st.columns(len(conditions))
    for idx, (condition, value) in enumerate(conditions.items()):
        with cols[idx]:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{condition.replace('_', ' ').title()}</div>
                    <div class="metric-value">{value}</div>
                </div>
            """, unsafe_allow_html=True)

def display_impact(impact):
    """
    Display economic impact information in a styled card
    """
    st.markdown(f"""
        <div class="impact-card">
            <h3>📈 Economic Impact</h3>
            <p>{impact}</p>
        </div>
    """, unsafe_allow_html=True)

def plot_prediction_chart(predictions):
    """
    Create an enhanced prediction distribution chart
    """
    diseases, confidences = zip(*predictions)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create bars with gradient colors
    colors = ['#4CAF50' if 'healthy' in d.lower() else '#f44336' for d in diseases]
    bars = ax.bar(diseases, confidences, color=colors, alpha=0.8)
    
    # Customize chart appearance
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Confidence Score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def display_metrics(disease, confidence):
    """
    Display prediction metrics in styled cards
    """
    severity = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
    color = "#f44336" if severity == "High" else "#ff9800" if severity == "Medium" else "#4CAF50"
    
    st.markdown(f"""
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-title">Disease</div>
                <div class="metric-value">{disease}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Confidence</div>
                <div class="metric-value">{confidence:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Risk Level</div>
                <div class="metric-value" style="color: {color}">{severity}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_history_entry(entry):
    """
    Display a single history entry in a styled card
    """
    st.markdown("""
        <div class="history-card">
            <div class="history-header">
                <div class="history-timestamp">{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</div>
            </div>
            <div class="history-content">
                <div class="history-image">
                    {st.image(entry['image'], use_container_width=True)}
                </div>
                <div class="history-predictions">
                    {predictions_list(entry['predictions'])}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def predictions_list(predictions):
    """
    Format predictions for display
    """
    return "\n".join([f"<div class='prediction-item'>{disease}: {conf:.1%}</div>" 
                     for disease, conf in predictions])

def apply_custom_css():
    """Apply custom CSS styles to the application"""
    st.markdown("""
        <style>
        /* Main Layout and Colors */
        .stApp {
            background-color: #f5f7f9;
        }
        
        .main-title {
            color: #1a472a;
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            padding: 1rem;
            margin-bottom: 2rem;
            background: linear-gradient(120deg, #e8f5e9, #c8e6c9);
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Fix for streamlit container */
        .block-container {
            padding-top: 5rem;
            padding-bottom: 0;
            max-width: 100%;
        }
        
        /* Cards Styling */
        .upload-card, .results-card, .history-card, .info-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border: 1px solid #e0e0e0;
        }
        
        .card-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a472a;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #4CAF50;
        }
        
        .upload-card {
            border-left: 5px solid #4CAF50;
        }
        
        .results-card {
            border-left: 5px solid #2196F3;
        }
        
        /* Fix header alignments */
        h1, h2, h3, h4, h5, h6 {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* Fix streamlit elements alignment */
        .stMarkdown {
            margin-bottom: 0 !important;
        }
        
        .element-container {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Metrics Display */
        .metrics-container {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .metric-card {
            background-color: white;
            padding: 1.2rem;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            text-align: center;
            flex: 1;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
        }
        
        .metric-title {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            color: #2e7d32;
            font-size: 1.4rem;
            font-weight: bold;
        }
        
        /* Disease Information Display */
        .disease-header {
            background: linear-gradient(120deg, #4CAF50, #2E7D32);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .disease-header h2 {
            margin: 0;
            font-size: 1.8rem;
        }
        
        .category-tag {
            background-color: rgba(255,255,255,0.2);
            padding: 0.3rem 1rem;
            border-radius: 15px;
            font-size: 0.9rem;
            margin-top: 0.5rem;
            display: inline-block;
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #f5f7f9;
            border-radius: 10px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 500;
            color: #666;
            transition: all 0.2s;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            padding: 1rem 0;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(120deg, #4CAF50, #2E7D32);
            color: white;
            border: none;
            padding: 0.7rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
            width: 100%;
            margin-top: 1rem;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46,125,50,0.2);
        }
        
        /* History Display */
        .history-container {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .prediction-item {
            display: flex;
            justify-content: space-between;
            padding: 0.8rem;
            border-bottom: 1px solid #eee;
            transition: background-color 0.2s;
        }
        
        .prediction-item:hover {
            background-color: #f8f9fa;
        }
        
        .disease-name {
            color: #1a472a;
            font-weight: 500;
        }
        
        .confidence {
            color: #4CAF50;
            font-weight: bold;
        }
        
        /* Image Upload Area */
        .uploadfile {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background-color: #f8f9fa;
            transition: all 0.3s;
            margin: 1rem 0;
        }
        
        .uploadfile:hover {
            border-color: #2E7D32;
            background-color: #f0f7f0;
        }
        
        /* Sidebar Styling */
        .css-1d391kg {
            background-color: #f5f7f9;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        
        /* Info Cards */
        .symptoms-card {
            border-left: 5px solid #f44336;
        }
        
        .treatment-card {
            border-left: 5px solid #2196F3;
        }
        
        .prevention-card {
            border-left: 5px solid #4CAF50;
        }
        
        .pesticides-card {
            border-left: 5px solid #ff9800;
        }
        
        /* Card Content */
        .card-content {
            padding: 1rem 0;
        }
        
        /* Fix for nested elements */
        .upload-card > div {
            margin-bottom: 1rem;
        }
        
        .upload-card > div:last-child {
            margin-bottom: 0;
        }
        
        /* Section spacing */
        .section-container {
            margin: 1rem 0;
            padding: 0;
        }
        
        /* Text containers */
        .text-container {
            text-align: left;
            margin: 0.5rem 0;
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background-color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            border: 1px solid #e0e0e0;
            margin: 0.5rem 0;
        }
        
        /* Loading Spinner */
        .stSpinner > div {
            border-top-color: #4CAF50 !important;
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            background-color: #333;
            color: white;
            text-align: center;
            padding: 5px 10px;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
        }
    
        .section-header {
            color: #1a472a;
            font-size: 1.8rem;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #4CAF50;
        }
        
        .info-section {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .condition-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin: 0.5rem 0;
            border: 1px solid #e0e0e0;
        }
        
        .condition-title {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        
        .condition-value {
            color: #2e7d32;
            font-size: 1.2rem;
            font-weight: bold;
        }
        
        .symptoms {
            border-left: 4px solid #f44336;
        }
        
        .treatment {
            border-left: 4px solid #2196F3;
        }
        
        .prevention {
            border-left: 4px solid #4CAF50;
        }
        
        .conditions {
            border-left: 4px solid #ff9800;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Fix for markdown content */
        .stMarkdown > div > div > p {
            margin-bottom: 0.5rem;
        }
        
        .stMarkdown ul {
            padding-left: 1.5rem;
            margin: 0.5rem 0;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .metrics-container {
                flex-direction: column;
            }
            
            .metric-card {
                margin-bottom: 1rem;
            }
            
            .main-title {
                font-size: 2rem;
            }
            
            .card-header {
                font-size: 1.3rem;
            }
        }
                
        .results-section {
            margin-top: 2rem;
            border-top: 2px solid #e0e0e0;
            padding-top: 2rem;
        }
        
        .results-title {
            color: #1a472a;
            font-size: 2rem;
            font-weight: 600;
            text-align: center;
            margin-bottom: 1.5rem;
            background: linear-gradient(120deg, #e8f5e9, #c8e6c9);
            padding: 1rem;
            border-radius: 10px;
        }
        
        .full-width-card {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
            border: 1px solid #e0e0e0;
            width: 100%;
        }
        
        .analysis-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin: 1.5rem 0;
        }
        
        .prediction-chart {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .alternatives-list {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
        }
        
        .alternative-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem;
            background: white;
            border-radius: 8px;
            margin: 0.5rem 0;
            transition: all 0.2s;
        }
        
        .alternative-item:hover {
            transform: translateX(5px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .disease-info-tabs {
            margin-top: 2rem;
            border-top: 1px solid #e0e0e0;
            padding-top: 1rem;
        }
        
        .tab-content {
            padding: 1.5rem;
            background: white;
            border-radius: 0 10px 10px 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .info-item {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
        }
        
        .info-item.symptom {
            border-left-color: #f44336;
        }
        
        .info-item.treatment {
            border-left-color: #2196F3;
        }
        
        .info-item.prevention {
            border-left-color: #4CAF50;
        }
        
        .info-item.pesticide {
            border-left-color: #ff9800;
        }
        
        .condition-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .impact-box {
            background: #fff3e0;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
            margin: 1rem 0;
        }
        
        /* Smooth scrolling for results */
        html {
            scroll-behavior: smooth;
        }
        
        .scroll-target {
            scroll-margin-top: 2rem;
        }
        
        /* Results metrics enhancement */
        .results-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .metric-box {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            transition: transform 0.2s;
        }
        
        .metric-box:hover {
            transform: translateY(-5px);
        }
        
        .metric-value.large {
            font-size: 1.8rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .analysis-grid {
                grid-template-columns: 1fr;
            }
            
            .results-metrics {
                grid-template-columns: 1fr;
            }
            
            .info-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
    """, unsafe_allow_html=True)

def get_theme():
    """Return theme configuration"""
    return {
        'primaryColor': '#4CAF50',
        'backgroundColor': '#f5f7f9',
        'secondaryBackgroundColor': '#ffffff',
        'textColor': '#31333F',
        'font': 'sans-serif'
    }


def main():
    st.set_page_config(
        page_title="Crop Disease Detection System",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load and apply custom CSS
    apply_custom_css()

    # Add sidebar
    with st.sidebar:
        st.markdown("### About")
        st.info("""
        This system uses advanced AI to detect diseases in:
        - Cashew
        - Cassava
        - Maize
        - Tomato
        """)
        
        st.markdown("### System Info")
        st.code(f"""
        Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}
        Model: ResNet50
        Resolution: 224x224
        """)

    st.markdown("<h1 class='main-title'>🌿 Crop Disease Detection System</h1>", unsafe_allow_html=True)
    
    tabs = st.tabs(["Disease Detection", "Analysis History", "Help"])

    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            
            # Image preprocessing options
            with st.expander("Image Enhancement Options", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    brightness = st.slider("Brightness", 0.0, 2.0, 1.0)
                    contrast = st.slider("Contrast", 0.0, 2.0, 1.0)
                with col_b:
                    sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0)
                    apply_clahe = st.checkbox("Apply CLAHE")
                
                # Apply preprocessing
                processed_image = image
                if brightness != 1.0:
                    processed_image = ImagePreprocessor.adjust_brightness(processed_image, brightness)
                if contrast != 1.0:
                    processed_image = ImagePreprocessor.adjust_contrast(processed_image, contrast)
                if sharpness != 1.0:
                    processed_image = ImagePreprocessor.adjust_sharpness(processed_image, sharpness)
                if apply_clahe:
                    processed_image = ImagePreprocessor.apply_clahe(processed_image)
                else:
                    processed_image = image
                
            st.image(processed_image, caption="Image for Analysis", use_container_width=True)

            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing..."):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    results = analyze_image(processed_image, device)
                    
                    if results:
                        st.markdown("---")
                        display_results(results, processed_image)

    with tabs[1]:
        display_history()

    with tabs[2]:
        display_guide()

def process_image(image, brightness, contrast, sharpness, apply_clahe):
    """Process image with selected enhancements"""
    processed = image
    if brightness != 1.0:
        processed = ImagePreprocessor.adjust_brightness(processed, brightness)
    if contrast != 1.0:
        processed = ImagePreprocessor.adjust_contrast(processed, contrast)
    if sharpness != 1.0:
        processed = ImagePreprocessor.adjust_sharpness(processed, sharpness)
    if apply_clahe:
        processed = ImagePreprocessor.apply_clahe(processed)
    return processed

def analyze_image(image, device):
    """Perform image analysis and return results"""
    label_mapping = {
        'Cashew anthracnose': 0, 'Cashew gumosis': 1, 'Cashew healthy': 2,
        'Cashew leaf miner': 3, 'Cashew red rust': 4, 'Cassava bacterial blight': 5,
        'Cassava brown spot': 6, 'Cassava green mite': 7, 'Cassava healthy': 8,
        'Cassava mosaic': 9, 'Maize fall armyworm': 10, 'Maize grasshoper': 11,
        'Maize healthy': 12, 'Maize leaf beetle': 13, 'Maize leaf blight': 14,
        'Maize leaf spot': 15, 'Maize streak virus': 16, 'Tomato healthy': 17,
        'Tomato leaf blight': 18, 'Tomato leaf curl': 19,
        'Tomato septoria leaf spot': 20, 'Tomato verticulium wilt': 21
    }
    
    model = load_model("best_model.pth", len(label_mapping), device)
    return predict_image(image, model, label_mapping, device) if model else None

def display_results(predictions, image):
    """Display analysis results with comprehensive disease information"""
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.markdown('<div class="results-title">Analysis Results</div>', unsafe_allow_html=True)
    
    # Display primary diagnosis and metrics
    top_disease, top_confidence = predictions[0]
    
    # Display metrics in full width
    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="big-font">{top_disease}</div>
                <div>Primary Diagnosis</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with metrics_cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="big-font">{top_confidence:.1%}</div>
                <div>Confidence</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with metrics_cols[2]:
        severity = "High" if top_confidence > 0.8 else "Medium" if top_confidence > 0.6 else "Low"
        severity_color = "#f44336" if severity == "High" else "#ff9800" if severity == "Medium" else "#4CAF50"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="big-font" style="color: {severity_color}">{severity}</div>
                <div>Risk Level</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display visualization and predictions in a grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="prediction-chart">', unsafe_allow_html=True)
        st.markdown("### Prediction Distribution")
        fig = plot_prediction_chart(predictions)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="alternatives-list">', unsafe_allow_html=True)
        st.markdown("### Alternative Possibilities")
        for disease, conf in predictions[1:]:
            st.markdown(
                f"""
                <div class="alternative-item">
                    <span class="disease-name">{disease}</span>
                    <span class="confidence">{conf:.1%}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display disease information if available
    if top_disease in DISEASE_INFO:
        st.markdown('<div class="disease-info-section">', unsafe_allow_html=True)
        st.markdown("## Detailed Disease Information")
        disease_info = DISEASE_INFO[top_disease]
        
        if 'healthy' in top_disease.lower():
            display_healthy_info(disease_info)
        else:
            display_disease_info(top_disease, disease_info)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Save to history
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        'timestamp': datetime.now(),
        'image': image,
        'predictions': predictions
    })
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_history():
    """Display analysis history"""
    st.markdown("""
    <div class="history-card">
        <div class="card-header">Analysis History</div>
""", unsafe_allow_html=True)
    
    if 'history' not in st.session_state or not st.session_state.history:
        st.info("No analysis history available yet.")
        return
    
    for idx, entry in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Analysis {idx + 1} - {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(entry['image'], use_container_width=True)
            with cols[1]:
                for disease, conf in entry['predictions']:
                    st.markdown(f"""
                        <div class="prediction-item">
                            <span class="disease-name">{disease}</span>
                            <span class="confidence">{conf:.1%}</span>
                        </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_guide():
    """Display user guide information"""
    st.markdown("## How to Use the System")
    
    # Create guide sections
    sections = {
        "📸 Image Capture": [
            "Use well-lit, clear images",
            "Focus on the affected area",
            "Include both healthy and diseased parts",
            "Avoid blurry photos"
        ],
        "🔍 Analysis Process": [
            "Upload your image",
            "Use image enhancement if needed",
            "Click 'Analyze' and wait for results",
            "Review the detailed diagnosis"
        ],
        "📊 Understanding Results": [
            "Check the primary diagnosis",
            "Review confidence scores",
            "Read disease information",
            "Follow treatment recommendations"
        ]
    }
    
    cols = st.columns(len(sections))
    for col, (title, items) in zip(cols, sections.items()):
        with col:
            st.markdown(f"### {title}")
            for item in items:
                st.markdown(f"- {item}")

def display_knowledge_base():
    """Display disease knowledge base"""
    st.markdown("## Disease Knowledge Base")
    
    # Create filters
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_crop = st.selectbox(
            "Select Crop",
            ["All", "Cashew", "Cassava", "Maize", "Tomato"]
        )
    
    # Filter diseases based on selection
    filtered_diseases = {
        name: info for name, info in DISEASE_INFO.items()
        if selected_crop == "All" or selected_crop in name
    }
    
    # Display diseases
    for disease_name, disease_info in filtered_diseases.items():
        with st.expander(disease_name):
            display_disease_info(disease_name, disease_info)

if __name__ == "__main__":
    main()