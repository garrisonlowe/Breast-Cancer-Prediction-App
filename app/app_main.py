import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# Load and clean data
def get_clean_data():
    # Load data
    data = pd.read_csv(r'data\cancer_data.csv')
    # Drop columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    # Map diagnosis to binary
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    # Drop duplicates
    data = data.drop_duplicates()
    # Reset index
    data = data.reset_index(drop=True)
    
    return data

# Add sidebar
def add_sidebar():
    st.sidebar.header('Cell Neuclei Measurements')
    st.sidebar.write('Please adjust the measurements of the tissue samples to see the prediction results.')
    
    # Load data
    data = get_clean_data()
    
    # Create slider labels
    slider_labels = [
        ('Radius (Mean)', 'radius_mean'),
        ('Texture (Mean)', 'texture_mean'),
        ('Perimeter (Mean)', 'perimeter_mean'),
        ('Area (Mean)', 'area_mean'),
        ('Smoothness (Mean)', 'smoothness_mean'),
        ('Compactness (Mean)', 'compactness_mean'),
        ('Concavity (Mean)', 'concavity_mean'),
        ('Concave Points (Mean)', 'concave points_mean'),
        ('Symmetry (Mean)', 'symmetry_mean'),
        ('Fractal Dimension (Mean)', 'fractal_dimension_mean'),
        ('Radius (Standard Error)', 'radius_se'),
        ('Texture (Standard Error)', 'texture_se'),
        ('Perimeter (Standard Error)', 'perimeter_se'),
        ('Area (Standard Error)', 'area_se'),
        ('Smoothness (Standard Error)', 'smoothness_se'),
        ('Compactness (Standard Error)', 'compactness_se'),
        ('Concavity (Standard Error)', 'concavity_se'),
        ('Concave Points (Standard Error)', 'concave points_se'),
        ('Symmetry (Standard Error)', 'symmetry_se'),
        ('Fractal Dimension (Standard Error)', 'fractal_dimension_se'),
        ('Radius (Worst)', 'radius_worst'),
        ('Texture (Worst)', 'texture_worst'),
        ('Perimeter (Worst)', 'perimeter_worst'),
        ('Area (Worst)', 'area_worst'),
        ('Smoothness (Worst)', 'smoothness_worst'),
        ('Compactness (Worst)', 'compactness_worst'),
        ('Concavity (Worst)', 'concavity_worst'),
        ('Concave Points (Worst)', 'concave points_worst'),
        ('Symmetry (Worst)', 'symmetry_worst'),
        ('Fractal Dimension (Worst)', 'fractal_dimension_worst')
    ]

    input_dict = {}

    # Create sliders
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
                            label, # label
                            float(data[key].min()), # min value
                            float(data[key].max()), # max value
                            float(data[key].mean()), # default value
                            key=key # key
                        )

    return input_dict

# Scale values
def get_scaled_values(input_dict):
    data = get_clean_data()
    
    X = data.drop('diagnosis', axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        # Get min and max values
        max_value = X[key].max()
        min_value = X[key].min()
        # Scale value
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value
    
    return scaled_dict

# Create radar chart
def get_radar_chart(input_data):
    
    input_data = get_scaled_values(input_data)
    
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points', 'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'],
            input_data['texture_mean'],
            input_data['perimeter_mean'],
            input_data['area_mean'],
            input_data['smoothness_mean'],
            input_data['compactness_mean'],
            input_data['concavity_mean'],
            input_data['concave points_mean'],
            input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'],
            input_data['texture_se'],
            input_data['perimeter_se'],
            input_data['area_se'],
            input_data['smoothness_se'],
            input_data['compactness_se'],
            input_data['concavity_se'],
            input_data['concave points_se'],
            input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'],
            input_data['texture_worst'],
            input_data['perimeter_worst'],
            input_data['area_worst'],
            input_data['smoothness_worst'],
            input_data['compactness_worst'],
            input_data['concavity_worst'],
            input_data['concave points_worst'],
            input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']     
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

# Adding the predictions
def add_predictions(input_data):
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Cell Cluster Prediction")
    st.write('The model predicts that the cell cluster is: ')
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)
    
    
    st.write('Probability of being Benign: ', model.predict_proba(input_array_scaled)[0][0])
    st.write('Probability of being Malignant: ', model.predict_proba(input_array_scaled)[0][0])
    
    st.write('This app is for demonstration/education purposes only. Please consult with a medical professional for accurate diagnosis.')
    

# Main function
def main():
    # Set page configuration
    st.set_page_config(
        page_title='Breast Cancer Prediction App', 
        page_icon=':female-doctor:',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    
    with open("assets/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Loading data for sidebar
    input_data = add_sidebar()
    
    # Create containers for the sidebar and main content
    with st.container():
        st.title('Breast Cancer Prediction App')
        st.write('Please connect this app to your cytology lab to help you predict if a patient has breast cancer from your tissue samples. This app uses a trained logistic regression model to make predictions. You can update the measurements of the tissue samples in the sidebar sliders to see the prediction results.')
    
    # Create columns
    col1, col2 = st.columns([4, 1])
    
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        
    with col2:
        add_predictions(input_data)
        
        
        



if __name__ == '__main__':
    main()