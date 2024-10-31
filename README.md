# Breast Cancer Prediction App

Developed a logistic regression model using SciKit-learn to predict whether a cell is benign or malignant. Designed and deployed a custom interface with Streamlit to visualize the predictions, and published the application for broader accessibility.

[Try it out on Streamlit here!](https://breast-cancer-prediction-app-garrison.streamlit.app)

## Data Preparation

### Data Cleaning

The `get_clean_data` function is responsible for loading and cleaning the cancer data. The steps involved are:

1. Load the data from the CSV file.
2. Drop unnecessary columns (`Unnamed: 32` and `id`).
3. Map the `diagnosis` column to binary values (`M` to `1` and `B` to `0`).
4. Drop duplicate rows.
5. Reset the index of the DataFrame.

```python
def get_clean_data():
    # Load data
    data = pd.read_csv(r'data/cancer_data.csv')
    # Drop columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    # Map diagnosis to binary
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    # Drop duplicates
    data = data.drop_duplicates()
    # Reset index
    data = data.reset_index(drop=True)
    
    return data
```

### Model Creation

The `create_model` function is responsible for creating and evaluating a logistic regression model. The steps involved are:

1. Split the data into features (`X`) and target (`y`).
2. Split the data into training and testing sets.
3. Scale the feature data.
4. Train a logistic regression model on the training data.
5. Test the model on the testing data.
6. Print the accuracy and classification report.

```python
def create_model(data):

    # Split data into X and y
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Train model and fit it
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Test model
    y_pred = model.predict(X_test)
    
    # Print results
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Classification Report: \n', classification_report(y_test, y_pred))
    
    return model, scaler
```

### Main Function and Model Export

The `main` function orchestrates the data cleaning, model creation, and saving of the model and scaler. The steps involved are:

1. Run the `get_clean_data` function to clean the data.
2. Run the `create_model` function to create and evaluate the model.
3. Save the trained model as a pickle file.
4. Save the scaler as a pickle file.

```python
def main():
    
    # Run functions
    data = get_clean_data()
    
    model, scaler = create_model(data)
    
    # Save model as a pickle file
    with open('model/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
        
    # Save scaler as a pickle file
    with open('model/scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
```

## Creating the Streamlit App
You can find all of my source code for the streamlit app in the app_main.py file above.

### Sidebar Function

The `add_sidebar` function adds a sidebar to the Streamlit app for adjusting cell nuclei measurements. The steps involved are:

1. Add a header and description to the sidebar.
2. Load the cleaned data.
3. Create slider labels for various measurements.
4. Create sliders for each measurement with min, max, and default values based on the data.
5. Return a dictionary of the input values from the sliders.

```python
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
```

### Scaling Function

The `get_scaled_values` function scales the input values from the sidebar based on the min and max values of the cleaned data. The steps involved are:

1. Load the cleaned data.
2. Drop the `diagnosis` column to get the feature data.
3. For each input value, calculate the scaled value using min-max scaling.
4. Return a dictionary of the scaled values.

```python
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
```

### Radar Chart Function

The `get_radar_chart` function generates a radar chart based on the scaled input data. The steps involved are:

1. Scale the input data using the `get_scaled_values` function.
2. Define the categories for the radar chart.
3. Create a radar chart with three traces: mean values, standard error values, and worst values.
4. Update the layout to set the radial axis range and show the legend.
5. Return the radar chart figure.

```python
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
```

### Prediction Function

The `add_predictions` function loads the trained model and scaler, scales the input data, and makes predictions. The steps involved are:

1. Load the trained model and scaler from pickle files.
2. Convert the input data to a NumPy array and reshape it.
3. Scale the input data using the loaded scaler.
4. Make a prediction using the loaded model.
5. Display the prediction result and probabilities in the Streamlit app.
6. Include a disclaimer about the app's purpose.

```python
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
    st.write('Probability of being Malignant: ', model.predict_proba(input_array_scaled)[0][1])
    
    st.write('This app is for demonstration/education purposes only. Please consult with a medical professional for accurate diagnosis.')
```

### Main Function

The `main` function sets up the Streamlit app, including page configuration, sidebar, and main content. The steps involved are:

1. Set the page configuration with title, icon, layout, and sidebar state.
2. Load and apply custom CSS for styling.
3. Load input data from the sidebar.
4. Create a container for the main content and display the app title and description.
5. Create columns for displaying the radar chart and predictions.
6. Display the radar chart in the first column.
7. Display the predictions in the second column.

```python
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
```

## Custom CSS

The custom CSS styles the Streamlit app, including padding, border, background color, and specific styles for diagnosis labels. The styles are:

1. `.st-emotion-cache-j5r0tf`: Adds padding, border-radius, background color, and border properties.
2. `.diagnosis`: Sets the text color, padding, and border-radius for diagnosis labels.
3. `.diagnosis.benign`: Sets the background color for benign diagnosis labels.
4. `.diagnosis.malignant`: Sets the background color for malignant diagnosis labels.

```css
.st-emotion-cache-j5r0tf {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #16171d;
    border: #f3f3f6;
    border-color: #f3f3f6;
    border-width: 1px;
}

.diagnosis {
    color: #f3f3f6;
    padding: 0.2em 0.5rem;
    border-radius: 0.5em;
}

.diagnosis.benign {
    background-color: #2ecc71;
}

.diagnosis.malignant {
    background-color: #cb685d;
}
```

## Photos of finished Streamlit App
![alt text](screenshot.png)