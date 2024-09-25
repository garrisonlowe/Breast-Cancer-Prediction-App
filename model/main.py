import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

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


# Create model
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

    
if __name__ == '__main__':
    main()