import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder



#Part 1: Data Loading and Preprocessing
def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    df['brand'] = df['CarName'].apply(lambda x: x.split()[0].lower())
    corrections = {
        'vokswagen': 'volkswagen',
        'maxda': 'mazda',
        'porcshce': 'porsche', 
        'vw': 'volkswagen',
        'toyouta': 'toyota'
        
    }
    df['brand'] = df['brand'].replace(corrections)
    
    numerical_features = [f for f in df.select_dtypes(include=['int64', 'float64']).columns if f not in ['car_ID', 'price']]
    categorical_features = [f for f in df.columns if f not in numerical_features + ['car_ID', 'price', 'CarName']]
    
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    X = df[numerical_features + categorical_features]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders, numerical_features + categorical_features





#Part 2: Linear Regression Model from Scratch
def train_linear_regression(X_train, y_train, feature_names, n_iterations=1500, learning_rate=0.01):
    weights = np.zeros(X_train.shape[1])
    bias = 0
    
    for _ in range(n_iterations):
        y_pred = np.dot(X_train, weights) + bias
        dw = (1/len(X_train)) * np.dot(X_train.T, (y_pred - y_train))
        db = (1/len(X_train)) * np.sum(y_pred - y_train)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    
    return weights, bias, feature_names




#Part 3: Weights of different features
def plot_feature_importance(feature_names, weights):
    importance = pd.DataFrame({'Feature': feature_names, 'Weight': weights}) \
                  .sort_values('Weight', key=abs, ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(importance['Feature'], importance['Weight'])
    plt.title('Feature Importance in Price Prediction')
    plt.xlabel('Weight Magnitude')
    plt.tight_layout()
    plt.show()





#Part 4: Sample car price prediction
def predict_sample_car(sample_car_data, feature_names, weights, bias, scaler, label_encoders):
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    if 'CarName' in sample_car_data:
        brand = sample_car_data['CarName'].split()[0].lower()

        corrections = {
            'vokswagen': 'volkswagen',
            'maxda': 'mazda',
            'porcshce': 'porsche', 
            'vw': 'volkswagen',
            'toyouta': 'toyota'
        }
        brand = corrections.get(brand, brand)
        if 'brand' in label_encoders:
            le = label_encoders['brand']
            input_df['brand'] = le.transform([brand])[0] if brand in le.classes_ else 0

    for feature in feature_names:
        if feature in sample_car_data and feature != 'brand':
            input_df[feature] = sample_car_data[feature]
    
    for col in label_encoders:
        if col in input_df.columns and col != 'brand':
            le = label_encoders[col]
            val = str(input_df[col].iloc[0])
            input_df[col] = le.transform([val])[0] if val in le.classes_ else 0

    scaled_data = scaler.transform(input_df)
    return (np.dot(scaled_data, weights) + bias)[0]





#Execution
X_train, X_test, y_train, y_test, scaler, label_encoders, feature_names = load_and_preprocess_data('CarPrice_Assignment.csv')
weights, bias, _ = train_linear_regression(X_train, y_train, feature_names)

sample_car = {
    'CarName': 'toyota camry',
    'symboling': 1,
    'fueltype': 'gas',
    'aspiration': 'std',
    'doornumber': 'four',
    'carbody': 'sedan',
    'drivewheel': 'fwd',
    'enginelocation': 'front',
    'wheelbase': 95.5,
    'carlength': 175.6,
    'carwidth': 66.2,
    'carheight': 54.9, 
    'curbweight': 2550,
    'enginetype': 'ohc', 
    'cylindernumber': 'four', 
    'enginesize': 125,
    'fuelsystem': 'mpfi', 
    'boreratio': 3.35, 
    'stroke': 3.4,
    'compressionratio': 9.0, 
    'horsepower': 115, 
    'peakrpm': 5500,
    'citympg': 29, 
    'highwaympg': 36
}

predicted_price = predict_sample_car(sample_car, feature_names, weights, bias, scaler, label_encoders)
print(f"Predicted price: ${predicted_price:,.2f}")

plot_feature_importance(feature_names, weights)