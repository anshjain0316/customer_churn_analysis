
def predict_churn(customer_data):
    """
    Predict customer churn probability
    
    Parameters:
    customer_data: dict or DataFrame with features: ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure']... (total: 38)
    
    Returns:
    probability: float (0-1) - probability of churn
    prediction: int (0/1) - binary prediction
    """
    import joblib
    import pandas as pd
    
    model = joblib.load('best_churn_model_logistic_regression.pkl')
    
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    
    probability = model.predict_proba(customer_data)[0][1]
    prediction = model.predict(customer_data)[0]
    
    return {
        'churn_probability': probability,
        'will_churn': bool(prediction),
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
    }
