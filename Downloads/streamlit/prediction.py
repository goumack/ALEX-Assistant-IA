# prediction.py

def predict_new_observation(model, observation, mean, std):
    standardized_obs = (observation - mean) / std
    prediction = model.predict([standardized_obs])
    return prediction
