import pickle
import pandas as pd
import os
predicted_list = []

def predict_face_shape():
    with open('model_2.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    with open('scaler_2.pkl', 'rb') as file:
        scaler = pickle.load(file)

    df = pd.read_csv('insightface_latest.csv', usecols=['chin_angle_1', 'chin_angle_2', 'cheek_bone_angle', 'ratio_1', 'ratio_2',
                                'ratio_3', 'ratio_4','ratio_5','ratio_6','n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12'])
    for index, row in df.iterrows():
        X_scaled = scaler.transform([row.values])
        y_pred = loaded_model.predict(X_scaled)
        predicted_list.append(y_pred[0])
    return predicted_list




