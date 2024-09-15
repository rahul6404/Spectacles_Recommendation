import pandas as pd
df = pd.read_csv("insightface_latest_normalized.csv")
df1 = df[df['face_shape'] != 'Round_face_shape']
df1.to_csv('without_round_face_shape.csv', index=False)