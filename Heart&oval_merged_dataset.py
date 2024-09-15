import pandas as pd
df = pd.read_csv('insightface_latest_normalized.csv')
df['face_shape'] = df['face_shape'].replace(['Heart_face_shape', 'Oval_face_shape'], 'Heart_or_Oval')
df.to_csv('Merged_Heart_&_Oval.csv', index=False)
