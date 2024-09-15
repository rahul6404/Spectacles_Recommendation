import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
df = pd.read_csv('merged_file_4.csv', index_col=False)

df_heart = df.loc[df["face_shape"] == "Heart_face_shape"]
df_oblong = df.loc[df["face_shape"] == "Oblong_face_shape"]
df_oval = df.loc[df["face_shape"] == "Oval_face_shape"]
df_round = df.loc[df["face_shape"] == "Round_face_shape"]
df_square = df.loc[df["face_shape"] == "Square_face_shape"]

print(df_heart.describe())
print("--------------------------------------------")
print(df_oblong.describe())
print("--------------------------------------------")
print(df_oval.describe())
print("--------------------------------------------")
print(df_round.describe())
print("--------------------------------------------")
print(df_square.describe())



