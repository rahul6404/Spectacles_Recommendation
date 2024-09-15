import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("normalized_file.csv")

ax = df.boxplot(column="chin_angle_1", by="face_shape", figsize=(8, 4))
ax.set_ylabel("angle_1")
ax.set_title("box_plot")

plt.show()