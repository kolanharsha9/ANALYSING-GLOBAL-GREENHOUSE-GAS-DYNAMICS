
#%%
import pandas as pd

dataset1 = pd.read_csv("C:/Users/YASH/Documents/Intro-to-Data-Mining-Project/dataset.csv")

print(dataset1.head)

# %%

columns_to_drop = ['ISO2','CTS_Full_Descriptor','CTS_Code','F1986','F1987','F1988','F1989','F1990','F1991','F1992','F1993','F1994','F1995','F1996','F1997','F1998','F1999','F2000']

dataset1.drop(columns=columns_to_drop,inplace=True)

dataset1.to_csv("dataset.csv", index = False)


# %%
