import pandas as pd
from .fourier_analysis import getSpectum


df = pd.read_csv("./data/charts.csv.zip")

for idx, row in df.iterrows():
    spectrum, rate = getSpectum(df.loc[idx,'url'], "data/temp",df.loc[idx,'title'])

    print(df.loc[idx,'url'])