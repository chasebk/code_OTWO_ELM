# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
filename = "mape_eu_stability.csv"
usecols = [0, 1, 2, 3, 4, 5]
header = ["MLNN", "ELM", "GA-ELM", "PSO-ELM", "TWO-ELM", "OTWO-ELM"]
df = pd.read_csv(filename, usecols=usecols, header=0, index_col=False)

x = range(1, 16)

# multiple line plot
plt.plot( x, df["MLNN"], marker='o', markerfacecolor='blue', markersize=6, color='skyblue', linewidth=2)
plt.plot( x, df["ELM"], marker='', color='olive', linewidth=2)
plt.plot( x, df["GA-ELM"], marker='', color='orange', linewidth=2, linestyle='dashed')
plt.plot( x, df["PSO-ELM"], marker='^', color='black', linewidth=2, linestyle='dashed')
plt.plot( x, df["TWO-ELM"], marker='+', color='gray', linewidth=2, linestyle='dashed')
plt.plot( x, df["OTWO-ELM"], marker='*', color='red', linewidth=2, linestyle='dashed', label="OTWO-MLNN")
plt.legend()
plt.show()
