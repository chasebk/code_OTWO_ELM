import matplotlib.pyplot as plt
import matplotlib.lines as lines
from pandas import read_csv
import numpy as np

## https://matplotlib.org/api/markers_api.html

## Paras

x_label = "Time (5 minutes)"


#y_label = "Megabytes"        # EU
#title = "EU dataset"
#read_filepath = "pred_eu_SL5.csv"
#write_filepath = "new/2two_elm_eu_sl2"
#write_filepath = "new/4otwo_elm_eu_sl5"
#point_number = 600
#point_start = 1500

y_label = "Bytes"           # UK
title = "UK dataset"
read_filepath = "pred_uk_SL2.csv"
#write_filepath = "new/4otwo_elm_uk_sl2"
write_filepath = "new/4otwo_elm_uk_sl2"
point_number = 600
point_start = 1200


filetypes = [".png", ".jpg", ".eps", ".pdf"]
colnames = ['True', "GA", "PSO", "TWO", "OTWO", "DE", "ABFO", "more1", "more2"]
results_df = read_csv(read_filepath, header=None,names=colnames, index_col=False, engine='python')

real = results_df['True'].values
flgann = results_df['OTWO'].values
x = np.arange(point_number)

plt.plot(x, real[point_start:point_start + point_number], color='#8c564b', label='Actual')
plt.plot(x, flgann[point_start:point_start + point_number], color='#17becf', linestyle='dashed', label='Predict')

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(title)
plt.legend()
for ft in filetypes:
    plt.savefig(write_filepath + ft, bbox_inches="tight")
plt.show()
