from utils.IOUtil import read_dataset_file, save_formatted_data_csv
import numpy as np

filename = "internet_traffic_eu_5m.csv"
pathsave = "test.csv"

dataset = read_dataset_file(filename, usecols=[1], header=0)

t1 = dataset[:, 0:1] / 8
t2 = dataset[:, 0:1] / (8*1024)
t3 = dataset[:, 0:1] / (8 * 1024 * 1024)

done = np.concatenate((dataset, t1, t2, t3), axis=1)

save_formatted_data_csv(done, pathsave, "")






