import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

plt.rc('figure', figsize=(5, 7))

# Fixing random state for reproducibility
np.random.seed(19680801)

# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low))

fig, axs = plt.subplots(2, 3)

# basic plot
axs[0, 0].boxplot(data)
axs[0, 0].set_title('basic plot')

# notched plot
axs[0, 1].boxplot(data, 1)
axs[0, 1].set_title('notched plot')

# change outlier point symbols
axs[0, 2].boxplot(data, 0, 'gD')
axs[0, 2].set_title('change outlier\npoint symbols')

# don't show outlier points
axs[1, 0].boxplot(data, 0, '')
axs[1, 0].set_title("don't show\noutlier points")

# horizontal boxes
axs[1, 1].boxplot(data, 0, 'rs', 0)
axs[1, 1].set_title('horizontal boxes')

# change whisker length
axs[1, 2].boxplot(data, 0, 'rs', 0, 0.75)
axs[1, 2].set_title('change whisker length')

fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                    hspace=0.4, wspace=0.3)


filename = "mape_uk_stability.csv"
usecols = [0, 1, 2, 3, 4, 5]
header = ["MLNN", "ELM", "GA-ELM", "PSO-ELM", "TWO-ELM", "OTWO-ELM"]
df = pd.read_csv(filename, usecols=usecols, header=0, index_col=False)
data = df.values
data =data.transpose()
data = data.transpose()


   
# Multiple box plots on one Axes
fig, ax1 = plt.subplots()
bp = ax1.boxplot(data, 0, 'rD')

## change outline color, fill color and linewidth of the boxes
for box in bp['boxes']:
    # change outline color
    box.set( color='#7570b3', linewidth=2)

    
## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3', linewidth=2)
    
## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3', linewidth=2)
    
## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='black', linewidth=2)


# Hide these grid behind plot objects
ax1.set_axisbelow(True)
ax1.set_title('Internet traffic from EU cities')
ax1.set_ylabel('MAPE')

# Set the axes ranges and axes labels
xtickNames = plt.setp(ax1, xticklabels=header)
plt.setp(xtickNames, rotation=45, fontsize=8)



plt.savefig("hehe.png", bbox_inches = "tight")

plt.show()
