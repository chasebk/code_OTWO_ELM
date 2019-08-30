pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#C03028',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

# Pandas for managing datasets
import pandas as pd

# Matplotlib for additional customization
from matplotlib import pyplot as plt
plt.rc('figure', figsize=(4.5, 5))

# Seaborn for plotting and styling
import seaborn as sns

# Read dataset
df = pd.read_csv('Pokemon.csv', index_col=0)

x = range(1, 16)
filename = "mape_eu_stability.csv"
usecols = [0, 1, 2, 3, 4, 5]
header = ["MLNN", "ELM", "GA-ELM", "PSO-ELM", "TWO-ELM", "OTWO-ELM"]
df = pd.read_csv(filename, usecols=usecols, header=0, index_col=False)


# Display first 5 observations
df.head()


# Recommended way
#sns.lmplot(x='Attack', y='Defense', data=df)
 
# Alternative way
# sns.lmplot(x=df.Attack, y=df.Defense)

#g = sns.violinplot(data=df, inner=None, color=".8")
# Swarm plot with Pokemon color palette
g = sns.swarmplot(data=df, palette=pkmn_type_colors)
#g.set_xticklabels(rotation=30)

              
# Hide these grid behind plot objects
plt.title('EU cities')
plt.ylabel('MAPE')
plt.xticks(rotation=30)

plt.savefig("eu_hihi.eps", bbox_inches = "tight")

plt.show()


#https://elitedatascience.com/python-seaborn-tutorial
#https://seaborn.pydata.org/generated/seaborn.stripplot.html
#https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.violinplot.html
#https://www.google.com/search?q=plotting+a+series+of+stacked+histograms&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiZhZCjpdviAhXR-GEKHUaGAMgQ_AUIECgB&biw=1920&bih=976#imgrc=NUtjPVjJ-Vj25M:
#https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn-factorplot
#https://seaborn.pydata.org/generated/seaborn.violinplot.html
#https://www.google.com/search?q=Labeling+violin+in+seaborn+with+median+value&tbm=isch&source=univ&sa=X&ved=2ahUKEwi1i8bmo9viAhVGa94KHSXNB_gQsAR6BAgIEAE#imgrc=_
#https://seaborn.pydata.org/generated/seaborn.swarmplot.html













