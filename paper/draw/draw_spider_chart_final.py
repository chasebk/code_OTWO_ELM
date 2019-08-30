import pandas as pd
from math import pi
import matplotlib.pyplot as plt


colors = ["teal", "red", "green", "blue", "orange", "yellow"]
titles = ["MLNN","ELM", "GA-ELM", "PSO-ELM", "TWO-ELM", "OTWO-ELM"]

#Create a data frame from Messi and Ronaldo's 6 Ultimate Team data points from FIFA 18
k2 = {'MLNN':16.716,'ELM':16.292,'GA-ELM':15.378,'PSO-ELM':15.394,'TWO-ELM':15.400,'OTWO-ELM':15.363}
k5 = {'MLNN':15.857,'ELM':15.744,'GA-ELM':14.914,'PSO-ELM':14.834, 'TWO-ELM':14.885,'OTWO-ELM':14.874}


data = pd.DataFrame([k2, k5], index = ["k=2", "k=5"])
print(data)

Attributes =list(data)
AttNo = len(Attributes)


values = data.iloc[0].tolist()
values += values [:1]
angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles += angles [:1]

values1 = data.iloc[1].tolist()
values1 += values1 [:1]
angles1 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles1 += angles1 [:1]


#Create the chart as before, but with both Ronaldo's and Messi's angles/values
ax = plt.subplot(111, polar=True)

#Add the attribute labels to our axes
plt.xticks(angles[:-1], Attributes)



#Plot the line around the outside of the filled area, using the angles and values calculated before
ax.plot(angles,values)
ax.fill(angles, values, 'teal', alpha=0.1)

ax.plot(angles1,values1)
ax.fill(angles1, values1, 'red', alpha=0.1)

# ax.plot(angles2,values2)
# ax.fill(angles2, values2, 'green', alpha=0.1)
#
# ax.plot(angles3,values3)
# ax.fill(angles3, values3, 'blue', alpha=0.1)
#
# ax.plot(angles4,values4)
# ax.fill(angles4, values4, 'orange', alpha=0.1)
#
# ax.plot(angles5,values5)
# ax.fill(angles5, values5, 'yellow', alpha=0.1)




#Give the plot a title and show it
ax.set_title("Sliding window k = 2")

plt.savefig("hehe.png", bbox_inches="tight")
plt.show()


