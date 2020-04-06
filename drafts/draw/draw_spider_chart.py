import pandas as pd
from math import pi
import matplotlib.pyplot as plt

#Create a data frame from Messi and Ronaldo's 6 Ultimate Team data points from FIFA 18
Messi = {'Pace':89,'Shooting':90,'Passing':86,'Dribbling':95,'Defending':26,'Physical':61}
Ronaldo = {'Pace':90,'Shooting':93,'Passing':82,'Dribbling':90,'Defending':33,'Physical':80}

data = pd.DataFrame([Messi,Ronaldo], index = ["Messi","Ronaldo"])
print(data)

Attributes =list(data)
AttNo = len(Attributes)



values = data.iloc[1].tolist()
values += values [:1]

angles = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles += angles [:1]



#Find the values and angles for Messi - from the table at the top of the page
values2 = data.iloc[0].tolist()
values2 += values2 [:1]

angles2 = [n / float(AttNo) * 2 * pi for n in range(AttNo)]
angles2 += angles2 [:1]


#Create the chart as before, but with both Ronaldo's and Messi's angles/values
ax = plt.subplot(111, polar=True)

#Add the attribute labels to our axes
plt.xticks(angles[:-1],Attributes)

#Plot the line around the outside of the filled area, using the angles and values calculated before
ax.plot(angles,values)
#Fill in the area plotted in the last line
ax.fill(angles, values, 'teal', alpha=0.1)

ax.plot(angles2,values2)
ax.fill(angles2, values2, 'red', alpha=0.1)

#Rather than use a title, individual text points are added
plt.figtext(0.1,0.9,"Messi",color="red")
plt.figtext(0.1,0.85,"v")
plt.figtext(0.1,0.8,"Ronaldo",color="teal")

#Give the plot a title and show it
ax.set_title("Ronaldo >< Messi")
plt.show()


