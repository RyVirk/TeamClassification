import csv
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

Handle = open("JoinedStats.csv",encoding='latin-1')

df = pd.read_csv('JoinedStats.csv')
#print(df)

df = df[['TotalTouches','TouchesD3','TouchesM3','TouchesA3', 'G+A-PK']]

df['PercentTouchD3'] = df['TouchesD3'] / df['TotalTouches']
df['PercentTouchM3'] = df['TouchesM3'] / df['TotalTouches']
df['PercentTouchA3'] = df['TouchesA3'] / df['TotalTouches']
df['Sum'] = df['PercentTouchA3'] + df['PercentTouchM3'] + df['PercentTouchD3']

squad = ['Ajax','Atalanta','Atletico Madrid','Barcelona','Basaksehir','Bayern Munich','Chelsea','Club Brugge','Dortmund',
'DynamoKyiv','Ferencvaros','Inter','Juventus','Krasondar','Lazio','Liverpool','Loko Moscow',
'Monchen Gladbach','Manchester City','Manchester Utd','Marseille','Midtjylland','Olympiacos','Paris SG','Porto',
'RB Leipzig','RB Salzburg','Real Madrid', 'Rennes','Sevilla','Shakhtar','Zenit']

#df['squad'] = squad


tf = df[['PercentTouchD3','PercentTouchM3','PercentTouchA3', 'G+A-PK']]

df = df[['PercentTouchD3','PercentTouchM3','PercentTouchA3']]

'''x = df.values
print(x)'''


kmeans = KMeans(n_clusters=4, random_state=100)

kmeans = kmeans.fit(df)

labels = kmeans.predict(df)
# centroid values
centroid = kmeans.cluster_centers_
# cluster values
clusters = kmeans.labels_.tolist()

df['cluster'] = clusters
df['squad'] = squad
tf['cluster'] = clusters
tf['squad'] = squad

tf.to_csv('touches_tableau.csv', index=False)
#print(df.head())

#print(range(len(df)))


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

ax.scatter(df['PercentTouchD3'], df['PercentTouchM3'], df['PercentTouchA3'], c = df['cluster'], marker = 'o', s=100)

ax.set_xlabel('Def3Touches')
ax.set_ylabel('Mid3Touches')
ax.set_zlabel('Att3Touches')
for x, y, z, label in zip(df.PercentTouchD3, df.PercentTouchM3, df.PercentTouchA3, df.squad):
    ax.text(x, y, z, label, fontsize = 'xx-small')
plt.title('3D Scatter of Possession by Third')
plt.show()
