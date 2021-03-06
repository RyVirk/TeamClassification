import csv
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#removed unwanted columns and joined 4 datasets seen in folder separately outside of main file

#try with raw data without filtering, and select files(attacking/defensive) as well, binary classification, try KNN neighbors too

'''
JOINED STATS TABLE START
'''

Handle = open("JoinedStats.csv",encoding='latin-1')

df = pd.read_csv('JoinedStats.csv')
#print(df)

df = df[['Poss','G+A-PK','npxG+xA','TotalTouches','TouchesD3','TouchesM3','TouchesA3','TklWon','TklD3','TklM3','TklA3',
'TotalPressures','SuccessfulPressures','Press%','PressD3','PressM3','PressA3','PassesCompleted','PassesAttempted','PassCompletion%']]

from sklearn import preprocessing

x = df.values
#print(x)

scaler = preprocessing.MinMaxScaler()

x_scaled = scaler.fit_transform(x)

X_norm = pd.DataFrame(x_scaled)
#print(X_norm)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
reduced = pd.DataFrame(pca.fit_transform(X_norm))

squad = ['Ajax','Atalanta','Atletico Madrid','Barcelona','Basaksehir','Bayern Munich','Chelsea','Club Brugge','Dortmund',
'DynamoKyiv','Ferencvaros','Inter','Juventus','Krasondar','Lazio','Liverpool','Loko Moscow',
'Monchen Gladbach','Manchester City','Manchester Utd','Marseille','Midtjylland','Olympiacos','Paris SG','Porto',
'RB Leipzig','RB Salzburg','Real Madrid', 'Rennes','Sevilla','Shakhtar','Zenit']

#reduced['Squad'] = squad

#print(reduced)


kmeans = KMeans(n_clusters=5, random_state=100)

kmeans = kmeans.fit(reduced)


labels = kmeans.predict(reduced)
# centroid values
centroid = kmeans.cluster_centers_
# cluster values
clusters = kmeans.labels_.tolist()

reduced['cluster'] = clusters
#print(clusters)
reduced['squad'] = squad
reduced.columns = ['x', 'y', 'cluster', 'squad']
#print(reduced)
reduced.head()


#SENDING DATAFRAME TO CSV FOR TABLEAU VISUALS  
reduced.to_csv('JoinedForTableau.csv' , index = False)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="dark")
ax = sns.lmplot(x="x", y="y", hue='cluster', data = reduced, legend=False,
fit_reg=False, height = 15, scatter_kws={"s": 250})
texts = []
for x, y, s in zip(reduced.x, reduced.y, reduced.squad):
    texts.append(plt.text(x, y, s))
ax.set(ylim=(-2, 2))
plt.tick_params(labelsize=15)
plt.xlabel("PC 1", fontsize = 20)
plt.ylabel("PC 2", fontsize = 20)
plt.show()

'''
JOINED STATS TABLE END
'''



'''
ATTACKING START
'''

Handle = open("JoinedStats.csv",encoding='latin-1')

df = pd.read_csv('JoinedStats.csv')
#print(df)

df = df[['Poss','G+A-PK','npxG+xA','TotalTouches','TouchesD3','TouchesM3','TouchesA3','PassesCompleted','PassesAttempted','PassCompletion%']]

from sklearn import preprocessing

x = df.values
#print(x)

scaler = preprocessing.MinMaxScaler()

x_scaled = scaler.fit_transform(x)

X_norm = pd.DataFrame(x_scaled)
#print(X_norm)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
reduced = pd.DataFrame(pca.fit_transform(X_norm))

squad = ['Ajax','Atalanta','Atletico Madrid','Barcelona','Basaksehir','Bayern Munich','Chelsea','Club Brugge','Dortmund',
'DynamoKyiv','Ferencvaros','Inter','Juventus','Krasondar','Lazio','Liverpool','Loko Moscow',
'Monchen Gladbach','Manchester City','Manchester Utd','Marseille','Midtjylland','Olympiacos','Paris SG','Porto',
'RB Leipzig','RB Salzburg','Real Madrid', 'Rennes','Sevilla','Shakhtar','Zenit']

#reduced['Squad'] = squad

#print(reduced)


kmeans = KMeans(n_clusters=5, random_state=100)

kmeans = kmeans.fit(reduced)


labels = kmeans.predict(reduced)
# centroid values
centroid = kmeans.cluster_centers_
# cluster values
clusters = kmeans.labels_.tolist()

reduced['cluster'] = clusters
#print(clusters)
reduced['squad'] = squad
reduced.columns = ['x', 'y', 'cluster', 'squad']
#print(reduced)
reduced.head()


#SENDING DATAFRAME TO CSV FOR TABLEAU VISUALS  
#reduced.to_csv('AttackingForTableau.csv' , index = False)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="dark")
ax = sns.lmplot(x="x", y="y", hue='cluster', data = reduced, legend=False,
fit_reg=False, height = 15, scatter_kws={"s": 250})
texts = []
for x, y, s in zip(reduced.x, reduced.y, reduced.squad):
    texts.append(plt.text(x, y, s))
ax.set(ylim=(-2, 2))
plt.tick_params(labelsize=15)
plt.xlabel("PC 1", fontsize = 20)
plt.ylabel("PC 2", fontsize = 20)
plt.show()

'''
ATTACKING END
'''

'''
DEFENDING START
'''

Handle = open("JoinedStats.csv",encoding='latin-1')

df = pd.read_csv('JoinedStats.csv')
#print(df)

df = df[['TklWon','TklD3','TklM3','TklA3','TotalPressures','SuccessfulPressures','Press%','PressD3','PressM3','PressA3']]

from sklearn import preprocessing

x = df.values
#print(x)

scaler = preprocessing.MinMaxScaler()

x_scaled = scaler.fit_transform(x)

X_norm = pd.DataFrame(x_scaled)
#print(X_norm)

from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
reduced = pd.DataFrame(pca.fit_transform(X_norm))

squad = ['Ajax','Atalanta','Atletico Madrid','Barcelona','Basaksehir','Bayern Munich','Chelsea','Club Brugge','Dortmund',
'DynamoKyiv','Ferencvaros','Inter','Juventus','Krasondar','Lazio','Liverpool','Loko Moscow',
'Monchen Gladbach','Manchester City','Manchester Utd','Marseille','Midtjylland','Olympiacos','Paris SG','Porto',
'RB Leipzig','RB Salzburg','Real Madrid', 'Rennes','Sevilla','Shakhtar','Zenit']

#reduced['Squad'] = squad

#print(reduced)


kmeans = KMeans(n_clusters=5, random_state=100)

kmeans = kmeans.fit(reduced)


labels = kmeans.predict(reduced)
# centroid values
centroid = kmeans.cluster_centers_
# cluster values
clusters = kmeans.labels_.tolist()

reduced['cluster'] = clusters
#print(clusters)
reduced['squad'] = squad
reduced.columns = ['x', 'y', 'cluster', 'squad']
#print(reduced)
reduced.head()


#SENDING DATAFRAME TO CSV FOR TABLEAU VISUALS  
#reduced.to_csv('DefendingForTableau.csv' , index = False)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="dark")
ax = sns.lmplot(x="x", y="y", hue='cluster', data = reduced, legend=False,
fit_reg=False, height = 15, scatter_kws={"s": 250})
texts = []
for x, y, s in zip(reduced.x, reduced.y, reduced.squad):
    texts.append(plt.text(x, y, s))
ax.set(ylim=(-2, 2))
plt.tick_params(labelsize=15)
plt.xlabel("PC 1", fontsize = 20)
plt.ylabel("PC 2", fontsize = 20)
plt.show()

'''
DEFENDING END
'''
















