# TeamClassification
## Champions League Teams 2020/2021 Classification
- Classifying Team Play Styles From The UCL Season
- Utilizing Python(Pandas, Matplotlib,SciKitLearn,Numpy)

Dataset Originally From fbref.com

### Implementation Insights
#### 2D Analysis
- The plan was to analyze different playstyles from the current Champions League Season using Kmeans clustering
* After pulling the necessary statistics from different tables on fbref.com,
 we combined the most crucial statistics into JoinedStats.csv.
    * We believed these to be the best predictors of playstyle from the dataset that we found.

- Using sklearn.decomposition, we boiled down all of the statistics in JoinedStats.csv
into two stats, an x and y coordinate.
- Then we used Kmeans to cluster the plotted pointes and matplotlib to plot them.
- The results can also be seen in the attatched Tableau file.

#### 3D Third of the Field Analysis

- The second plan of analysis was to analyze playstyle based on which third of the pitch
each team had the most touches in.
- The same style of analysis was implemented, except we used a 3D scatter plot with each
Third of the field being an axis in the plot.
- In order to view this plot, you must run the 3dtouches.py file.
- We also analyzed how each cluster shown in this plot affected goals scored vs goals conceded, which can be found in the tableau file.
- Determined that teams that had the most possession in the attacking and mid thirds scored the most, and conceded the least.
