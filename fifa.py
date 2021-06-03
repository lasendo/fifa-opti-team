#######################################################################################
#
#  BIG DATA FUNDAMENTALS | MSc in Data Analytics | Strathclyde University
#
# ASSIGNMENT 1: Data analytics with python
#
# Author:               Iker Lasa Ojanguren
# Student number:       201987654
# Starting date:        16/10/2019
# Ending date:          11/11/2019
# Script number:        1/2
# Data:                 Fifa19 EA sports game
# Data file:            data.csv (4 KB)
# Source:               Kaggle: https://www.kaggle.com/karangadiya/fifa19
#                        >User: karangadiya
#                        >Licence: CC BY-NC-SA 4.0
# Code
#  Structure:           1- Importing
#                       2- Data loading
#                       3- Basic notions
#                       4- Cleaning
#                       5- Descriptive charts
#                          A- Number of players by different categories
#                              I)   Position
#                                     Bars with custom colors
#                              II)  Club
#                              III) Nationality
#                          B- Relations: Special score, Money and Wage
#                       6- PCA for dimensionality reduction
#                          A- Correlation analysis
#                          B- PC determination
#                       7- Method 1: Unsupervised analysis
#                          A- Clustering over PCs
#                              I)   KMeans
#                                    -> 3 Clusters by 2,3 PC dimensions
#                                    -> Figure out what clusters mean:
#                                         + Related with positions
#                                         + Custom cluster colors -> created a function (cluster_coloring)
#                             II)   Agglomerative clustering
#                                    -> Analyze same number of clusters
#                            III)   Dendogram
#                                    -> Analyze different number of clusters
#                                    -> Not used in the report.
#                       8- Method 2: Supervised analysis
#
##########################################################################################
# ------------------------------------------------------------------------------------
# 1- IMPORTING
# ------------------------------------------------------------------------------------
import pandas as pd  # To work with dataframes, read csv, plotting
import numpy as np   # To work with numeric vectors
import random
import re            # To substract strings
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For violin, correlation plots
from sklearn.decomposition import PCA # Principal component analysis. Here used for dimensionality reduction
from mpl_toolkits.mplot3d import Axes3D # For 3d scatterplot visualization
from sklearn import cluster, metrics # For clustering techniques
from sklearn.preprocessing import scale # For normalizing data
from scipy.cluster.hierarchy import dendrogram, linkage # Dendogram for exploring clusteri
# ------------------------------------------------------------------------------------
# 2- DATA LOADING
# ------------------------------------------------------------------------------------
path_fifa = "C:\\Users\\win10\\Documents\\STRATHCLYDE\\3- Big Data Fundamentals\\1- Assignment\\fifa19\\"
name_fifa = "data_fifa.csv"
fifa_original = pd.read_csv(path_fifa+name_fifa)
fifa = fifa_original.copy()
# Indicate path where pdf figures should be created
path_out = "C:\\Users\\win10\\PycharmProjects\\BigDataFundamentls\\Project 1\\Output\\"
# ------------------------------------------------------------------------------------
# 3- BASIC NOTIONS
# ------------------------------------------------------------------------------------
# > Dimensions
#    + 18207 rows   , players
#    +    89 columns, attributes
print(fifa.shape)
# > Columns info
columns = fifa.columns
print(fifa.info())
# > Check null values:
null_columns = {}
if (fifa.isnull().any().any()):
    for col in columns:
        if( fifa[col].isnull().any()):
            null_columns[col]= fifa[col].isnull().sum()
print(null_columns)
# >> Info:
#  + Club has null values (241) Why?
null_club = fifa[fifa.Club.isnull()]
no_price = fifa[fifa.Value == "€0"]
no_cost = fifa[fifa.Wage =="€0"]
print(null_club.Name.tolist()==no_price.Name.tolist())
print(null_club.Name.tolist()==no_cost.Name.tolist())
# ++ They are the only ones with 0€ salary too, which makes sense.
#  + Position score columns have 2085 null value
#     -> Because GKs are not taken into account.
#  + Lots of Column score attributes have 48 null values. Why?
# ------------------------------------------------------------------------------------
# 4- CLEANING
# ------------------------------------------------------------------------------------
# > Clean value column price: Charater M and K to million and thousan euros
def clean_value(value_string):
    if value_string[-1] == "M": # Million
        return (float(value_string[0:-1]) * 10 ** 6)
    elif value_string[-1] == "K": # Thousand
        return (float(value_string[0:-1]) * 10 ** 3)
    else:
        return (float(value_string))
fifa["Value_clean"] = fifa.Value.map(lambda x: clean_value(re.sub(r'€', '', x)))
fifa["Wage_clean"] = fifa.Wage.map(lambda x: clean_value(re.sub(r'€', '', x)))
fifa[fifa.Value_clean > 0][["Name","Value_clean","Wage_clean"]].sort_values(by="Value_clean",ascending=False)
# ------------------------------------------------------------------------------------------
# 5- DESCRIPTIVE CHARTS
# ------------------------------------------------------------------------------------------
## A- Number of players by different categories
# > I) POSITIONS ----------------------------------------------------------------------------------------------
# >> Dataframes with amount of players by each unique position present in the data.
number_players_by_positions = fifa.groupby(["Position"]).Position.count()
# >> Dataframe to dictionary for an easier handling
PP = number_players_by_positions.to_dict()
print("Number of positions: ", len(list(number_players_by_positions)))
# >> Fixed custom colors:
position_colors = {"ST": "#000000","CF": "#F8CECC","RW": "#CCCC00","LW": "#FF66FF", "CAM": "#FF3333", "CM": "#6600CC",
 "RM": "#003300","LM": "#3399FF","CDM": "#666666","RWB": "#00FFFF",
 "LWB": "#FF9933","RB": "#0073E6", "LB": "#FFFF33", "CB": "#990000", "GK": "#66CC00"}
color_bars = []
# >>> Get order of position names for bars
position_ordered=number_players_by_positions.sort_values(ascending=True).index.to_list()
for pos in position_ordered:
    if pos in position_colors.keys(): # Add custom color
        color_bars.append(position_colors[pos])
    else: # If the position is not between the selection add gray color
        color_bars.append("#e1e1e1")
number_players_by_positions.sort_values(ascending=True).plot.barh(color = color_bars) # Add custom color list
plt.title("Number of players by position")
plt.savefig(path_out+"NPlayPosition.pdf")
plt.close()
# > II) CLUBS    ----------------------------------------------------------------------------------------------
club_list = fifa.Club.unique()
number_players_by_clubs = fifa.groupby(["Club"]).Club.count()
print("Number of clubs: ", len(list(number_players_by_clubs)))
number_players_by_clubs.hist(grid=False, bins=len(set(list(number_players_by_clubs))), density=True,rwidth=0.9,color = "gray")
plt.title("Amount of clubs by number of players")
plt.savefig(path_out+"NPlayClubs.pdf")
plt.close()
# > III) NATIONALITY ------------------------------------------------------------------------------------------
number_players_by_nationality = fifa.groupby(["Nationality"]).Nationality.count()
print("Number of Nationalities: ", len(list(number_players_by_nationality)))
number_players_by_nationality[number_players_by_nationality>400].sort_values(ascending=True).plot.barh(color="gray")
plt.title("Number of players by nationality")
plt.savefig(path_out+"NPlayNationality.pdf")
plt.close()
## B- Relations: Special score, Money and Wage
# > Special/price
plt.scatter(fifa.Special, fifa.Value_clean, alpha=0.1)
plt.xlabel("Special score")
plt.ylabel("Value (€)")
plt.title("Special score and Price")
plt.savefig(path_out+"special_VS_price.pdf");#plt.show()
plt.close()
# > Special/wage
plt.scatter(fifa.Special,fifa.Wage_clean,alpha=0.1,c="orange")
plt.xlabel("Special score")
plt.ylabel("Wage (€)")
plt.title("Special score and Wage")
plt.savefig(path_out+"special_VS_wage.pdf");
plt.close()
# > PriceValue/wage
plt.scatter(fifa.Value_clean,fifa.Wage_clean,alpha=0.1,c="orange")
plt.xlabel("Value (€)")
plt.ylabel("Wage (€)")
plt.title("Value and Wage")
plt.savefig(path_out+"value_VS_wage.pdf");
plt.close()
# --------------------------------------------------------------------------------------------------------------------
# 6- PCA for DIMENSIONALITY REDUCTION
#    Reference: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# --------------------------------------------------------------------------------------------------------------------
# A- Correlation analysis
# >> Filter just Numeric columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
column_numerics = fifa.select_dtypes(include=numerics).columns
# Manualy copy the wanted columns from the previous selection, excluding ID, Value_clean, Wage_clean
score_columns = ['Age', 'Overall', 'Potential', 'Special',
       'International Reputation', 'Weak Foot', 'Skill Moves', 'Jersey Number',
       'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
print(len(score_columns))
fifa_scores = fifa[score_columns].dropna(axis=0)
# >> Correlation plot
plt.figure(figsize=(5,4))
corr = fifa_scores.corr()
sns.heatmap(corr,cmap="bwr",vmin=-1,vmax=1)
#plt.show(); plt.close()
plt.savefig(path_out+"corrplot.pdf")
plt.close()# >> Correlation plot
plt.figure(figsize=(5,4))
corr = fifa_scores.corr()
sns.heatmap(corr,cmap="bwr",vmin=-1,vmax=1)
#plt.show(); plt.close()
plt.savefig(path_out+"corrplot.pdf")
plt.close()
# B- PC Determination
# >> Explained variance vs. Number of components
pca = PCA().fit(fifa_scores)
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker=".")
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Explained variance vs. Number of components")
plt.savefig(path_out+"PCAperformance.pdf")
plt.close()
# >> 3 components
pca3 = PCA(n_components=3)
pca3.fit(fifa_scores)
print("Explained variance 1PC, 2PC, 3PC")
print(pca3.explained_variance_ratio_)
# plot data
Xpca3 = pca3.transform(fifa_scores)
X3    = pca3.inverse_transform(Xpca3)
X3=Xpca3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(X3[:, 0], X3[:, 1], X3[:,2], alpha=0.02, marker=".")
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.title("3PC scatterplot")
plt.savefig(path_out+"PCA_3components.pdf")
plt.close()
# >> 2 components
pca2 = PCA(n_components=2)
pca2.fit(fifa_scores)
print(pca2)
print(pca2.explained_variance_ratio_)
# > Plot dimensionality reduction to 2PCs
Xpca2 = pca2.transform(fifa_scores)
X2    = pca2.inverse_transform(Xpca2)
X2=Xpca2
plt.scatter(X2[:, 0], X2[:, 1], alpha=0.02, marker=".")
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title("2PC Scatterplot")
plt.savefig(path_out+"PCA_2components.pdf")
plt.close()
# >> Component meaning
plt.matshow(pca3.components_,cmap="bwr",vmin=-1,vmax=1)
plt.yticks([0,1,2],['1st Comp','2nd Comp', '3rd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(score_columns)),score_columns,rotation=65,ha='left')
plt.tight_layout()
plt.savefig(path_out+"PCA_3correlations.pdf")
plt.close()
# ----------------------------------
# 7- METHOD 1: Unsupervised Analysis
# ----------------------------------
pca_scores_scaled2 = scale(X2)
X2=pca_scores_scaled2
# Plot players in principal plane by position
color_list_players_position = []
filtered_indexes = fifa[score_columns].dropna(axis=0).index.to_list()
list_positions = fifa.ix[filtered_indexes,"Position"].to_list()
Xpos= []
for pos_idx in range(len(list_positions)):
    if list_positions[pos_idx] in position_colors.keys():
        color_list_players_position.append(position_colors[list_positions[pos_idx]])
        Xpos.append(X2[pos_idx])
Xpos=np.array(Xpos)
plt.scatter(Xpos[:, 0], Xpos[:, 1], alpha=0.3, marker=".", c = color_list_players_position)
plt.xlabel('PC1_scaled')
plt.ylabel('PC2_scaled')
plt.title("Players by main Positions")
plt.savefig(path_out+"MainPositions_PCplane.pdf")
plt.close()
# A- Clustering over PCs (Principal components)
## I) KMeans
##   KMeans is more intuitive since the resulting number of clusters is fixed.
##   Using euclidean distance over the PC dimension the clusters centroid define
##   the mean position, which is easy to understand.
# > 2PC KMeans clustering
kmeans = cluster.KMeans(n_clusters=3, n_init=300) # Run with 300 times with different initial centroid sets.
kmeans.fit(pca_scores_scaled2)
# >> Plot clusters data
#     The model assigns labels to the cluster randomly depending on the initialization of centroids.
#      In order to refer to the same clusters we order them by the 2nd PC in ascending order using a custom
#      function.
color_barplot = ["r","g","b"]
color_to_cluster = {"r":0,"g":1,"b":2}
def cluster_coloring(labels, X1, center=[]):
    """
    Give colors to points respect to their cluster label in *labels*. Colors for clusters are given ordered according
    to their centroids X1 coordinate from lowest to highest.
    :param labels: Cluster index of every point. [0,0,1,0,2,...]
    :param X1: One numerical variable for the centroids estimation for the color assignation
    :return: color_list, cluster_color: array with colors for every point, colors for clusters sorted from lowest X1 component
    """
    # > Number of clusters
    Nclus = len(set(labels))
    # > Order of cluster-labels in labels
    cluster_label = []
    for idx in range(len(labels)):
        if len(cluster_label) < Nclus:
            if labels[idx] not in cluster_label:
                cluster_label.append(labels[idx])
        else:
            break
    # > Determine centroids
    centroids = [0]*Nclus
    number_of_points = [0]*Nclus
    # >> Sum component X1 of clusters
    for idx in range(len(labels)):
        centroids[labels[idx]] += X1[idx]
        number_of_points[labels[idx]] += 1
    # >> Divide by number of points incluster for mean
    for clus_idx in range(len(set(labels))):
        centroids[clus_idx] = centroids[clus_idx]/number_of_points[clus_idx]

    colors = ["r","g","b"]
    # >> Labels according to ascending order for the cluster centroid X1 component
    centroid_labels = [ii for ii in range(Nclus)]
    yx = sorted(zip(centroids, centroid_labels))
    cluster_label = [x for y, x in yx]
    # >> Label to color
    label_to_color = {}
    for idx in range(len(colors)):
        label_to_color[cluster_label[idx]] = colors[idx]
    color_list = []
    for label in labels:
        color_list.append(label_to_color[label])
    # >> Cluster to color
    if len(centers)<1: # Ignore center colors
        colors_clust = colors
    else:
        # >> Ranking the numbers, source:
        # https: // stackoverflow.com / questions / 5284646 / rank - items - in -an - array - using - python - numpy - without - sorting - array - twice
        ranking_idx = np.array(center)
        temp = ranking_idx.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(ranking_idx))
        colors_clust = [colors[idx] for idx in ranks]
    return(color_list, colors_clust)

numbers = list(set(kmeans.labels_))
centers = np.array(kmeans.cluster_centers_)
# >>> Calculate colors for each player according to cluster
color_list, cluster_colors = cluster_coloring(kmeans.labels_, X2[:, 1], center=centers[:,1])
# >>> Plot all players using custom colors
plt.scatter(X2[:, 0], X2[:, 1], alpha=0.02, marker=".", c = color_list)
plt.xlabel('PC1_scaled')
plt.ylabel('PC2_scaled')
plt.legend([])
# >>> Plot cluster centroids with the clusters custom color
plt.scatter(centers[:,0], centers[:,1], marker="x", color=cluster_colors)
plt.title("Three clusters for pca scaled 2 components")
plt.savefig(path_out+"CLUSTER_3PCA_2.pdf")
plt.close()
# >> Join clusters to original fifa pandas
#     Take into account indexing of null values that have been filtered for the pca
filtered_index = fifa[["ID"]+score_columns].dropna(axis=0)
filtered_index["Cluster"] = [color_to_cluster[col] for col in color_list]#kmeans.labels_
filtered_index["Cluster"] = filtered_index["Cluster"].apply(np.int64)
filtered_index["PC1"] = X2[:, 0]
filtered_index["PC2"] = X2[:, 1]
fifa_all = pd.merge(fifa, filtered_index[["ID","Cluster","PC1","PC2"]], on='ID', how='outer')
print(fifa_all.info())
cluster_by_position = fifa_all.groupby(["Cluster","Position"]).count()
cluster_by_position.to_csv(path_out+"clusters_3PCA_2.csv")
# >> Number of players by clusters and positions
ordered_positions = number_players_by_positions.sort_values(ascending=True).index.to_list()
position_cluster = fifa_all.groupby(["Position","Cluster"]).size().unstack().reindex(ordered_positions)
position_cluster.plot(kind='barh', stacked=True, color=color_barplot)
plt.title("Number of players by position and cluster (2PCs)")
plt.savefig(path_out+"NPlayPositionCluster3PCA2.pdf")
plt.close()
# > 3PC KMeans clustering
pca_scores_scaled3 = scale(X3)
kmeans = cluster.KMeans(n_clusters=3, n_init=300)
kmeans.fit(pca_scores_scaled3)
X3=pca_scores_scaled3
# >> plot data 3D
numbers = list(set(kmeans.labels_))
centers = np.array(kmeans.cluster_centers_)
# >>> Calculate colors for each player according to cluster
color_list, cluster_colors = cluster_coloring(kmeans.labels_, X3[:, 1], center=centers[:,1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# >>> Plot all players using custom colors
plt.scatter(X3[:, 0], X3[:, 1], X3[:, 2], alpha=0.02, marker=".", c = color_list)
ax.set_xlabel('PC1_scaled')
ax.set_ylabel('PC2_scaled')
ax.set_zlabel('PC3_scaled')
# >>> Plot cluster centroids with the clusters custom color
plt.scatter(centers[:,0], centers[:,1], centers[:,2], marker="x", color=cluster_colors)
plt.title("Three clusters for pca scaled 3 components")
plt.savefig(path_out+"CLUSTERS_3PCA_3.pdf")
plt.close()
# >> Plot data 1PC-2PC plane
# >>> Plot all players using custom colors
plt.scatter(X3[:, 0], X3[:, 1], alpha=0.02, marker=".", c = color_list)
plt.xlabel('PC1_scaled')
plt.ylabel('PC2_scaled')
plt.legend(["Clus0","Clus1","Clus3"])
# >>> Plot cluster centroids with the clusters custom color
plt.scatter(centers[:,0], centers[:,1], marker="x", color=cluster_colors)
plt.title("Three clusters for pca scaled 3 components")
plt.savefig(path_out+"CLUSTER_3PCA_3_Plane.pdf")
plt.close()
# >> Join clusters to original fifa pandas, play with indexing of null values that have been filter for the pca
filtered_index = fifa[["ID"]+score_columns].dropna(axis=0)
filtered_index["Cluster"] = [color_to_cluster[col] for col in color_list] #kmeans.labels
filtered_index["Cluster"] = filtered_index["Cluster"].apply(np.int64)
filtered_index["CP1"] = X2[:, 0]
filtered_index["CP2"] = X2[:, 1]
fifa_all = pd.merge(fifa, filtered_index[["ID","Cluster","CP1","CP2"]], on='ID', how='outer')
print(fifa_all.info())
cluster_by_position = fifa_all.groupby(["Cluster","Position"]).count()
cluster_by_position.to_csv(path_out+"clusters_3PCA_3.csv")
# >> Number of players by clusters and positions
ordered_positions = number_players_by_positions.sort_values(ascending=True).index.to_list()
position_cluster = fifa_all.groupby(["Position","Cluster"]).size().unstack().reindex(ordered_positions)
position_cluster.plot(kind='barh', stacked=True, color=color_barplot)
plt.title("Number of players by position and cluster (3PCs)")
plt.savefig(path_out+"NplayPositionCluster3PCA3.pdf")
plt.close()

## II) Agglomerative clustering
# > Select 3 clusters, from evidence of previous analysis.
#     Explore manually with different *linkage*=["average","ward","complete"]
model = cluster.AgglomerativeClustering(n_clusters=3, linkage="ward", affinity="euclidean")
pca_scores_scaled3 = scale(X3) # Consider X2 or X3 for 2 or 3 PCs, respectivelly
model.fit(pca_scores_scaled3)

numbers = list(set(model.labels_))
color_list, cluster_colors = cluster_coloring(model.labels_, X3[:, 1])
# Plot agglomerative clusters
plt.scatter(X3[:, 0], X3[:, 1], alpha=0.02, marker=".", c = color_list) #model.labels_)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
plt.close()
# >> Join clusters to original fifa pandas, play with indexing of null values that have been filter for the pca
filtered_index = fifa[["ID"]+score_columns].dropna(axis=0)
filtered_index["Cluster"] = model.labels_#[color_to_cluster[col] for col in color_list]
filtered_index["Cluster"] = filtered_index["Cluster"].apply(np.int64)
filtered_index["CP1"] = X3[:, 0]
filtered_index["CP2"] = X3[:, 1]
fifa_all = pd.merge(fifa, filtered_index[["ID","Cluster","CP1","CP2"]], on='ID', how='outer')
print(fifa_all.info())
cluster_by_position = fifa_all.groupby(["Cluster","Position"]).count()
cluster_by_position.to_csv(path_out+"clusters_3PCA_3.csv")
# >> Number of players by clusters and positions
ordered_positions = number_players_by_positions.sort_values(ascending=True).index.to_list()
position_cluster = fifa_all.groupby(["Position","Cluster"]).size().unstack().reindex(ordered_positions)
position_cluster.plot(kind='barh', stacked=True)#, color=color_barplot)
plt.title("Number of players by position and cluster (3PCs)")
plt.savefig(path_out+"NplayPositionCluster4PCA3_agglomerative.pdf")
plt.close()
# III) Dendogram
# Following the analysis and found meaning of the clusters, going further into
# the clustering analysis does not seem necessary.
# Anyway this section is included for completeness, but not used for the report.
"""
mod = "average"
model = linkage(X2, mod) # Using 2PCs, average linkage
plt.figure()
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(model, leaf_rotation=90., leaf_font_size=8.,)
plt.savefig(path_out+"dendogram_"+mod+".pdf")
plt.close()
"""