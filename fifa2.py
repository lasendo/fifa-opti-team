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
# Script number:        2/2
# Data:                 Fifa19 EA sports game
# Data file:            data.csv (4 KB)
# Source:               Kaggle: https://www.kaggle.com/karangadiya/fifa19
#                        >User: karangadiya
#                        >Licence: CC BY-NC-SA 4.0
# Code
#  Structure:           1- Importing
#                       2- Data loading
#                       4- Cleaning
#                       8- Method 2: Supervised analysis --------- 2/2
##                          A- Regression
###                             I)   Variables to include
###                                     + Correlations for filtering
###                                     + Scattermatrix
###                            II)   Explore regression models
#                                       + AIC of many models
#                                       + Select model
#                                       + Diagnostics with residuals
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
from scipy.cluster.hierarchy import dendrogram, linkage # Dendogram for exploring clustering
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
from statsmodels.formula.api import ols # In order to define the model with strings
import statsmodels.stats.api as sms
import math
from statsmodels.sandbox.regression.predstd import wls_prediction_std
# ------------------------------------------------------------------------------------
# 2- DATA LOADING
# ------------------------------------------------------------------------------------
path_fifa = "C:\\Users\\win10\\Documents\\STRATHCLYDE\\3- Big Data Fundamentals\\1- Assignment\\fifa19\\"
name_fifa = "data_fifa.csv"
fifa_original = pd.read_csv(path_fifa+name_fifa)
fifa = fifa_original.copy()
# Indicate path where pdf figures should be created
path_out = "C:\\Users\\win10\\PycharmProjects\\BigDataFundamentls\\Project 1\\Output\\"
path_reg = path_out+"Regression\\"
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
# ------------------------------------------------------------------------------------
#  ... (script 1/2)
# ------------------------------------------------------------------------------------
# ----------------------------------
# 8- METHOD 2: Supervised Analysis
# ----------------------------------
# I) Variables to include
# >> Take just numerical columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
column_numerics = fifa.select_dtypes(include=numerics).columns[2:]
# >> Filter players with a price value associated (€>0)
fifa_valued = fifa[fifa.Value_clean>0]
fifa_valued = fifa_valued[column_numerics].copy()
# >> From previous scatterplots there is evidence of an exponential relation
fifa_valued["ln_Value"]= np.log(fifa_valued.Value_clean)
# >> Calculate correlation for Value and its natural logarithm
corr = fifa_valued.corr().loc[["Value_clean", "ln_Value"],:]
# >>> Take just high correlations
correlated_attributes = corr.loc[:, abs(corr.loc['ln_Value']) > 0.6].columns
corr = fifa_valued[correlated_attributes].corr()
sns.heatmap(corr,cmap="bwr",vmin=-1,vmax=1)
plt.savefig(path_reg+"correlated_heatmap.pdf")
plt.close()
fifa_valued_lab = fifa_valued[correlated_attributes].copy()
fifa_valued_lab.columns = [col[0] for col in fifa_valued_lab.columns]
scatter_matrix(fifa_valued_lab, alpha=0.02, color="mediumturquoise")
plt.savefig(path_reg+"scatter_matrix.png")
plt.close()

# II) Explore regression models
# > Custom functions
def create_models(Yindep_str,variables_list,data,file="reggression_models",without_intercept=False):
    import itertools
    combinations = []
    stuff = variables_list
    for L in range(1, len(stuff) + 1):
        for subset in itertools.combinations(stuff, L):
            combinations.append(list(subset))

    models_str = []
    models = []
    f = open(file+".txt","w")
    for subset in combinations:
        mod_str = Yindep_str + " ~ "+subset[0]
        if len(subset)>1:
            for var_idx in range(1, len(subset)):
                mod_str+=" + "+subset[var_idx]

        # > Apply the *mod_str* model
        models_str.append(mod_str)
        mod = ols(mod_str, data=data).fit()
        models.append(mod)
        print(mod_str, file=f)
        print(mod.summary(), file=f)

        # > Apply the *mod_str* model without the constant
        if without_intercept:
            mod_str_no_const = mod_str + " -1"
            models_str.append(mod_str_no_const)
            mod_no_constant=ols(mod_str_no_const,data=data).fit()
            models.append(mod_no_constant)
            print(mod_str_no_const,file=f)
            print(mod_no_constant.summary(),file=f)
    f.close()
    return(models_str,models)


def compare_models(models, model_names,file,title=""):
    # > Take Akaike's Information Criterion
    AIC_models = np.array([mod.aic for mod in models])
    # > Rank the models
    ranking = AIC_models.argsort()

    # > Barplot of the models
    # >> Colors for number of independent variables
    colors = ["green","blue","orange","black","red"]
    colors = ["mediumseagreen", "mediumturquoise", "sandybrown", "gray", "darksalmon"]
    number_variables = [len(model_leg)-1 for model_leg in model_names]
    number_variables = [colors[aux-1] for aux in number_variables]
    # >> Create DF to plot
    aic_df = pd.DataFrame({"names":model_names,"AIC":AIC_models,"Nvar":number_variables})
    # >>> Filter models by AIC
    aic_df = aic_df[aic_df.AIC < 30000].copy()
    # >>> Plot models with color by their number of independent variavbles
    aic_df.plot.barh(x='names', y='AIC', rot=0,
                     color=[colors[len(model_leg)-2] for model_leg in aic_df.names.to_list()],legend=False)
    plt.xlabel("AIC")
    plt.title(title)
    plt.savefig(file+".pdf")
    plt.close()
    return(ranking)


# > Create all the models
variables = ['Overall', 'Potential', 'Special', 'Reactions', 'Composure']
Mstr, Models = create_models("ln_Value",variables ,fifa_valued,file=path_reg+"Value_regressions")

# > Short names to the initial character
Short_model_legend = []
for mod_str in Mstr:
    if "-1" in mod_str:
        legend_mod = "-"
    else:
        legend_mod = "+"
    for var in variables:
        if var in mod_str:
            legend_mod+=var[0]
    Short_model_legend.append(legend_mod)

# > Create AIC comparison graph
print(compare_models(Models,Short_model_legend,file=path_reg+"Value_regressions",
                     title="ln(Value) Least Squares Regression AIC"))

# > Analise selected model, diagnostics
# >> Create again the selected model
#      Simple regression: ln(Value) = B1*Overall+B2*Potential
value_model = ols("ln_Value ~ Overall + Potential", data=fifa_valued).fit()
print(value_model.summary())
pred_val = value_model.fittedvalues.copy()
true_val = fifa_valued["ln_Value"].values.copy()
residual = true_val - pred_val
plt.rcParams["figure.figsize"] = (6,2)

# >> Residuals vs. fitted
plt.scatter(pred_val, residual, alpha=0.02)
plt.title("(a) Residuals vs. fitted")
plt.ylabel("Residual")
plt.xlabel("Fitted value",labelpad=-30)
plt.savefig(path_reg+"ResidualFitted.pdf")
plt.close()
#print(sms.linear_harvey_collier(value_model))

# >> QQ-Plot
fig, ax = plt.subplots(figsize=(6,2))
sp.stats.probplot(residual, plot=ax, fit=True)
ax.get_lines()[0].set_markerfacecolor('#1e76b5')
ax.get_lines()[0].set_markeredgecolor('#1d6da5')
ax.get_lines()[0].set_alpha(0.02)
ax.get_lines()[1].set_color("orangered")
plt.title("(b) QQ plot")
plt.ylabel("Sample Quantiles")
plt.xlabel("Theoretical Quantiles",labelpad=-30)
plt.savefig(path_reg+"ResidualNormality.pdf")
plt.close()






