import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from fifaOPTI import optimal_team, investment_outcome, club_score
# ------------------------------------------------------------------------------------
# 2- DATA LOADING
# ------------------------------------------------------------------------------------
# TODO: if possible, read data directly from the web
fifa_original = pd.read_csv(Path.cwd() / 'input' / 'data_fifa.csv')
fifa = fifa_original.copy()
# Indicate path where pdf figures should be created
path_out = Path.cwd() / 'output'
path_out.mkdir(exist_ok=True)
path_opt = Path.cwd() / 'Optimization'
path_opt.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------------
# 4- CLEANING
# ------------------------------------------------------------------------------------
# > Clean value column price: Charater M and K to million and thousand euros


def clean_value(value_string):
    if value_string[-1] == "M": # Million
        return float(value_string[0:-1]) * 10 ** 6
    elif value_string[-1] == "K": # Thousand
        return float(value_string[0:-1]) * 10 ** 3
    else:
        return float(value_string)


# Delete euro symbol
fifa["Value_clean"] = fifa.Value.map(lambda x: clean_value(re.sub(u"\u20AC", '', x)))
fifa["Wage_clean"] = fifa.Wage.map(lambda x: clean_value(re.sub(u"\u20AC", '', x)))

# Calculate optimal team with:
#       + *score* as Special/Overall attribute
#       + total_budget
#       + formation
# Example:
# TODO: solve infeasible problem error
print(optimal_team(fifa, total_budget=10**6, goodness_score="Overall", formation="1442"))

# > Plot mean optimal score for different total budgets
# >> With Special
investment_outcome(fifa, num=40, goodness_score="Special", formation="1442", lines={"color": "mediumseagreen", "lty":"-"})
investment_outcome(fifa, num=40, goodness_score="Special", formation="1433", lines={"color": "mediumturquoise", "lty":"-"})
plt.title("Investment vs. Optimal Team's mean Special")
plt.xlabel(u"Total Budget (M\u20AC)")
plt.ylabel("Mean Special")
plt.ylim(1150, 2300)
plt.legend(loc='lower right')
plt.savefig(path_opt / "InvestmentEvolutionSpecial.pdf")
plt.close()
# >> With Overall
investment_outcome(fifa, num=40, goodness_score="Overall", formation="1442",lines={"color": "mediumseagreen", "lty": "-"})
investment_outcome(fifa, num=40, goodness_score="Overall", formation="1433",lines={"color": "mediumturquoise", "lty": "-"})
plt.title("Investment vs. Optimal Team's mean Overall")
plt.xlabel(u"Total Budget (M\u20AC)")
plt.ylabel("Mean Overall")
plt.ylim(50, 90)
plt.legend(loc='lower right')
plt.savefig(path_opt / "InvestmentEvolutionOverall.pdf")
plt.close()
# III) Reference of clubs' scores
#  MEAN SCORES OF TEAMS ARE HELPFUL AS REFERENCES
#     Calculate mean scores of a set of FEW well known clubs:
#        + With default formation
#        + With default *score*=Special
# Example for a small set:
#  > few_club_list = ["FC Barcelona", "Real Madrid", "Manchester City", "Manchester United"]
#  > for cb in few_club_list:
#  > print(cb, club_score(cb))
# -----------------------------------------
# > Calculate mean scores of ALL clubs:
#      + With different formations
#      + With different *scores*={Special, Overall}

club_list = fifa.Club.unique()
special_club1433 = {}
special_club1442 = {}
overall_club1433 = {}
overall_club1442 = {}
for cb in club_list:
    try:
        special_club1433[cb] = club_score(cb, formation="1433", score="Special")[0]
        special_club1442[cb] = club_score(cb, formation="1442", score="Special")[0]
        overall_club1433[cb] = club_score(cb, formation="1433", score="Overall")[0]
        overall_club1442[cb] = club_score(cb, formation="1442", score="Overall")[0]
    except:
        print("There is not such club in the dataset")

special_club_pd = pd.DataFrame(data={"club_list": list(special_club1433.keys()),
                                     "1433":list(special_club1433.values()),
                                     "1442": list(special_club1442.values())})
overall_club_pd = pd.DataFrame(data={"club_list": list(overall_club1433.keys()),
                                     "1433":list(overall_club1433.values()),
                                     "1442": list(overall_club1442.values())})
# > Plot distributions
# >>> Special
special_club_pd.boxplot()
plt.title("Distribution of mean Special for the 651 clubs")
plt.ylim(1150, 2300)
plt.xlabel("Formation")
plt.ylabel("mean Special of team")
plt.savefig(path_opt / "SpecialMeanBoxplot.pdf")
plt.close()
# >>> Overall
overall_club_pd.boxplot()
plt.title("Distribution of mean Overall for the 651 clubs")
plt.xlabel("Formation")
plt.ylabel("mean Overall of team")
plt.ylim(50, 90)
plt.savefig(path_opt / "OverallMeanBoxplot.pdf")
plt.close()