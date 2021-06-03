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
# Script number:        +Extra (3/2)
# Data:                 Fifa19 EA sports game
# Data file:            data.csv (4 KB)
# Source:               Kaggle: https://www.kaggle.com/karangadiya/fifa19
#                        >User: karangadiya
#                        >Licence: CC BY-NC-SA 4.0
# Code
#  Structure:           1- Importing
#                       2- Data loading
#                       4- Cleaning
#                       9- Method 3: MILP Optimization
#                          I) Implement model of optimal team (optimal_team function)
#                         II) Investment outcome
#                             + Plot mean optimal score for different total budgets
#                        III) Reference of clubs' scores
#                             + Plot distributions
##########################################################################################
# ------------------------------------------------------------------------------------
# 1- IMPORTING
# ------------------------------------------------------------------------------------
import re
import statistics

import numpy as np
from ortools.linear_solver import pywraplp # optimization module
import pandas as pd


# -------------------------------------------------------------------------------
# MIXED INTEGER LINEAR PROGRAMMING (MILP) Optimization
# -------------------------------------------------------------------------------
# I) Implement model of optimal team (optimal_team function)


def optimal_team(total_budget = 10000, goodness_score="Special", formation = "1433"):
    """
    Function to calculate the OPTIMAL TEAM given a limited total budget

    X -> player selection variable
         =0 if player is not in optimal team
         =1 if player is in optimal team

    OBJECTIVE ->  sum(S_{ii,pp}*x_{ii,pp}) all pp in position and ii in available_players_in_position_pp

        where S is a score that tells how good a player is. We take this to be the Score, since
        we found that is the one correlated with plenty other attributes and takes most of the
        information as PCA has shown.

    RESTRICTIONS ->

        + Limited budget:
            sum(P_{ii,pp}*x_{ii,pp}) all pp in position and ii in available_players_in_position_pp  <= total_budget

        + Just 11 players, limited amount by position
            for all positions pp:
             sum( x_{ii,pp}  all ii in available_players_in_position_pp )  == fixed_amount[pp]

        e.g.:
                sum( x_{ii,'GK'} all ii in available_GK_players) == 1,

                sum( x_{ii,'CB'} all ii in available_CB_players) == 2


    :param total_budget: the total money to purchase players
    :param formation: The selected player positioning system on soccer pitch

    :return [team_score, spent_budget, remaining_budget]

            team_score: dictionary with position key and name, price and used "goodness" score (Special)
            spent_budget: sum of the cost of all selected players
            remaining_budget: total_budget - spent_budget
    """

    # > Define solver 'COIN-Branch and Cut'
    solver = pywraplp.Solver('FIFA', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    # > SET PARAMETERS according to FORMATION:
    #          + position of players,
    #          + amount of players per position
    formations = ["1433", "1442"]
    if formation == "1433":
        posi = ["GK", "CB", "LB", "RB", "CM", "ST"]
        posi_fixed = {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CM": 3, "ST": 3}
    elif formation == "1442":
        posi = ["GK", "CB", "LB", "RB", "CM", "LM", "RM", "ST"]
        posi_fixed = {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CM": 2, "LM": 1, "RM": 1, "ST": 2}
    elif formation[0] == "-":
        posi = [formation[1:]]
        posi_fixed = {formation[1:]: 1}
    else:
        raise ValueError("Please enter a valid formation in ", formations)

    posi_players = {}
    for position in posi:
        # TODO: clarify what is PP
        if position in PP.keys():
            posi_players[position] = PP[position]

    max_players = max(posi_players.values())

    # > DATA to DICTionaries since they are easy to access
    def clean_value(value_string):
        if value_string[-1] == "M":
            return float(value_string[0:-1]) * 10 ** 6
        elif value_string[-1] == "K":
            return float(value_string[0:-1]) * 10 ** 3
        else:
            return float(value_string)

    # Make square data:
    #  The varible for player selection must be a square matrix.
    #  We take the matrix to have a size of the number of positions in posi and
    #  the maximum number of available players among different possitions. The extra
    #   player indexes should not be taken into account. To avoid this we just give them
    #   negative scores and a non affordable value. The optimization will not take these
    #  players into account, sice they are not in the feasible set.

    # Two functions fill these values:
    def cancell_score(vec, max_val):
        aux = vec
        while len(aux) < max_val:
            aux.append(-100.0)
        return aux

    def cancell_price(vec, max_val):
        aux = vec
        while len(aux) < max_val:
            aux.append(2*total_budget)
        return aux

    # > Record the data to dictionaries
    names = {}
    scores = {}
    price = {}
    for position in posi:
        fifa_position_df = fifa[fifa.Position == position]
        names[position] = fifa_position_df.Name.to_list()
        scores[position] = cancell_score(fifa_position_df[goodness_score].to_list(), max_players)
        aux_value = fifa_position_df.Value.to_list()
        price[position] = cancell_price(list(map(lambda x: clean_value(x.replace("€", "")), aux_value)), max_players)

    # > Some players' value is 0€, to neglect those we change the prize to be
    #    non-affordable
    no_free_players = True
    if no_free_players:
        for pos_idx in range(len(posi)):
            for player_idx in range(max_players):
                if price[posi[pos_idx]][player_idx]==0:
                    price[posi[pos_idx]][player_idx] = total_budget + 1

    # > DEFINE our player selection variables
    x = {}
    for pos_idx in range(len(posi)):
        for player_idx in range(max_players):
            x[pos_idx, player_idx] = solver.BoolVar('x[%ii,%ii]' % (pos_idx, player_idx))

    # > Objective Function:
    #     Our objective is to maximize the teams players' skill score.
    solver.Maximize(solver.Sum([scores[posi[ii]][jj] * x[ii, jj] for ii in range(len(posi))
                                for jj in range(posi_players[posi[ii]])]))

    # > RESTRICTIONS
    # >> Limited budget
    solver.Add((total_budget - solver.Sum([price[posi[ii]][jj] * x[ii, jj]
                                           for ii in range(len(posi)) for jj in range(max_players)])) >= 0.0)
    # >> Amount of players in each position
    for ii in range(len(posi)):
        solver.Add(solver.Sum([x[ii, jj] for jj in range(posi_players[posi[ii]])]) == posi_fixed[posi[ii]])

    # > SOLVE
    sol = solver.Solve()
    # >> Check feasibility
    if sol == solver.INFEASIBLE:
        print("The problem is not feasible, there is no solution.")
        raise EnvironmentError("The problem is not feasible")
    elif sol == solver.OPTIMAL:
        print("An OPTIMAL TEAM has been found for a BUDGET of %i€" %total_budget)
        print("")

        # >>> We get if the players are included in the optimal team of 11
        xsol = {}
        for ii in range(len(posi)):
            for jj in range(max_players):
                xsol[ii, jj] = x[ii, jj].solution_value()

        team_score = {}
        spent_budget = 0
        for ii in range(len(posi)):
            count = 0
            for jj in range(max_players):
                if xsol[ii, jj] == 1:
                    count += 1
                    team_score[posi[ii] + str(count)]={"name": names[posi[ii]][jj],"price": price[posi[ii]][jj], "score": scores[posi[ii]][jj]}
                    spent_budget += price[posi[ii]][jj]

        remaining_budget = (total_budget - spent_budget)

    return team_score, spent_budget, remaining_budget

def mean_team_score(team_score, attribute = "score"):
    """
    This function calculates the teams [mean,mean-1sd,mean+1sd] for an input attribute given the output dictonary *team_score*
    from the function *optimal_team*

    :param team_score: output dictionary from *optimal_team* output
    :param attribute: attribute from the dictionary that wants to be meaned
    :return: list [mean,mean-1sd,mean+1sd] of the selected attribute
    """
    value_list = []
    for key in team_score.keys():
        value_list.append(team_score[key][attribute])
    return statistics.mean(value_list), statistics.stdev(value_list)


def club_score(club, formation="1433", score="Special"):
    """
    Funtion to calculate the mean and standard deviation (sd) of the selected *score* of the team of
    a given specific club present in the dataset.

    e.g.   FC Barcelona, formation: 1433
        ----------------------------------------------------------
          "GK" -> 1 GK player with the highest *score*
          "CB" -> 2 CB players with the highest *score*
          "LB" -> 1 LB player with the highest *score*
          "RB" -> 1 single RB player with the highest *score*
          "CM" -> 2 CM players with the highest *score*
          "ST" -> 3 ST players with the highest *score*


    :param club: The club to calculate the mean score
    :param formation: The kind of formation to use. This affects the positions considered.
    :return: LIST [mean,mean-sd,mean+sd,sd]
    """

    # The club is not in the dataset
    if club not in fifa.Club.unique():
        raise ValueError("There is not such club(",club,") in the dataset")

    # The score selected is not a column of the dataset
    if score not in fifa.columns:
        raise ValueError("There is not such a score-column in the dataset")

    # Define position parameters
    formations = ["1433", "1442"]
    if formation == "1433":
        posi = ["GK", "CB", "LB", "RB", "CM", "ST"]
        posi_fixed = {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CM": 3, "ST": 3}
    elif formation == "1442":
        posi = ["GK", "CB", "LB", "RB", "CM", "LM", "RM", "ST"]
        posi_fixed = {"GK": 1, "CB": 2, "LB": 1, "RB": 1, "CM": 2, "LM": 1, "RM": 1, "ST": 2}
    else:
        raise ValueError("Please enter a valid formation in ", formations)

    # Filter the club
    club = fifa[fifa.Club == club]

    # Scores of the best 11 formation
    scores_team = []
    for position in posi:
        # > For each position take the needed amount of best players according to the selected score
        aux = club[club.Position == position].sort_values(by=score, ascending=False).head(posi_fixed[position])[
            score].to_list()
        # > Add scores to list
        scores_team = scores_team + aux

    mean = statistics.mean(scores_team)
    sd = statistics.stdev(scores_team)

    return mean, mean - sd, mean + sd, sd


def investment_outcome(start=275000, end=10**8, num=5, goodness_score="Special",
                       formation = "1442", lines={"color":"green","lty":"-"}):
    """
    Function to visualize how mean optimal score behaves for increasing total_budget

    :param start: minimum total_budget
    :param end: maximum total_budget
    :param num: number of optimal teams to calculate, points in plot
    :param formation: The selected player positioning system on soccer pitch
    :return: shows plot of mean and sd of score for the different total_budgets
    """

    # > Divide the budget axis with two different densities
    x_budget1 = np.linspace(start=start,stop=end/3, num= num//2)
    x_budget2 = np.linspace(start=end/3, stop=end, num=num//2+1)
    x_budget = np.array(list(x_budget1)+list(x_budget2)[1:])
    # > Calculate the optimal [mean_score, score_stdev] for all Total Budgets
    investemt_performance = list(map(lambda x: mean_team_score(optimal_team(total_budget=x,
                                                                            goodness_score=goodness_score,
                                                                            formation=formation)[0]), x_budget))
    # > Take mean and stdev to arrays
    investemt_performance_mean = [investemt_performance[jj][0] for jj in range(len(investemt_performance))]
    investemt_performance_stdev = [investemt_performance[jj][1] for jj in range(len(investemt_performance))]

    # > Plot errorbar using mean and stdev
    x_budget = np.array([budget / 10 ** 6 for budget in list(x_budget)])
    plt.errorbar(x_budget,
                 investemt_performance_mean,
                 yerr=investemt_performance_stdev,
                 c=lines["color"],
                 linestyle=lines["lty"],
                 label=(formation+"_formation"),
                 barsabove=True,  # include bars
                 marker="s",  # square marker
                 alpha=0.5)  # add transparency





