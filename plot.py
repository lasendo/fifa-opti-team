
# FIFA.py
# INTERESTING GRAPHS

plt.figure(figsize=(20,15))
sns.violinplot(x='Position', y='Special', data=fifa_all)
plt.title("Special score by position")
#plt.show()
plt.close()
fifa_all[["Value_clean","Position"]].plot(kind='box')
plt.savefig(path_out+"special_value_violin.pdf")
plt.close()
plt.figure(figsize=(20,15))
sns.violinplot(x='Position', y='Value_clean', data=fifa_all[fifa_all.Value_clean>0])
plt.title("Special score by value")
plt.savefig(path_out+"special_value_violin.pdf")
plt.close()


# FIFA2.py

# Plotting
def multiple_reg(Ydependent, *Xindependents, constant=False, path_out="", file="regression.txt", title="",
                 xlabel="", ylabel="Dependent variable", plot_attribute=0, show=False):

    assert(plot_attribute < len(Xindependents))
    # Handdle integrity of options:
    if constant: plot_attribute+=1
    if xlabel=="": xlabel=str(plot_attribute+1)+" independent variable"

    XX = []
    for idx in range(len(Ydependent)):
        aux = []
        if constant: aux = [1] # Add one
        for var_idx in range(len(Xindependents)):
            aux.append(Xindependents[var_idx][idx])
        XX.append(aux)
    XX = np.array(XX)

    model = sm.OLS(Ydependent, XX)
    results = model.fit()
    print(results.summary(), file=open(path_out + file+".txt", "w"))

    plt.scatter(XX[:,plot_attribute], Ydependent, c="blue", alpha=0.02)
    Xaxis = np.linspace(min(XX[:,plot_attribute]), max(XX[:,plot_attribute]), 1000)
    Yfit = 0
    for param_idx in range(len(results.params)):
        Xaux = np.linspace(min(XX[:, param_idx]), max(XX[:, param_idx]), 1000)
        Yfit = Yfit + np.dot(results.params[param_idx], Xaux)
    plt.plot(Xaxis, Yfit, c="red")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()
    plt.savefig(path_out+file+".pdf")
    plt.close()
    return(results)

X = fifa_valued.Overall.to_list()
Y = fifa_valued.ln_Value.to_list()
mod1=multiple_reg(Y, X, constant=False, path_out=path_out,file="simple_regression", title="Model for player value prediction",
                 xlabel="Overall",ylabel="ln(Value)",plot_attribute=0)

X2 = fifa_valued.Potential.to_list()
mod2=multiple_reg(Y, X, X2, constant=False, path_out=path_out,file="multiple_regression",
                  title="Model for player value prediction",
                 xlabel="Potential",ylabel="Value",plot_attribute=1)

X3 = fifa_valued.Special.to_list()
mod3=multiple_reg(Y, X, X2, X3, constant=False, path_out=path_out,file="multiple_regression",
                  title="Model for player value prediction",
                 xlabel="Overall",ylabel="Value",plot_attribute=0, show=True)

print("Best models order: ")
print(compare_models(mod1,mod2,mod3))


# FIFA OPTI

# FURTHER STUDY
# Calculate mean scores of ALL clubs by LEAGUE
#      + With default formation
#      + With default *score*=Special

# TODO: fix number of clubs that do not match between fifa and club_league.csv
"""
club_league_pd = pd.read_csv(path_fifa+"club_league.csv")
club_league = dict(zip(club_league_pd.Club, club_league_pd.League))

clubs_known_league = club_league_pd.Club.to_list()
score_club = {}
for cb in clubs_known_league:
    try:
        score_club[cb] = club_score(cb)[0]
    except ValueError as e:
        print(e)

score_club_pd = pd.DataFrame(list(score_club.items()), columns=['club_name', 'mean_score'])
score_club_pd["League"] = list(map(lambda x: club_league[x],score_club_pd.Club))
import seaborn as sns
sns.violinplot(x='League', y='mean_score', data=score_club_pd)
plt.title("Distribution of Mean Special score of clubs by League")
plt.show()
"""


















