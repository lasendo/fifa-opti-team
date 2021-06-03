# fifa-opti-team
Optimize a football team with a given budget

        ___________________________________
       |                 |                 |
       |___              |              ___|
       |_  |             |             |  _|
      .| | |.           ,|.           .| | |.
      || | | )         ( | )         ( | | ||
      '|_| |'           `|'           `| |_|'
       |___|             |             |___|
       |                 |                 |
       |_________________|_________________|

using FIFA data from kaggle: 
Kaggle: https://www.kaggle.com/karangadiya/fifa19
                        >User: karangadiya
                        >Licence: CC BY-NC-SA 4.0
                        
The aim is to create the best team possible for a given amount of available money, the budget.

A MILP knapsap-like model is created in order with ORtools library in order to maximize a "goodness" score of the team. 
This score can be either the "Special" or "General" atttribute of players.

The formation (e.g. 1443) should be given so that positions of the players can be set. Just players for the specific positions chosen are considered.


