import os

import pickle

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

from scipy.stats import poisson

from collections import Counter


## Read in the datasets
# Player statistics data
player_path = open(f'{os.path.dirname(os.getcwd())}\\data\\Players\\cleaned_player_stats.pk', 'rb')
player_stats = pickle.load(player_path)

# gameweek, results statistics data and current PL standings
results_path = open(f'{os.path.dirname(os.getcwd())}\\data\\Results\\results_stats.pk', 'rb')
result_stats = pickle.load(results_path)

pl_table = pd.read_csv(f'{os.path.dirname(os.getcwd())}\\data\\Table\\table.csv')
gameweeks = pd.read_csv(f'{os.path.dirname(os.getcwd())}\\data\\Fixtures\\fixtures.csv')
mapper = pickle.load(open(f'{os.path.dirname(os.getcwd())}\\data\\Maps\\Team_maps.pickle', 'rb'))
        
def df_to_upper(df):
    return df.apply(lambda x: x.astype(str).str.upper())

gameweeks = df_to_upper(gameweeks).replace(mapper)
gameweeks['gameweek'] = gameweeks['gameweek'].astype(int)
gameweek = min(gameweeks['gameweek'])

# Apply linear regression on the full player list
# Need to apply data imputing so that all player data is of equal size - use auto encoding combined with nn maybe?
# Simple linear regression on player value vs points
# Add weights based on injuries or absence = 0
# Uses linear regression to approximate the form of a player from his points
# If not enough game data we need to use stats from last season - need to play with the numbers
def pred_points_weeks(player_stats, gameweek):
    points_prediction_dict = {}
    for k,v in player_stats.items():
        if player_stats[k]['stats'] is None:
            points_prediction_dict[k] = 0
        else: # need to modify to function during games
            if gameweek < 6:
                base_prediction = {k:90*v['details']['ls_ppm'] for k,v in player_stats.items()}
                points =  np.array([base_prediction[k]] * (7 - gameweek) + list(player_stats[k]['stats']['Points']))
                gameweeks = np.array(range(len(points))).reshape(-1, 1)
            else:
                points = np.array(player_stats[k]['stats']['Points'])
                gameweeks =  np.array(range(len(points))).reshape(-1, 1)
            reg1 = LinearRegression().fit(gameweeks,points)
            points_prediction = reg1.predict(int(max(gameweeks))+1)
            if player_stats[k]['details']['fitness'] == 75:
                points_prediction = float(points_prediction) * 0.75
            elif player_stats[k]['details']['fitness'] == 50:
                points_prediction = float(points_prediction) * 0.50
            elif player_stats[k]['details']['fitness'] == 25:
                points_prediction = float(points_prediction) * 0.25
            elif player_stats[k]['details']['fitness'] != 100:
                points_prediction = 0
                
            points_prediction_dict[k] = float(points_prediction)
        
    return points_prediction_dict

def drop_unfit(player_stats):
    return {k:v for k,v in player_stats.items() if v['details']['fitness'] == 100}

# no data on them
def drop_new_transfers(player_stats):
    return {k:v for k,v in player_stats.items() if v['details']['ls_games_played'] is not None}

# Function to remove suspended players
def drop_suspended(player_stats):
    return None

def drop_low_games(player_stats):
    new_dict = {}
    for k,v in player_stats.items():
        try:
            if float(v['details']['ls_games_played']) >= 15:
                new_dict[k] = v
        except TypeError:
            pass
    return new_dict

player_stats = drop_unfit(player_stats)
player_stats = drop_low_games(player_stats)

match_odds = {}
# Add odds and goals stats to the table - these are just approximate and need to be updated weekly
for match in range(len(gameweeks['gameweek'])):
    match_odds[match] = {}
    # get the home and away team
    home_team = gameweeks.iloc[match]['home_team']
    away_team = gameweeks.iloc[match]['away_team']
    
    # No data for promoted teams so copy a team who finished low last season - Need to modify this for when season kicks off
    if home_team not in result_stats.keys():
        home_team = 'BHA'
    if away_team not in result_stats.keys():
        away_team = 'BHA'
    # get the overall averages for win, loss, draw, goals scored, goals conceded
    average_win_rate = sum([v['win_rate_home'] for k,v in result_stats.items()]) / len(result_stats)
    average_draw_rate = sum([v['draw_rate_home'] for k,v in result_stats.items()]) / len(result_stats)
    average_lose_rate = sum([v['lose_rate_home'] for k,v in result_stats.items()]) / len(result_stats)
    average_gs_home = sum([v['goals_scored_home'] for k,v in result_stats.items()]) / len(result_stats)
    average_gc_home = sum([v['goals_conceded_home'] for k,v in result_stats.items()]) / len(result_stats)
    average_gs_away = sum([v['goals_scored_away'] for k,v in result_stats.items()]) / len(result_stats)
    average_gc_away = sum([v['goals_conceded_away'] for k,v in result_stats.items()]) / len(result_stats)

    # get the necessary statistics from the results_stats dict
    home_win_odds = result_stats[home_team]['win_rate_home']
    away_win_odds = result_stats[away_team]['win_rate_away']
    home_lose_odds = result_stats[home_team]['lose_rate_home']
    away_lose_odds = result_stats[away_team]['lose_rate_away']
    home_draw_odds = result_stats[home_team]['draw_rate_home']
    away_draw_odds = result_stats[away_team]['draw_rate_away']
    gs_home = result_stats[home_team]['goals_scored_home']
    gc_home = result_stats[home_team]['goals_conceded_home']
    gs_away = result_stats[away_team]['goals_scored_away']
    gc_away = result_stats[away_team]['goals_conceded_away']

    # generate home/away strengths and compare to try and approximate overall match odds
    home_attack_strength =  gs_home / average_gs_home
    home_defence_strength =  gc_home / average_gc_home
    away_attack_strength =  gs_away / average_gs_away
    away_defence_strength = gc_away / average_gc_away

    predicted_home_goals = home_attack_strength * away_defence_strength * average_gs_home
    predicted_away_goals = away_attack_strength * home_defence_strength * average_gs_away

    # Use Poisson distribution to predict the odds of each team scoring 1-5 goals and generate a N x M matrix
    probability_matrix = []
    for h_goals in range(0,6):
        probability_matrix.append([])
        for a_goals in range(0,6): 
            score_prob = poisson.pmf(k=h_goals, mu=predicted_home_goals) * poisson.pmf(k=a_goals, mu=predicted_away_goals)
            probability_matrix[h_goals].append(score_prob)

    df_prob = pd.DataFrame(probability_matrix, index = range(0,6), columns = range(0,6))

    # calculate probability of clean sheets, goals, results
    away_clean_sheet = sum(df_prob.iloc[0,:])
    home_clean_sheet = sum(df_prob.iloc[:,0])
    away_win = sum(df_prob.values[np.triu_indices(df_prob.values.shape[0], 1)])
    home_win = sum(df_prob.values[np.tril_indices(df_prob.values.shape[0], -1)])
    draw = sum(np.diag(df_prob))

    # Add data to the odds dictionary
    match_odds[match]['home_team'] = gameweeks.iloc[match]['home_team']
    match_odds[match]['away_team'] = gameweeks.iloc[match]['away_team']
    match_odds[match]['score_odds'] = df_prob
    match_odds[match]['home_cs'] = home_clean_sheet
    match_odds[match]['away_cs'] = away_clean_sheet
    match_odds[match]['home_win'] = home_win
    match_odds[match]['away_win'] = away_win
    match_odds[match]['draw'] = draw

def temp_func(player_stats, gameweek):   

    # Generate team specific player stats - goals, assits, cleansheets, mins played
    team_stats = {}
    for i, team in enumerate(gameweeks['home_team'].unique()):
        team_players = [k for k,v in player_stats.items() if v['details']['club'] == team]
        team_stats[team] = {}
        for i, player in enumerate(team_players):
            team_stats[team][player] = {}
            team_stats[team][player]['position'] = player_stats[player]['details']['position']
            team_stats[team][player]['avg_mins'] = player_stats[player]['details']['MinutesPlayed'] / 38
            team_stats[team][player]['avg_gs'] = player_stats[player]['details']['GoalsScored'] / 38
            team_stats[team][player]['avg_assists'] = player_stats[player]['details']['Assists'] / 38
            team_stats[team][player]['avg_cs'] = player_stats[player]['details']['CleanSheets'] / 38
    
        # Generate team averages for each position
        def team_avg(position):
            avg_mins = np.mean([v['avg_mins'] for k,v in team_stats[team].items() if player_stats[k]['details']['position'] == position])
            avg_cs = np.mean([v['avg_cs'] for k,v in team_stats[team].items() if player_stats[k]['details']['position'] == position])
            avg_assists = np.mean([v['avg_assists'] for k,v in team_stats[team].items() if player_stats[k]['details']['position'] == position])
            avg_gs = np.mean([v['avg_gs'] for k,v in team_stats[team].items() if player_stats[k]['details']['position'] == position])
            return {'avg_mins':avg_mins, 'avg_cs':avg_cs, 'avg_assists':avg_assists, 'avg_gs':avg_gs}
        
        avg_gk = team_avg('Goalkeeper')
        avg_def = team_avg('Defender')
        avg_mid = team_avg('Midfielder')
        avg_for = team_avg('Forward')
    
        team_stats[team]['goalkeeper_avgs'] = avg_gk
        team_stats[team]['defender_avgs'] = avg_def
        team_stats[team]['midfielder_avgs'] = avg_mid
        team_stats[team]['forward_avgs'] = avg_for
    
    points_prediction_dict = pred_points_weeks(player_stats, gameweek)
    
    # Using fantasy points try and approximate the points per position/player
    def points_predictor(team, week):
    
        _team = [k for k,v in player_stats.items() if v['details']['club'] == week[team]]
        predicted_points_dict = {}
        
        for player in _team:
            position = team_stats[week[team]][player]['position']
            avg_assists = team_stats[week[team]][player]['avg_assists']
            avg_gs = team_stats[week[team]][player]['avg_gs'] 
    
            # Hard coded values are Fantasy football points system
            min_score = (team_stats[week[team]][player]['avg_mins'] / 60) * 2
            if min_score > 2.0:
                min_score = 2.0      
    
            # testing on def first - try not to hard code
            def points_per_score(pos, team):
                points_prob_matrix = []
                for i in week['score_odds'].columns:
                    points_prob_matrix.append([])
                    for j in week['score_odds'].index:
                        prob = week['score_odds'].iloc[i,j]            
    
                        def cs_gs():
                            if j == 0:
                                cs_points = 4
                                gc_points = 0
                            elif j > 2:
                                cs_points = 0
                                gc_points = -1
                            else:
                                cs_points = 0
                                gc_points = 0
                            
                            return cs_points + gc_points
                        
                        if pos == 'Goalkeeper':
                            save_points = 0 # need to come back to this and add points for saves and pens for all below
                            pen_save_points = 0
                            points_prob_matrix[i].append((prob, save_points + pen_save_points + cs_gs()))
                        
                        elif pos == 'Defender':
                            cs_gs()
                            gs_points = avg_gs * j * 6
                            assist_points = avg_assists * j * 3
                            points_prob_matrix[i].append((prob, gs_points + assist_points + cs_gs()))
                            
                        elif pos == 'Midfielder':
                            if j == 0:
                                cs_points = 1
                            else:
                                cs_points = 0
                            gs_points = avg_gs * j * 5
                            assist_points = avg_assists * j * 3
                            points_prob_matrix[i].append((prob, gs_points + assist_points + cs_points))
                            
                        elif pos == 'Forward':
                            gs_points = avg_gs * j * 4
                            assist_points = avg_assists * j * 3
                            points_prob_matrix[i].append((prob, gs_points + assist_points))
                     
                # Need to aggregate point probabilities due to rounding
                points_prob_df = pd.DataFrame(points_prob_matrix, index = range(0,6), columns = range(0,6))
                points_prob_dict = {}
                
                for i in range(len(points_prob_df)):
                    points = np.mean([y for x,y in points_prob_df.iloc[:,i]])
                    if points in points_prob_dict.keys():
                        points_prob_dict[points] = points_prob_dict[points] + sum([x for x,y in points_prob_df.iloc[:,i]])
                    else:
                        points_prob_dict[points] = sum([x for x,y in points_prob_df.iloc[:,i]])
                predicted_points = max(points_prob_dict, key = points_prob_dict.get)              
                
                return predicted_points
                
            points = points_prediction_dict[player]
            if points < 0.0:
                predicted_points = 0.0
            else:
                predicted_points = (points_per_score(position, week[team]) + points) / 2
            predicted_points_dict[player] = predicted_points
    
        return predicted_points_dict
    
    # =============================================================================
    #             sum([y for x,y in points_prob_array]) # Check this later - probability doesn't quite add up to one - may be rounding error
    # ============================================================================= 
    
    player_points_dict = {}
    
    # This section needs new logic - it struggles to account for double weeks    
    # Use a combination of match odds and player stats to get a more accurate reading on their predicted points
    for i in gameweeks['gameweek'].unique():
        player_points_dict[i] = {}
        gameweek_x = gameweeks[gameweeks['gameweek'] == i]
    
        # Loop through for every team
        for team,v in result_stats.items():
            # Check the next gameweek and see if there are any double games
            team_gameweek = pd.concat([gameweek_x[gameweek_x['home_team'] == team], gameweek_x[gameweek_x['away_team'] == team]])
            
            if len(team_gameweek) == 1:
                ix = int(team_gameweek.index.values[0])
                if team_gameweek.iloc[0]['home_team'] == team:
                    player_points_dict[i] = dict(player_points_dict[i], **points_predictor('home_team', match_odds[ix]))
                else:
                    player_points_dict[i] = dict(player_points_dict[i], **points_predictor('away_team', match_odds[ix]))
            elif len(team_gameweek) == 2:
                print(team, i)
                ix = int(team_gameweek.index.values[0])
                ix2= int(team_gameweek.index.values[1])
                
                if team_gameweek.iloc[0]['home_team'] == team:
                    match1 = points_predictor('home_team', match_odds[ix])
                    if team_gameweek.iloc[1]['home_team'] == team:
                        match2 = points_predictor('home_team', match_odds[ix2])
                    else:
                        match2 = points_predictor('away_team', match_odds[ix2])
                else:
                    match1 = points_predictor('away_team', match_odds[ix])
                    if team_gameweek.iloc[1]['home_team'] == team:
                        match2 = points_predictor('home_team', match_odds[ix2])
                    else:
                        match2 = points_predictor('away_team', match_odds[ix2])
                
                player_points_dict[i] = dict(player_points_dict[i], **{k: match1.get(k, 0) + match2.get(k, 0) for k in set(match1) & set(match2)})  
            else:
                predicted_points_dict = {}
                players = [k for k,v in player_stats.items() if v['details']['club'] == team]
                for player in players:
                    predicted_points_dict[player] = 0.0
                   
                player_points_dict[i] = dict(player_points_dict[i], **predicted_points_dict)

    # intitiate vals
    avg_pred_points = {}
    for week,val in player_points_dict.items():
        for k,v in val.items():
            avg_pred_points[k] = 0
    #aggregate vals       
    for week,val in player_points_dict.items():
        for k,v in val.items():
            avg_pred_points[k] += v
    
    player_value_prices ={}
    for k,v in player_stats.items():
        try:
            player_value_prices[k] = {}
            player_value_prices[k]['price'] = float(v['details']['Price'].split('£')[1])
            player_value_prices[k]['value'] = avg_pred_points[k]/6
        except:
            player_value_prices[k]['value'] = 0
            pass
        
    return player_value_prices

# Chooses the optimal items based on the items' weight and value. There are 2
# Constratints; W represents the total allowed maximum weight and capacity of
# the items and C represents the exact number of items we want returned
# i.e. The optimal n items from the item list
# Function not working 100% but is close enough for good results
# Easier to implement post-filter logic than to add extra rules to knapsack

def xknapsack(capacity, nbitems, maxitems, weight, value):
    #ks_matrix = np.zeros((capacity + 1, nbitems, maxitems + 1))
    ks_matrix = [[[[0,[]] for k in range(maxitems + 1)] for j in range(nbitems)] for i in range(capacity + 1)] 
    for j in range(0, nbitems - 1):
        for i in range(1, capacity + 1):
            for k in range(1, maxitems + 1):
                if weight[j] > i: # If adding this object would exceed the current weight limit (i), append the last score to the matrix
                    ks_matrix[i][j+1][k] = ks_matrix[i][j][k]
                else:
                    if j < k: # If the item number is smaller than the current item limit (k), take the maximum value between the value of
                              # the last matrix position and the value of adding this item instead
                        temp = {'1': ks_matrix[i][j][k][0], '2':ks_matrix[i-weight[j]][j][k][0] + value[j]}
                        ks_matrix[i][j+1][k][0] =  max(temp.items())[1]
                        if max(temp.items())[0] == '1':
                            ks_matrix[i][j+1][k][1] = ks_matrix[i][j][k][1]
                        else:
                            ks_matrix[i][j+1][k][1] = ks_matrix[i-weight[j]][j][k][1] + [j]
                            
                    else: # If the item number is exceeded then calculate the values from the item limit matrix before (k-1) where the weight
                          # limit is not exceeded
                        prev = {}                        
                        for z in range(j+1):
                            prev[z] = {}
                            prev[z]['value'] = ks_matrix[i-weight[j]][z][k-1][0]
                            prev[z]['items'] = ks_matrix[i-weight[j]][z][k-1][1]
                          # Take the max value between the last matrix position and the value of using the old item and new value instead
                        temp = {'1':ks_matrix[i][j][k][0],'2':max([v['value'] for k,v in prev.items()]) + value[j]}
                        ks_matrix[i][j+1][k][0] = max(temp.items())[1]
                        
                        if max(temp.items())[0] == '1':
                            ks_matrix[i][j+1][k][1] = ks_matrix[i][j][k][1]
                        else:       
                            ks_matrix[i][j+1][k][1] = prev[max(prev, key=lambda k: prev[k]['value'])]['items'] + [j]
                            
    result = 0
    items = []
    for x in ks_matrix[capacity]:
        for y in x:
            if y[0] > result:
                result = y[0]
                items = y[1]
    
    return result, items

def find_optimal_players(players_dict, budget, maxitems):
    value = [round(v['value']*10) for k,v in players_dict.items()]
    weight = [round(v['price']*10) for k,v in players_dict.items()]
    nbitems = len(players_dict)
    results = xknapsack(budget, nbitems, maxitems, weight, value)
    players = [(k,v['price'],v['value']) for k,v in players_dict.items()]
    best_players = [players[i] for i in results[1]]
    overall_score = results[0]/10
    overall_price = sum([x[1] for x in best_players])
    
    return best_players, overall_price, overall_score


def get_worst_club(club_players, players):
    res = {x:z/y for x,y,z in [x for x in players if x[0] in club_players]}
    return [x for x,y in Counter(res).most_common()[3-len(res):]]


def get_worst_pos(pos, num, players):
    res = {x:z/y for x,y,z in players if x in pos}
    return [x for x,y in Counter(res).most_common()[num-len(res):]]


def initiate_info(test_players, player_stats):
    player_clubs = {x:player_stats[x]['details']['club'] for x,y,z in test_players}
    club_counts = Counter(player_clubs.values())
    return ([x for x,y,z in test_players if player_stats[x]['details']['position'] == 'Goalkeeper'],
            [x for x,y,z in test_players if player_stats[x]['details']['position'] == 'Defender'],
            [x for x,y,z in test_players if player_stats[x]['details']['position'] == 'Midfielder'],
            [x for x,y,z in test_players if player_stats[x]['details']['position'] == 'Forward'],
            player_clubs,
            club_counts,
            [k for k,v in club_counts.items() if v > 3])

 
def find_replacement_players(n_gk, n_def, n_mid, n_for, budget, player_data):

    n = n_gk + n_def + n_mid + n_for
    
    all_players = temp_func(player_data, gameweek)  # Add pos to this function so we can use it to check if budget is possible
    if sum(sorted({v['price'] for k,v in all_players.items()})[:n])*10 > budget:
        raise ValueError('no player left within budget') # Add pos to this function to get an accurate read
    
    new_player_stats = player_data
    test_players = find_optimal_players(temp_func(player_data, gameweek), budget=budget, maxitems=n)
    budget = budget - test_players[1]*10
    new_test_players = test_players[0]
    goalkeepers, defenders, midfielders, forwards, player_clubs, club_counts, remove_clubs = initiate_info(new_test_players, player_data)
    
    res = []
    while len(goalkeepers) != n_gk or len(defenders) != n_def or len(midfielders) != n_mid or len(forwards) != n_for or max(club_counts.values()) > 3:
        print('Calulating...')
        
        worst_players = []
        for pos, num in [(goalkeepers, n_gk), (defenders, n_def), (midfielders, n_mid), (forwards, n_for)]:
            
            if len(pos) > num:
                worst_players.extend(get_worst_pos(pos, num, new_test_players))
                
        for club in remove_clubs:
            remove_club_players = [x for x,y,z in test_players if player_data[x]['details']['club'] == club]
            
            if len(remove_club_players) > 0:
                worst_players.extend(get_worst_club(remove_club_players, new_test_players))
                
                    
        if len(worst_players) > 0:
    
            new_player_stats = {k:v for k,v in new_player_stats.items() if k not in [x[0] for x in new_test_players]}
            
            new_test_players = [x for x in new_test_players if x[0] not in worst_players]
            goalkeepers, defenders, midfielders, forwards, player_clubs, club_counts, remove_clubs = initiate_info(new_test_players, player_data)
            player_pool = {}
            for pos, str1, num in [(goalkeepers, 'Goalkeeper', n_gk), (defenders, 'Defender', n_def), (midfielders, 'Midfielder', n_mid), (forwards, 'Forward', n_for)]:
                if len(pos) < num:
                    player_pool.update({k:v for k,v in new_player_stats.items() if v['details']['position'] == str1})
                    
            budget = int(sum([v['price'] for k,v in all_players.items() if k in worst_players])*10)
            x = len(worst_players)
            
            player_pool = temp_func(player_pool, gameweek)
            if sum(sorted({v['price'] for k,v in player_pool.items()})[:x])*10 > budget:
                raise ValueError('no player left within budget')
            
            test3 = find_optimal_players(player_pool, budget=budget, maxitems=x) 
    
            budget = budget - test3[1]*10
            new_test_players.extend(test3[0])
            res.extend(test3[0])
    
            goalkeepers, defenders, midfielders, forwards, player_clubs, club_counts, remove_clubs = initiate_info(new_test_players, player_data)
    
    print(new_test_players)
    
find_replacement_players(
        n_gk = 1,
        n_def = 1,
        n_mid = 3,
        n_for = 0,
        budget = 300,
        player_data = player_stats
        )  


team_path = open(f'{os.path.dirname(os.getcwd())}\\data\\Team\\Team_info.pk', 'rb')
team_info = pickle.load(team_path)

budget = team_info['info']['Bank Money']
team_players = team_info['players']['team_players']

substitute_dict = {}
for player in team_players:
    all_players[player]
# Add code to chose best formation from players chosen and best captain and vice captain for the week
    
    
    
    
    

    
    
    

# Use betting sites to identify other stats such as clean sheets and to score odds

# Use table position as an approximate estimator on who will win games further in the future - could maybe go deeper here

# If more than 1/2 injuries override rules and find best value transfer

# Set a threshold on whether to use a free transfer / bonus cards - use past data

# Set a threshold on whether it is worth it to spend 4 points on a transfer - use past data

# Give captain/vice captain to players with highest scores on team
