import os

import pandas as pd

import numpy as np

import pickle

from sklearn.linear_model import LinearRegression

from TeamMapper import df_ISO3_mapper

def get_gameweek():
    path = f'{os.path.dirname(os.getcwd())}\\data\\Fixtures\\fixtures.csv'
    df = pd.read_csv(path)
    return min(df['gameweek'])


# gets results from scores
def get_results(df):
    h_results = []
    a_results = []
    for row in df.iterrows():
        row = df.iloc[row[0]]
        if row['home_score'] > row['away_score']:
            h_results.append(1)
            a_results.append(-1)
        elif row['home_score'] < row['away_score']:
            h_results.append(-1)
            a_results.append(1)
        else:
            h_results.append(0)
            a_results.append(0)
    df['home_results'] = h_results
    df['away_result'] = a_results
    return df


# A minimum of 4 weeks of this season's data is needed.
# Will combine last season's data if we have less than 4.
# Note - newly promoted teams won't have relevant data (gets excluded later)
def get_old_data(mapper):
    path = f'{os.path.dirname(os.getcwd())}\\data\\Results\\results_2018.csv'
    res = pd.read_csv(path)
    res = df_ISO3_mapper(res, mapper)
    
    # Currently grabbing last season's data from an alternative source
    # Therefore some preprocessing is needed
    res.columns = ['gameweek','datetime','loc','home_team','away_team','result']
    # Get 12 - x games from last season
    res = res[res['gameweek'] <= (12 - get_gameweek())]
    res['home_score'] = [int(h) for h,a in res['result'].str.split(' - ')]
    res['away_score'] = [int(a) for h,a in res['result'].str.split(' - ')]
    del res['loc']
    del res['result']
    del res['gameweek']
    res = get_results(res)
    return res


def get_team_results(results):
    results_dict = {}
    teams = set(list(results['home_team']) + list(results['away_team']))
    for i in teams:
        # excludes recently demoted teams
        if len(i) == 3:
            # excludes teams with too few recent PL games
            home = results[results['home_team'] == i]
            away = results[results['away_team'] == i]
            if len(home) > 2 and len(away) > 2:
                results_dict[i] = {}
                
                all_games = pd.concat([home, away])
                results_h = [
                        x[1] for x in enumerate(
                                all_games['home_results']
                                ) if all_games.iloc[x[0]]['home_team'] == i
                        ]
                results_a = [
                        x[1] for x in enumerate(
                                all_games['away_result']
                                ) if all_games.iloc[x[0]]['away_team'] == i
                        ]
                all_results = results_h + results_a   
                all_games['results'] = all_results   
            
                home.sort_values(
                        by='datetime', inplace=True, ascending=True
                        )
                away.sort_values(
                        by='datetime', inplace=True, ascending=True
                        )
                all_games.sort_values(
                        by='datetime', inplace=True, ascending=True
                        )
            
                home['games_played'] = range(len(home))
                away['games_played'] = range(len(away))
                all_games['games_played'] = range(len(all_games))
            
                results_dict[i]['home'] = home
                results_dict[i]['away'] = away
                results_dict[i]['all_games'] = all_games
                
    return results_dict


def generate_results_prediction(results_dict):
    for k,v in results_dict.items():
        home_df = results_dict[k]['home']
        away_df = results_dict[k]['away']
        all_games_df = results_dict[k]['all_games'] 
    
        results_dict[k]['win_rate_home'] = len([x for x in home_df['home_results'] if x == 1]) / len(home_df['home_results'])
        results_dict[k]['draw_rate_home'] = len([x for x in home_df['home_results'] if x == 0]) / len(home_df['home_results'])
        results_dict[k]['lose_rate_home'] = len([x for x in home_df['home_results'] if x == -1]) / len(home_df['home_results'])
        results_dict[k]['win_rate_away'] = len([x for x in away_df['home_results'] if x == 1]) / len(away_df['home_results'])
        results_dict[k]['draw_rate_away'] = len([x for x in away_df['home_results'] if x == 0]) / len(away_df['home_results'])
        results_dict[k]['lose_rate_away'] = len([x for x in away_df['home_results'] if x == -1]) / len(away_df['home_results'])
        results_dict[k]['goals_scored_home'] = sum([x for x in results_dict[k]['home']['home_score']]) / len(home_df['home_score'])
        results_dict[k]['goals_conceded_home'] = sum([x for x in results_dict[k]['home']['away_score']]) / len(home_df['home_score'])
        results_dict[k]['goals_scored_away'] = sum([x for x in results_dict[k]['away']['away_score']]) / len(away_df['away_score'])
        results_dict[k]['goals_conceded_away'] = sum([x for x in results_dict[k]['away']['home_score']]) / len(home_df['away_score'])
    
        home = np.array(home_df['home_results'])
        away = np.array(away_df['away_result'])
        all_games_scores = np.array(all_games_df['results'])
    
        h_time = np.array(home_df['games_played']).reshape(-1, 1)
        a_time = np.array(away_df['games_played']).reshape(-1, 1)
        all_time = np.array(all_games_df['games_played']).reshape(-1, 1)
    
        reg1 = LinearRegression().fit(h_time,home)
        results_dict[k]['h_result_prediction'] = float(reg1.predict(int(max(h_time)) + 1))
        reg2 = LinearRegression().fit(a_time,away)
        results_dict[k]['a_result_prediction'] = float(reg2.predict(int(max(a_time)) + 1))
        reg3 = LinearRegression().fit(all_time,all_games_scores)
        results_dict[k]['result_prediction'] = float(reg3.predict(int(max(all_time)) + 1))
        
    return results_dict

    
def save_results_stats(res):
    path = f'{os.path.dirname(os.getcwd())}\\data\\Results\\results_stats.pk'
    with open(path, 'wb') as file:
        pickle.dump(res, file)       



if __name__ == '__main__':
    
    path = f'{os.path.dirname(os.getcwd())}\\data\\Maps\\Team_maps.pickle'
    with open(path, 'rb') as file:
        mapper = pickle.load(file)
        
    # results data for name mapping
    path = f'{os.path.dirname(os.getcwd())}\\data\\Results\\results.csv'
    results = pd.read_csv(path)
    results = df_ISO3_mapper(results, mapper)
    
    if get_gameweek() < 4:
        results = pd.concat([results, get_old_data(mapper)])
        
    results_dict = get_team_results(results)
    results_dict = generate_results_prediction(results_dict)
    
    save_results_stats(results_dict)
  
