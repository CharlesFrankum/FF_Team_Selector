import os

import pandas as pd

import numpy as np

import pickle

from sklearn.linear_model import LinearRegression

from Mapper import df_ISO3_mapper


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
def get_old_data():
    path = f'{os.path.dirname(os.getcwd())}\\data\\Results\\results_2018.csv'
    res = pd.read_csv(path)
    # Currently grabbing last season's data from an alternative source
    # Therefore some preprocessing is needed
    res.columns = ['gameweek','datetime','loc','home_team','away_team','result']
    res['home_score'] = [int(h) for h,a in res['result'].str.split(' - ')]
    res['away_score'] = [int(a) for h,a in res['result'].str.split(' - ')]
    del res['loc']
    del res['result']
    del res['gameweek']
    res = get_results(res)
    res['datetime'] = pd.to_datetime(res['datetime'])
    return res


# Modify to get last 5 home and last 5 away
def extract_most_recent_games(df, x):
    new_df = pd.DataFrame()
    teams = set(list(df['home_team']) + list(df['away_team']))
    for i in teams:
        # excludes recently demoted teams
        if len(i) == 3:
            home = df[df['home_team'] == i].sort_values(
                by='datetime', ascending=False
                )
            away = df[df['away_team'] == i].sort_values(
                by='datetime', ascending=False
                )
            # newly promoted teams to be based partially/fully off BHA
            if len(home) < round(x/2):
                temp_home =  df[df['home_team'] == 'BHA'].sort_values(
                    by='datetime', ascending=False
                    )
                temp_home = temp_home.tail(round(x/2)-len(home))
                home = pd.concat([home, temp_home])
                home = home.replace({'BHA': i})
            if len(away) < round(x/2):
                temp_away =  df[df['away_team'] == 'BHA'].sort_values(
                    by='datetime', ascending=False
                    )
                temp_away = temp_away.tail(round(x/2)-len(away))
                away = pd.concat([away, temp_away])
                away = away.replace({'BHA': i})
                
            home = home.head(round(x/2))
            away = away.head(round(x/2))
            team_df = pd.concat([home, away])
            new_df = pd.concat([new_df, team_df])

    return new_df


def get_team_results(results):
    results_dict = {}
    teams = set(list(results['home_team']) + list(results['away_team']))
    for i in teams:
        # excludes recently demoted teams
        if len(i) == 3:
            # excludes teams with too few recent PL games
            home = results[results['home_team'] == i]
            away = results[results['away_team'] == i]
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
            
            # Currently picking up duplicates - hotfix to remove
            home = home.drop_duplicates()
            away = away.drop_duplicates()
            all_games = all_games.drop_duplicates()
            
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


def GetWinRates(df, z):
    return len([x for x in df['home_results'] if x == z]) / len(df['home_results'])


def generate_results_prediction(results_dict):
    for k,v in results_dict.items():
        home_df = results_dict[k]['home']
        away_df = results_dict[k]['away']
        all_games_df = results_dict[k]['all_games'] 

        results_dict[k]['win_rate_home'] = GetWinRates(home_df, 1)
        results_dict[k]['draw_rate_home'] = GetWinRates(home_df, 0)
        results_dict[k]['lose_rate_home'] = GetWinRates(home_df, -1)
        results_dict[k]['win_rate_away'] = GetWinRates(away_df, 1)
        results_dict[k]['draw_rate_away'] = GetWinRates(away_df, 0)
        results_dict[k]['lose_rate_away'] = GetWinRates(away_df, -1)
        results_dict[k]['goals_scored_home'] = sum(
                [x for x in results_dict[k]['home']['home_score']]
                ) / len(home_df['home_score'])
        results_dict[k]['goals_conceded_home'] = sum(
                [x for x in results_dict[k]['home']['away_score']]
                ) / len(home_df['home_score'])
        results_dict[k]['goals_scored_away'] = sum(
                [x for x in results_dict[k]['away']['away_score']]
                ) / len(away_df['away_score'])
        results_dict[k]['goals_conceded_away'] = sum(
                [x for x in results_dict[k]['away']['home_score']]
                ) / len(home_df['away_score'])
    
        home = np.array(home_df['home_results'])
        away = np.array(away_df['away_result'])
        all_games_scores = np.array(all_games_df['results'])
    
        h_time = np.array(home_df['games_played']).reshape(-1, 1)
        a_time = np.array(away_df['games_played']).reshape(-1, 1)
        all_time = np.array(all_games_df['games_played']).reshape(-1, 1)
    
        reg1 = LinearRegression().fit(h_time,home)
        results_dict[k]['h_result_prediction'] = float(
                reg1.predict(int(max(h_time)) + 1)
                )
        reg2 = LinearRegression().fit(a_time,away)
        results_dict[k]['a_result_prediction'] = float(
                reg2.predict(int(max(a_time)) + 1)
                )
        reg3 = LinearRegression().fit(all_time,all_games_scores)
        results_dict[k]['result_prediction'] = float(
                reg3.predict(int(max(all_time)) + 1)
                )
        
    return results_dict

    
def save_results_stats(res):
    path = f'{os.path.dirname(os.getcwd())}\\data\\Results\\results_stats.pk'
    with open(path, 'wb') as file:
        pickle.dump(res, file)             

                    

if __name__ == '__main__':
    
    path = f'{os.path.dirname(os.getcwd())}\\data\\Results\\results.csv'
    results = pd.read_csv(path)
    results['datetime'] = pd.to_datetime(results['datetime'])
    
    if get_gameweek() < 6:
        # Currently old results haven't been pre-mapped
        map_path = f'{os.path.dirname(os.getcwd())}\\data\\Maps\\Team_maps.pickle'
        with open(map_path, 'rb') as f:
            mapper = pickle.load(f)
        
        results = pd.concat([results, get_old_data()])
        results = df_ISO3_mapper(results, mapper) 
        results = extract_most_recent_games(results, 10)
            
    results_dict = get_team_results(results)
    results_dict = generate_results_prediction(results_dict)
    
    save_results_stats(results_dict)
  
