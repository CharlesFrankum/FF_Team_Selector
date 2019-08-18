import os

import pickle

import numpy as np



def clean_history(stats):
    new_info = {}
    if stats['history'] is not None and len(stats['history']) != 0:
        last_season = stats['history'].iloc[0]
        ppm = last_season['Pts'] / last_season['MP']
        ppm_value = ppm / float(stats['details']['Price'].split('£')[1])
        games_played = last_season['MP'] / 90

        if np.isnan(ppm):
            ppm = 0
        if np.isnan(ppm_value):
            ppm_value = 0
        
        new_info['ls_ppm'] = ppm
        new_info['ls_ppm_value'] = ppm_value
        new_info['ls_games_played'] = games_played
        new_info['new_transfer'] = False
        new_info['MinutesPlayed'] = last_season['MP']
        new_info['GoalsScored'] = last_season['GS']
        new_info['Assists'] = last_season['A']
        new_info['CleanSheets'] = last_season['CS']
    else:
        new_info['ls_ppm'] = 0
        new_info['ls_ppm_value'] = 0
        new_info['ls_games_played'] = 0
        new_info['new_transfer'] = True
        new_info['MinutesPlayed'] = 0
        new_info['GoalsScored'] = 0
        new_info['Assists'] = 0
        new_info['CleanSheets'] = 0
    
    return new_info


def clean_season(stats_dict):
    temp_df = stats_dict['stats']

    # drops results which haven't been played
    if '-' not in temp_df.iloc[-1]['OPP']:
        temp_df = temp_df.iloc[:-1]
    
    if temp_df is not None and len(temp_df) != 0:
        
        h_team = [x.split(' ')[0] if '(A)' in x else stats_dict['details']['club']  for x in temp_df['OPP']]
        a_team = [x.split(' ')[0] if '(H)' in x else stats_dict['details']['club']  for x in temp_df['OPP']]
        h_score = [x.split(' ')[2] for x in temp_df['OPP']]
        a_score = [x.split(' ')[4] for x in temp_df['OPP']]
        value = [float(x.split('£')[1]) for x in temp_df['£']]
        
        temp_df['home_team'] = h_team
        temp_df['away_team'] = a_team
        temp_df['home_score'] = h_score
        temp_df['away_score'] = a_score
        temp_df['value'] = value
        
    else:
        return temp_df

    # Delete the opposition column
    del temp_df['OPP']
    del temp_df['£']

    # Rename the columns
    temp_df.columns = ['GameWeek','Points','MinutesPlayed','GoalsScored',
                       'Assists','CleanSheets','GoalsConceded','OwnGoals',
                       'PenaltySaves','PenaltyMisses','YellowCards',
                       'RedCards','Saves','Bonus','BonusPointSystem',
                       'Influence','Creativity','Threat','IctIndex',
                       'NetTransfers', 'SelectedBy', 'home_team',
                       'away_team', 'home_goals', 'away_goals', 'value']
    # Reorder columns
    return temp_df[['GameWeek','home_team','away_team','home_goals',
                    'away_goals','Points','MinutesPlayed','GoalsScored',
                    'Assists', 'CleanSheets','GoalsConceded','OwnGoals',
                    'PenaltySaves','PenaltyMisses','YellowCards',
                    'RedCards','Saves','Bonus','BonusPointSystem',
                    'Influence','Creativity','Threat','IctIndex',
                    'NetTransfers', 'SelectedBy', 'value']]



def save_player_stats(p_stats):
    path = f'{os.path.dirname(os.getcwd())}\\data\\Players\\cleaned_player_stats.pk'
    with open(path, 'wb') as file:
        pickle.dump(p_stats, file)       


if __name__ == '__main__':
    path = f'{os.path.dirname(os.getcwd())}\\data\\Players\\player_stats.pk'
    with open(path, 'rb') as f:
        player_stats = pickle.load(f)
        
    for player, stats in player_stats.items():
        history = clean_history(stats)    
        for k,v in history.items():
            player_stats[player]['details'][k] = v
        player_stats[player]['stats'] = clean_season(stats)
    save_player_stats(player_stats)
