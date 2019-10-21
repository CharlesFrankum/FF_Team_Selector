import os

import pickle

import numpy as np

import pandas as pd


def GetHistoryDataPoints(history):
    if history is not None and len(history) != 0:
        row = stats['history'].iloc[0]
        ppm = row['Pts'] / row['MP']
        value = float(stats['details']['Price'].split('£')[1])
        ppm_value = ppm / value
        gp = row['MP'] / 90

        if np.isnan(ppm):
            ppm = 0
        if np.isnan(ppm_value):
            ppm_value = 0
            
        dp = (ppm, ppm_value, gp, True, row['MP'], row['GS'], row['A'], row['CS'])
        
    else:
        dp = (0, 0, 0, False, 0, 0, 0, 0)
        
    return dp


def GenerateHistoryFeatures(stats):    
    x1, x2, x3, x4, x5, x6, x7, x8 = GetHistoryDataPoints(stats['history'])
    features = {'ls_ppm': x1,
                'ls_ppm_value': x2,
                'ls_games_played': x3,
                'new_transfer': x4,
                'MinutesPlayed': x5,
                'GoalsScored': x6,
                'Assists': x7,
                'CleanSheets': x8,
                    }
    stats['details'].update(features)


def GetFeatureColumns(df, stats):
    h_team, a_team, h_score, a_score, value = [], [], [], [], []
    for i, row in df.iterrows():
        game_data = row['OPP'].split()
        if '(A)' in game_data:
            h_team.append(game_data[0])
            a_team.append(stats['details']['club'])
        else:
            a_team.append(game_data[0])
            h_team.append(stats['details']['club'])
        h_score.append(float(game_data[2]))
        a_score.append(float(game_data[4]))
        value.append(float(row['£'].split('£')[1]))

    df['home_team'] = h_team
    df['away_team'] = a_team
    df['home_score'] = h_score
    df['away_score'] = a_score
    df['value'] = value
    return df


def RenameReorderCols(df):
        df.columns = ['GameWeek','Points','MinutesPlayed','GoalsScored',
                       'Assists','CleanSheets','GoalsConceded','OwnGoals',
                       'PenaltySaves','PenaltyMisses','YellowCards',
                       'RedCards','Saves','Bonus','BonusPointSystem',
                       'Influence','Creativity','Threat','IctIndex',
                       'NetTransfers', 'SelectedBy', 'home_team',
                       'away_team', 'home_goals', 'away_goals', 'value']
        # Reorder columns
        df = df[['GameWeek','home_team','away_team','home_goals',
                        'away_goals','Points','MinutesPlayed','GoalsScored',
                        'Assists', 'CleanSheets','GoalsConceded','OwnGoals',
                        'PenaltySaves','PenaltyMisses','YellowCards',
                        'RedCards','Saves','Bonus','BonusPointSystem',
                        'Influence','Creativity','Threat','IctIndex',
                        'NetTransfers', 'SelectedBy', 'value']]
        return df


def GenerateSeasonFeatures(player, stats):
    stats_df = stats['stats']
    if stats_df is not None and len(stats_df) != 0 and 'GW' in stats_df.columns:
        if 'Totals' in list(stats_df['GW']):
            stats_df = stats_df.iloc[:-1]
        else:
            raise ValueError(f'Totals missing from {player}')

        stats_df = GetFeatureColumns(stats_df, stats)
        h_team, a_team, h_score, a_score, value = [], [], [], [], []
        
        for i, row in stats_df.iterrows():
            game_data = row['OPP'].split()
            if '(A)' in game_data:
                h_team.append(game_data[0])
                a_team.append(stats['details']['club'])
            else:
                a_team.append(game_data[0])
                h_team.append(stats['details']['club'])
            h_score.append(float(game_data[2]))
            a_score.append(float(game_data[4]))
            value.append(float(row['£'].split('£')[1]))
            
        stats_df['home_team'] = h_team
        stats_df['away_team'] = a_team
        stats_df['home_score'] = h_score
        stats_df['away_score'] = a_score
        stats_df['value'] = value
        
        stats_df = stats_df.drop(columns=['£', 'OPP'])
        
        return RenameReorderCols(stats_df)


def save_player_stats(p_stats):
    path = f'{os.path.dirname(os.getcwd())}\\data\\Players\\cleaned_player_stats.pk'
    with open(path, 'wb') as file:
        pickle.dump(p_stats, file)       


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    path = f'{os.path.dirname(os.getcwd())}\\data\\Players\\player_stats.pk'
    with open(path, 'rb') as  f:
        player_stats = pickle.load(f)

    for player, stats in player_stats.items():
        GenerateHistoryFeatures(stats)
        player_stats[player]['stats'] = GenerateSeasonFeatures(player, stats)
    save_player_stats(player_stats)
