import os

import sys

sys.path.insert(1, f'{os.path.dirname(os.getcwd())}\\models\\')

from datetime import datetime

from time import sleep

import pandas as pd

from Mapper import df_ISO3_mapper


def get_fixture_data(url, driver):
    # Get Fixture data for gameweeks 1-38
    home_teams = []
    away_teams = []
    date_times = []
    gameweeks = []
    
    gw_counter = 0
    for i in range(1,39):
        gw_counter += 1
        week = url+str(i)
        driver.get(week)
        sleep(1)
        game_days = driver.find_elements_by_css_selector('div.sc-bdVaJa.eIzRjw')
        for day in game_days:
            date = day.find_element_by_tag_name('h4').text
            game_day = day.find_element_by_tag_name('ul').text
            games = game_day.split('\n')
            if ':' in game_day:
                # work around to keep loop consistent with game updates
                n_games = []
                for item in games:
                    new_items = item.split(':')
                    for i in new_items:
                        n_games.append(i)
                for i in range(0, len(n_games), 4):
                    home_teams.append(n_games[i])
                    away_teams.append(n_games[i+3])
                    
                    date_time = datetime.strptime(date, '%A %d %B %Y')
                    date_times.append(date_time)
                    gameweeks.append(gw_counter)
        df = pd.DataFrame({'home_team':home_teams,'away_team':away_teams,'datetime':date_times,'gameweek':gameweeks})
    return df[['home_team','away_team','gameweek','datetime']]


def save_csv(data):
    path = f'{os.path.dirname(os.getcwd())}\\data\\Fixtures\\fixtures.csv'
    data.to_csv(path, index=0, sep=',')

    
def collect(driver, mapper):
    print('Collecting fixtures...')
    fixtures_url = 'https://fantasy.premierleague.com/fixtures/'
    fixtures = get_fixture_data(fixtures_url, driver)
    fixtures = df_ISO3_mapper(fixtures, mapper)
    save_csv(fixtures)
