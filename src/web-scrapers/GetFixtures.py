import os

from datetime import datetime

from time import sleep

import pandas as pd

from tqdm import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.options import Options



def get_fixture_data(url):
    # Get Fixture data for gameweeks 1-38
    home_teams = []
    away_teams = []
    date_times = []
    gameweeks = []
    
    gw_counter = 0
    for i in tqdm(range(1,39)):
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
                for i in range(0, len(games), 3):
                    home_teams.append(games[i])
                    away_teams.append(games[i+2])
                    
                    time = games[i+1]
                    date_time = ' '.join([date, time])
                    date_time = datetime.strptime(date_time, '%A %d %B %Y %H:%M')
                    date_times.append(date_time)
                    gameweeks.append(gw_counter)
        df = pd.DataFrame({'home_team':home_teams,'away_team':away_teams,'datetime':date_times,'gameweek':gameweeks})
    return df[['home_team','away_team','gameweek','datetime']]


def save_csv(data):
    path = f'{os.path.dirname(os.getcwd())}\\data\\Fixtures\\fixtures.csv'
    data.to_csv(path, index=0, sep=',')



if __name__ == '__main__':
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome('chromedriver.exe', chrome_options=options)

    fixtures_url = 'https://fantasy.premierleague.com/fixtures/'
    fixtures = get_fixture_data(fixtures_url)
    save_csv(fixtures)
