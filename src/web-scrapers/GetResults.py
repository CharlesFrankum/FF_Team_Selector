import os

from datetime import datetime

from time import sleep

import pandas as pd

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException



def get_season_results(url):
    driver.get(url)
    # holder lists to store data and then append to dataframe
    datetime_temp = []
    home_team_temp = []
    away_team_temp = []
    home_score_temp = []
    away_score_temp = []
    
    for page in range(1,10):
        x = driver.page_source
        sleep(1)
        form_data = pd.read_html(x)
        # loop through the tables, clean the data and concatenate in a single table
        for row in enumerate(form_data):
            try:
                datetime_temp.append(datetime.strptime(row[1].iloc[0,0], '%d.%m.%y %H:%M'))
            except ValueError:
                datetime_temp.append(None)
            try:
                home_team_temp.append(row[1].iloc[0,2])
            except IndexError:
                home_team_temp.append(None)
            try:
                away_team_temp.append(row[1].iloc[0,4])
            except IndexError:
                away_team_temp.append(None)
            try:
                score = row[1].iloc[0,3]
                home_score_temp.append(int(score.split(' - ')[0]))
            except IndexError:
                home_score_temp.append(None)
            try:
                away_score_temp.append(int(score.split(' - ')[1]))
            except UnboundLocalError:
                away_score_temp.append(None)
        try:
            next_page = driver.find_element_by_css_selector("[title='Next Page']")
            next_page.click()
            sleep(1)
        except NoSuchElementException:
            pass
        
    # Create an emtpty table to be filled
    PL_results = pd.DataFrame({'datetime':datetime_temp,'home_team':home_team_temp,'away_team':away_team_temp,'home_score':home_score_temp,'away_score':away_score_temp})
    # reorder the columns
    PL_results = PL_results[['datetime','home_team','away_team','home_score','away_score']]
    # Drop rows with missing data and drop duplicates
    PL_results = PL_results.drop_duplicates()
    PL_results = PL_results.dropna()
    
    return PL_results


def add_results_outcomes(res):
    # Ready the results data for linear regression
    # Add a home/away W/L/D column to the table
    home_result = []
    away_result = []
    
    for i in range(len(res)):
        h_score = res.iloc[i]['home_score']
        a_score = res.iloc[i]['away_score']
        if h_score > a_score:
            home_result.append(1)
            away_result.append(-1)
        elif h_score == a_score:
            home_result.append(0)
            away_result.append(0)
        elif h_score < a_score:
            home_result.append(-1)
            away_result.append(1)
    
    res['home_results'] = home_result
    res['away_result'] = away_result
    # Convert to datetime
    res['datetime'] = pd.to_datetime(res['datetime'])
    return res


def save_csv(res):
    res_path = f'{os.path.dirname(os.getcwd())}\\data\\Results\\results.csv'
    res.to_csv(res_path, index=0, sep=',')



if __name__ == '__main__':
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome('chromedriver.exe', chrome_options=options)
    
    results_url = 'https://www.scorespro.com/soccer/england/premier-league/results/'

    results = get_season_results(results_url)
    results = add_results_outcomes(results)
    save_csv(results)
    
    driver.quit()
