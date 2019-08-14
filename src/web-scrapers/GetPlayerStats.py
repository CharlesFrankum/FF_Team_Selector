import os

import pickle

import pandas as pd

from datetime import datetime

from time import sleep

from tqdm import tqdm

import re

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains



def scrape_current_player_stats(url):
    driver.get(url)
    sleep(2)
    players_dict = {}                                              
    print('Collecting live player statistics..')
    for page in tqdm(range(1,19)):
        players = driver.find_elements_by_css_selector('#root > div:nth-child(2) > div > div > table > tbody > tr')                                                                                                
        # Get player stats
        for player in players:
            driver.execute_script("scroll(0,300);")
            ActionChains(driver).move_to_element(player).perform()
            sleep(0.5)
            # click on player info pop up
            player.find_element_by_css_selector('div > button').click()
            sleep(0.5)
            # Player name, club, position and fitness        
            info_box = driver.find_element_by_css_selector('div.ElementDialog__Summary-gmefnd-0.eSbpQR')
            details = info_box.find_elements_by_tag_name('div')[0].text.split('\n')
            name, pos, club = details[0], details[1], details[2]
            # Get the player fitness and expected return if available
            try:
                fitness = driver.find_element_by_css_selector('div.sc-bdVaJa.edJDIA').text
                if '%' in fitness:
                    fitness = int(''.join([x for x in fitness if x.isdigit()]))
                    recovery_date = None
                elif 'Expected back' in fitness:
                    fitness = 0 #! fix this
                    recovery_date = datetime.strptime(f"{fitness.split('Expected back ')[1]} 2019", '%d %M %Y')
                elif 'Suspended' in fitness:
                    fitness = 100
                    recovery_date = datetime.strptime(f"{fitness.split('Suspended until ')[1]} 2019", '%d %b %Y')
                else:
                    fitness = 0
                    recovery_date = 'Unknown'
            except:
                fitness = 100
                recovery_date = None
                
            # Get top level stats/info
            general_info = driver.find_element_by_css_selector('ul.ElementDialog__StatList-gmefnd-6.dYJASM').text.split('\n')
            info_tuples = []
            for i in range(0,len(general_info),2):
                info_tuples.append((general_info[i], general_info[i+1]))
            ff_stats = driver.find_element_by_css_selector('ul.ElementDialog__FantasyStatList-gmefnd-10.ldSydD').text.split('\n')
            ff_tuples = []
            for i in range(0,len(ff_stats),2):
                ff_tuples.append((ff_stats[i], ff_stats[i+1]))
            # Full player stats data for premier league season
            x = driver.page_source
            player_data = pd.read_html(x)
            if len(player_data) < 3:
                player_stats = None
                player_history = player_data[1]
                player_history.columns = [re.sub('\n','',x).strip().split(' ')[0] for x in player_history.columns]
            else:
                player_stats = player_data[1]
                player_stats.columns = [re.sub('\n','',x).strip().split(' ')[0] for x in player_stats.columns]
                player_history = player_data[2]
                player_history.columns = [re.sub('\n','',x).strip().split(' ')[0] for x in player_history.columns]        
            # Close the info pop up
            ActionChains(driver).send_keys(Keys.ESCAPE).perform()    
            players_dict[name] = {}
            players_dict[name]['details'] = {}
            players_dict[name]['details']['club'] = club
            players_dict[name]['details']['position'] = pos
            players_dict[name]['details']['fitness'] = fitness
            players_dict[name]['details']['recovery_date'] = recovery_date
            players_dict[name]['stats'] = player_stats
            players_dict[name]['history'] = player_history
            
            for i, x in info_tuples:
                players_dict[name]['details'][i] = x
            for i, x in ff_tuples:
                players_dict[name]['details'][i] = x
        # Next page
        driver.find_element_by_css_selector('button:nth-child(4)').click()
        sleep(1)
    
    return players_dict

def save_player_stats(player_stats):
    path = f'{os.path.dirname(os.getcwd())}\\data\\Players\\player_stats.pk'
    with open(path, 'wb') as file:
        pickle.dump(player_stats, file)       


if __name__ == '__main__':
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome('chromedriver.exe', chrome_options=options)
    
    fixtures_url = 'https://fantasy.premierleague.com/a/statistics/total_points'
    player_stats = scrape_current_player_stats(fixtures_url)
    save_player_stats(player_stats)
