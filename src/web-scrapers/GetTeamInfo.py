import os

import config

import pandas as pd

import pickle

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from time import sleep


def login_to_ff(home_url, transfer_url):
    driver.get(home_url)
    sleep(delay)
    username_el = driver.find_element_by_css_selector('#loginUsername')
    password_el = driver.find_element_by_css_selector('#loginPassword')
    query = '#root > div:nth-child(2) > div > div > div.sc-bdVaJa.edJDIA > form'
    login = driver.find_element_by_css_selector(query)  \
            .find_element_by_tag_name('button')         
    # Enter username/password and log in
    username_el.send_keys(config.fantasy_email)
    password_el.send_keys(config.fantasy_password)
    login.click()
    sleep(delay)
    # Open the transfer page
    driver.get(transfer_url)
    sleep(delay)


def get_current_team():
    # Get the current list of players and prices
    query = 'div.sc-bdVaJa.sc-bwzfXH.Pitch__ElementRow-sc-1mctasb-1.Pitch__PitchRow-sc-1mctasb-2.iZgvKz'
    player_rows = driver.find_elements_by_css_selector(query)
    row_text = []
    for row in player_rows:
        row = row.text.split('\n')
        for x in row:
            row_text.append(x)
    current_players = [
            x for i,x in enumerate(row_text) if (i/2).is_integer()
            ]
    current_prices = [
            float(x) for i,x in enumerate(row_text) if not (i/2).is_integer()
            ]
    #team_size = len(current_players)
    
    player_price = pd.DataFrame(
            {'team_players':current_players,'sell_price':current_prices}
            )
    player_price = player_price[['team_players','sell_price']]
    
    return player_price, {'Sell Price': sum(player_price['sell_price'])}


def get_scoreboard_info(info):
    query = 'div.Scoreboard__CostScoreboardUnit-sc-117tw9n-1.cMrwVx'
    transferboard_elements = driver.find_elements_by_css_selector(query)
    wild_card_status = transferboard_elements[2].text
    if wild_card_status == 'Play Wildcard':
        wild_card_status = True
    else:
        wild_card_status = False
    free_transfers = transferboard_elements[3].text.split('\n')[1]
    bank_money = float(transferboard_elements[5].text.split('\n')[1])
    info['Wild Card'] = wild_card_status
    info['Free Transfers'] = free_transfers
    info['Bank Money'] = bank_money
    return info


# Grab starting line up and captain and vice_captain info
def get_line_up(team_url):
    # Go to the team select page
    driver.get(team_url)
    sleep(delay)

    s_team_rows = driver.find_elements_by_css_selector('div.sc-bdVaJa.sc-bwzfXH.Pitch__ElementRow-sc-1mctasb-1.Pitch__PitchRow-sc-1mctasb-2.iZgvKz')
    b_team_row = driver.find_element_by_css_selector('div.Bench-sc-1sz52o9-0.gujtGz')
    
    players = []
    is_captain = []
    is_v_captain = []
    is_bench = []
    for row in s_team_rows:
        units = row.find_elements_by_css_selector('div.Pitch__PitchUnit-sc-1mctasb-3.gYXrCB')
        for unit in units:
            player = unit.find_elements_by_css_selector('div.Pitch__ElementName-sc-1mctasb-6.cAvCos')
            if not len(player) == 0:
                is_bench.append(0)
                players.append(player[0].text)
                player[0].click()
                sleep(0.5)
                dia_box = driver.find_element_by_css_selector('div.Dialog__StyledDialogBody-sc-5bogmv-7.heKQNQ')
                lis = dia_box.find_elements_by_tag_name('li')
                if len(lis) == 3:
                    if 'Make Captain' in lis[1].text:
                        is_v_captain.append(1)
                        is_captain.append(0)
                    elif 'Make Vice Captain' in lis[1].text:
                        is_captain.append(1)
                        is_v_captain.append(0)
                else:
                    is_captain.append(0)
                    is_v_captain.append(0)
                
                ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                sleep(0.5)
    units = b_team_row.find_elements_by_css_selector('div.Pitch__ElementName-sc-1mctasb-6')
    for unit in units:
        players.append(unit.text)
        is_captain.append(0)
        is_v_captain.append(0)
        is_bench.append(1)
                
    df = pd.DataFrame({
            'team_players':players,
            'is_captain':is_captain,
            'is_v_captain':is_v_captain,
            'is_bench': is_bench
            })
    return df


def get_cards_status(scoreboard):
    cards = driver.find_elements_by_css_selector('li.MyTeam__ChipItem-sc-6ytjxx-2.bDjYjd')
    for card in cards:
        card_data = card.text.split('\n')
        if card_data[1] == 'PLAY':
            scoreboard[card_data[0]] = 1
        else:
            scoreboard[card_data[0]] = 0
    return scoreboard
    

def save_team_info(team):
    path = f'{os.path.dirname(os.getcwd())}\\data\\Team\\Team_info.pk'
    with open(path, 'wb') as f:
        pickle.dump(team, f)


def get_transfer_info():
    login_to_ff(home_url, transfer_url)
    team_df, info = get_current_team()
    info = get_scoreboard_info(info)
    captain_df = get_line_up(team_url)
    df = pd.merge(team_df, captain_df, on = 'team_players')
    team_info = {
            'players': df,
            'info': get_cards_status(info)
            }
    save_team_info(team_info)



if __name__ == '__main__':
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome('chromedriver.exe', chrome_options=options)
    delay = 3
    
    home_url = 'https://fantasy.premierleague.com'
    transfer_url = 'https://fantasy.premierleague.com/a/squad/transfers'
    team_url = 'https://fantasy.premierleague.com/a/team/my'
    
    get_transfer_info()
