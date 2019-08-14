import os

import pandas as pd

import requests



def get_PL_table(url):
    x = requests.get(url)
    league_table = pd.read_html(x.content)[0]
    del league_table['Last 6']
    
    return league_table


def save_csv(tab):
    tab_path = f'{os.path.dirname(os.getcwd())}\\data\\Table\\table.csv'
    tab.to_csv(tab_path, index=0, sep=',')



if __name__ == '__main__':
    table_url = 'https://www.skysports.com/premier-league-table'
    pl_table = get_PL_table(table_url)
    save_csv(pl_table)
