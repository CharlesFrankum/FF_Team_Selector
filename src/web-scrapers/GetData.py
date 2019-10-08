import os

import pickle

import GetFixtures

import GetPlayerStats

import GetResults

import GetTable

import GetTeamInfo

import config

from StartWebdriver import launch

if __name__ == '__main__':
    
    if config.fantasy_email == '' or config.fantasy_password == '':
        raise PermissionError('Please update the config file')
    
    path = f'{os.path.dirname(os.getcwd())}\\data\\Maps\\Team_maps.pickle'
    with open(path, 'rb') as f:
        mapper = pickle.load(f)
        
    driver = launch()
    GetFixtures.collect(driver, mapper)
    GetPlayerStats.collect(driver, mapper)
    GetResults.collect(driver, mapper)
    GetTable.collect(mapper)
    GetTeamInfo.collect(driver)
    driver.quit()