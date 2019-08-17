import GetFixtures

import GetPlayerStats

import GetResults

import GetTable

import GetTeamInfo

from StartWebdriver import launch


if __name__ == '__main__':
    driver = launch()
    GetFixtures.collect(driver)
    GetPlayerStats.collect(driver)
    GetResults.collect(driver)
    GetTable.collect()
    GetTeamInfo.collect(driver)
    driver.quit()
