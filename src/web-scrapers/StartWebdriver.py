from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def launch():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome('chromedriver.exe', chrome_options=options)
    return driver
