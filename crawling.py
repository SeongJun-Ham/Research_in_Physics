import os
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import urllib.request

def img_crawler(target_name, target_image_count, target_save_dir):
    driver = webdriver.Chrome()

    os.chdir(target_save_dir)
    google_search_prefix_URL = "https://www.google.com/search?tbm=isch&q="
    driver.get(google_search_prefix_URL+target_name)

    for _ in range(2):
        driver.find_element(By.CSS_SELECTOR, "body").send_keys(Keys.END)
        time.sleep(1)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    image_info_list = driver.find_elements(By.CSS_SELECTOR, '.mNsIhb')

    image_and_name_list = []

    for i, image_info in enumerate(image_info_list):
        if i == target_image_count:
            break

        save_image = image_info.find_element(By.CSS_SELECTOR, 'img').get_attribute('src')

        image_path = os.path.join(target_name.replace(' ', '_') + '_' + str(i+1) + '.jpg')
        image_and_name_list.append((save_image, image_path))

    for i in range(len(image_and_name_list)):
        urllib.request.urlretrieve(image_and_name_list[i][0], image_and_name_list[i][1])

    driver.close()


if __name__ == "__main__":
    # query = "고양이"
    image_cnt = 100
    save_dir_cat = "E:/Users/sj879/Desktop/vscode/Research_in_Physics/cat_picture"
    save_dir_dog = "E:/Users/sj879/Desktop/vscode/Research_in_Physics/dog_picture"
    
    
    img_crawler("고양이", image_cnt, save_dir_cat)
    img_crawler("강아지", image_cnt, save_dir_dog)

