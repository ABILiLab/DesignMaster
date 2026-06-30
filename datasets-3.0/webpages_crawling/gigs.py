import re
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

import pickle


def extract_protac_ids(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # 无头模式
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    protac_ids = []

    while True:
        try:
            # 确保 tbody 存在
            tbody = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "tbMain"))
            )
            # 确保 tbody 内至少有一个 tr 元素
            WebDriverWait(driver, 10).until(
                lambda d: len(tbody.find_elements(By.TAG_NAME, "tr")) > 0
            )
            tr_elements = tbody.find_elements(By.TAG_NAME, "tr")

            print(f"当前页面找到 {len(tr_elements)} 个 tr 元素")

            # 若 tr_elements 为空，直接终止（冗余保护）
            if not tr_elements:
                print("未找到任何 tr 元素，终止爬取。")
                break

            # 每 3 个 tr 取第一个
            for i in range(0, len(tr_elements), 3):
                try:
                    tr = tr_elements[i]
                    a_tag = tr.find_element(By.TAG_NAME, "a")
                    href = a_tag.get_attribute("href")
                    match = re.search(r'id=(\d+)', href)
                    if match:
                        protac_id = match.group(1)
                        protac_ids.append(protac_id)
                        print(f"提取到 PROTAC ID: {protac_id}")
                except Exception as e:
                    print(f"处理第 {i} 个 tr 时出错：", e)

            # 处理分页
            try:
                pagination_container = driver.find_element(By.CLASS_NAME, "ui-pagination-container")
                next_link = pagination_container.find_element(By.XPATH, ".//a[contains(text(),'Next')]")
                # 点击前滚动到链接位置（可选）
                driver.execute_script("arguments[0].scrollIntoView();", next_link)
                print("点击 Next 分页链接...")
                next_link.click()
                # 等待新页面加载，可根据页面特性调整等待条件
                time.sleep(3)
            except Exception as e:
                print("没有找到 Next 分页链接，分页结束。")
                break

        except TimeoutException:
            print("等待 tr 元素超时，可能无数据。")
            break
        except Exception as e:
            print("处理 tbody 时出错:", e)
            break

    driver.quit()
    return protac_ids


if __name__ == '__main__':
    with open('linker_to_protac.pkl', 'rb') as handle:
        linker_to_protac = pickle.load(handle)
    for i in range(1, 2754):
        if i not in linker_to_protac.keys():
            print('------------------------' + str(i) + '-----------------------')
            try:
                url = "http://cadd.zju.edu.cn/protacdb/compound/dataset=linker&id=" + str(i)
                ids = extract_protac_ids(url)
            except Exception as e:
                continue
            linker_to_protac[i] = ids
            with open('linker_to_protac.pkl', 'wb') as handle:
                pickle.dump(linker_to_protac, handle)
