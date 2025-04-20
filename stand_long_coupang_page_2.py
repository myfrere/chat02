import requests
from selenium import webdriver
from selenium.webdriver.common.by import By

# URL 설정
url = 'https://www.coupang.com/np/search?q=%EA%B1%B0%EC%8B%A4+%EC%9E%A5%EC%8A%A4%ED%83%A0%EB%93%9C&channel=auto&component=&eventCategory=SRP&trcid=&traid=&sorter=scoreDesc&minPrice=&maxPrice=&priceRange=&filterType=&listSize=36&filter=&isPriceRange=false&brand=&offerCondition=&rating=0&page5&rocketAll=false&searchIndexingToken=1=9&backgroundColor='

# 크롬 드라이버 설정
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options)

# URL로 접근
driver.get(url)

# 페이지가 로딩될 때까지 대기 (필요한 경우)
driver.implicitly_wait(10)

# 검색 결과를 추출 (예: 제품 가격)
products = driver.find_elements(By.CLASS_NAME, 'search-product')

# 각 제품의 가격을 출력
for product in products:
    try:
        price = product.find_element(By.CLASS_NAME, 'price').text
        print(price)
    except:
        pass

driver.quit()
