from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import numpy as np

# Selenium을 사용하여 크롬 브라우저 열기
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # UI 없이 실행

# WebDriver 설정
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# 쿠팡 장스탠드 검색 URL
base_url = "https://www.coupang.com/np/search?component=&q=장스탠드&channel=user"

# 가격 데이터를 저장할 리스트
prices = []

# 첫 5페이지를 반복하며 가격 수집
for page_num in range(1, 6):
    try:
        # 페이지 URL 수정
        url = f"{base_url}&page={page_num}"
        driver.get(url)
        time.sleep(3)  # 페이지가 완전히 로드될 때까지 잠시 대기
        
        # 가격 정보 추출 (가격 클래스명은 쿠팡에서 확인한 후 수정)
        price_elements = driver.find_elements(By.CSS_SELECTOR, 'span.selling-price')  # 가격 클래스명 예시
        for price_element in price_elements:
            price_text = price_element.text.strip()
            if price_text:
                prices.append(int(price_text.replace(",", "").replace("원", "")))  # 가격만 숫자로 변환
        
        print(f"📄 {page_num}페이지 가격 수집 완료!")
        
    except Exception as e:
        print(f"페이지 {page_num} 로딩 실패: {str(e)}")

# 가격 데이터가 수집되었는지 확인
if prices:
    print(f"수집된 가격 데이터: {prices}")
else:
    print("수집된 가격 데이터가 없습니다.")

driver.quit()

# 가격 데이터 분석
if prices:
    # 가격 임계점 (예: 평균, 중간값 등)
    avg_price = np.mean(prices)
    median_price = np.median(prices)
    print(f"평균 가격: {avg_price:.2f}원")
    print(f"중간값 가격: {median_price:.2f}원")
