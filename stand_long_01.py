from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import numpy as np
import matplotlib.pyplot as plt
import time

# 셀레니움 옵션 설정
options = Options()
options.add_argument("--headless")  # 창 띄우지 않음
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0")

# 드라이버 생성
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

prices = []

# 쿠팡 1~5 페이지 순회
for page in range(1, 6):
    url = f"https://www.coupang.com/np/search?q=장스탠드&page={page}&channel=user"
    print(f"📄 {page}페이지 접속 중...")

    driver.get(url)
    time.sleep(3)  # 로딩 대기 (필요시 늘리기)

    # 가격 정보 수집
    try:
        elems = driver.find_elements(By.CSS_SELECTOR, "strong.price-value")
        print(f"✅ {page}페이지 가격 {len(elems)}개 발견")

        for e in elems:
            price_str = e.text.replace(",", "").replace("원", "").strip()
            if price_str.isdigit():
                price = int(price_str)
                if 5000 < price < 500000:
                    prices.append(price)
    except Exception as e:
        print(f"❌ {page}페이지 에러: {e}")

driver.quit()

# 가격 통계 출력
if prices:
    prices_np = np.array(prices)
    print("\n📊 수집 결과 요약")
    print("총 수집 상품 수:", len(prices_np))
    print("최저가:", prices_np.min(), "원")
    print("1사분위수:", np.percentile(prices_np, 25), "원")
    print("중앙값:", np.median(prices_np), "원")
    print("3사분위수:", np.percentile(prices_np, 75), "원")
    print("최고가:", prices_np.max(), "원")
    print("평균가:", int(prices_np.mean()), "원")

    # 히스토그램 시각화
    plt.hist(prices_np, bins=20, edgecolor='black')
    plt.title("쿠팡 장스탠드 가격 분포 (1~5페이지)")
    plt.xlabel("가격 (원)")
    plt.ylabel("상품 수")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ 수집된 가격 데이터가 없습니다.")
