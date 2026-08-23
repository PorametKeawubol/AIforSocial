# **🧪 Lab Sheet: Web Scraping Using Selenium in Python Virtual Environment**

---

## **📝 Lab Title**

Web Scraping with Selenium in a Python Virtual Environment

---

## **🎯 Objectives**

By the end of this lab, students will be able to:

* Set up and activate a Python virtual environment

* Install necessary packages using `pip`

* Use `chromedriver-autoinstaller` with `selenium` to scrape data

* Run a headless browser session for web scraping

* Extract web content using BeautifulSoup

---

## **🧰 Prerequisites**

* Basic knowledge of Python programming

* Python 3 installed on the system

* Google Chrome browser installed // [https://googlechromelabs.github.io/chrome-for-testing/](https://googlechromelabs.github.io/chrome-for-testing/) 

---

## **🖥️ Software Requirements**

* Python 3.x

* pip (Python package installer)

* Virtual environment (`venv`)

* Required Python packages:

  * `selenium`

  * `beautifulsoup4`

  * `chromedriver-autoinstaller`

  * `pandas`

---

## **📦 Steps**

### **✅ Step 1: Create and Activate a Virtual Environment**

**Linux / macOS:**

bash  
CopyEdit  
`python3 -m venv venv`  
`source venv/bin/activate`

**Windows:**

bash  
CopyEdit  
`python -m venv venv`  
`venv\Scripts\activate`

---

### **✅ Step 2: Install Required Packages**

bash  
CopyEdit  
`pip install selenium beautifulsoup4 pandas chromedriver-autoinstaller`

---

### **✅ Step 3: Write and Run the Python Script**

Create a file named `scraper.py` and paste the following code:

`import time`  
`from bs4 import BeautifulSoup`  
`from selenium import webdriver`  
`import chromedriver_autoinstaller`  
`from selenium.webdriver.common.by import By`  
`from selenium.webdriver.support.ui import WebDriverWait`  
`from selenium.webdriver.support import expected_conditions as EC`

`# Auto-install compatible ChromeDriver`  
`chromedriver_autoinstaller.install()`

`# Setup Chrome options (uncomment headless if you want no browser UI)`  
`chrome_options = webdriver.ChromeOptions()`  
`# chrome_options.add_argument('--headless')`  
`# chrome_options.add_argument('--no-sandbox')`  
`# chrome_options.add_argument('--disable-dev-shm-usage')`

`# Initialize the WebDriver`  
`driver = webdriver.Chrome(options=chrome_options)`

`# Target URL`  
`url = "https://www.kfc.co.th/menu/meals"`  
`driver.get(url)`

`# Wait for page JS to load (adjust if needed)`  
`time.sleep(5)`

`# Try to accept cookies by clicking the button`  
`try:`  
    `wait = WebDriverWait(driver, 10)`  
    `accept_btn = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))`  
    `accept_btn.click()`  
    `print("Clicked the accept cookies button.")`  
`except Exception as e:`  
    `print("Button not found or not clickable:", e)`

`# Save the page HTML to a file`  
`html = driver.page_source`  
`with open("kfc_menu_page.html", "w", encoding="utf-8") as f:`  
    `f.write(html)`

`# Parse the HTML with BeautifulSoup`  
`soup = BeautifulSoup(html, 'html.parser')`

`# Find all divs with class 'small-menu-product-header'`  
`headers = soup.find_all("div", class_="small-menu-product-header")`

`# Print out menu item titles`  
`print("📋 Menu Item Titles:")`  
`for idx, header in enumerate(headers, 1):`  
    `print(f"{idx}. {header.get_text(strip=True)}")`

`# Close the browser`  
`driver.quit()`

Run the script:

`python scraper.py`

---

Scaping Image   
[https://chatgpt.com/share/688b7ef3-6050-8013-a61a-d21b5b6bc813](https://chatgpt.com/share/688b7ef3-6050-8013-a61a-d21b5b6bc813)

