To create a **webhook for LINE Messaging API** using Flask, you’ll need to **set up a webhook URL**, configure **LINE Developers Console**, and **verify** the webhook token.

### **Steps Overview:**

1. **Create a LINE Messaging API** channel.

2. **Set up a Flask server** that listens to the webhook.

3. **Verify webhook token** (if needed).

4. **Configure the webhook URL** in LINE Developer Console.

---

### **1\. Create a LINE Messaging API Channel**

1. Go to the LINE Developers Console.

2. **Create a provider** and a **channel** under the **Messaging API** section.

3. Note down:

   * **Channel Secret** is 6e4c3c4466aa82a6b7f4ad860a021958

   * **Channel Access Token** (used for authorization)

---

### **2\. Set Up Flask Server with Webhook**

Create a new directory and set up a **virtual environment**:  
`mkdir line_webhook`  
`cd line_webhook`  
`python -m venv venv`  
`source venv/bin/activate  # On Windows: venv\Scripts\activate`

Install Flask and `line-bot-sdk` (LINE SDK for Python):  
`pip install flask line-bot-sdk`

Create `webhook_server.py`:

`from flask import Flask, request, abort`  
`from linebot import LineBotApi, WebhookHandler`  
`from linebot.exceptions import InvalidSignatureError`  
`from linebot.models import` MessageEvent`,TextMessage, TextSendMessage`

`app = Flask(__name__)`

`# Your LINE Channel Secret and Access Token`  
`CHANNEL_SECRET = 'YOUR_CHANNEL_SECRET'`  
`CHANNEL_ACCESS_TOKEN = 'YOUR_CHANNEL_ACCESS_TOKEN'`

`line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)`  
`handler = WebhookHandler(CHANNEL_SECRET)`

`# Webhook endpoint for LINE to call`  
`@app.route("/", methods=['POST'])`  
`def callback():`  
    `# Get X-Line-Signature header value`  
    `signature = request.headers['X-Line-Signature']`

    `# Get request body as text`  
    `body = request.get_data(as_text=True)`

    `try:`  
        `# Handle the webhook event`  
        `handler.handle(body, signature)`  
    `except InvalidSignatureError:`  
        `abort(400)`

    `return 'OK'`

`# Event handler for text messages`  
`@handler.add(MessageEvent, message=TextMessage)`  
`def handle_message(event):`  
    `# Send back the same text message`  
    `line_bot_api.reply_message(`  
        `event.reply_token,`  
        `TextSendMessage(text="Received: " + event.message.text))`

`if __name__ == "__main__":`  
    `app.run(port=5000)`

In the code above:

* **`LINE Channel Secret`** and **`Access Token`** are used to authenticate requests from LINE.

* **`/callback`** is the route where LINE will send the webhook requests.

* **`handle_message()`** handles incoming text messages and replies with the same message.

---

### **3\. Expose Local Server with Cloudflare**

# **Setting Up Cloudflare Tunnel for a LINE Bot Webhook**

## **Step 1\. Install Cloudflare Tunnel (`cloudflared`)**

### **Windows**

Install using **Winget**:  
winget install Cloudflare.cloudflared

Verify the installation:  
cloudflared \--version

### **Linux (Ubuntu/Debian)**

Download and install the package:  
curl \-L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \-o cloudflared.deb  
sudo dpkg \-i cloudflared.deb

Verify the installation:  
cloudflared \--version

---

## **Step 2\. Start the Flask Application**

Run your Flask application locally. By default, it should listen on **port 5000**.  
Example:  
Start the Flask server:  
`python webhook_server.py`

---

## **Step 3\. Start a Cloudflare Tunnel**

Open a new terminal and expose your local Flask application:  
cloudflared tunnel \--url http://localhost:5000

Cloudflare will generate a public HTTPS URL similar to:  
https://your-random-name.trycloudflare.com

Keep this terminal running while testing.  
---

## **Step 4\. Configure the LINE Webhook**

1. Open the **LINE Developers Console**.  
2. Select your **Messaging API** channel.  
3. Navigate to **Messaging API → Webhook URL**.  
4. Set the webhook URL using your Cloudflare Tunnel URL followed by **`/callback`**.

Example:  
https://your-random-name.trycloudflare.com/callback

5. Click **Update**.  
6. Click **Verify** to ensure LINE can reach your application.  
7. Enable **Use Webhook**.

---

## **Step 5\. Test the Webhook**

Send a message to your LINE Official Account.  
If the configuration is correct:

* LINE forwards the webhook request to your Flask application through Cloudflare Tunnel.  
* The Flask server logs the incoming request.  
* The bot processes the message and returns the configured response (e.g., echoing the received message).

Example server log:  
127.0.0.1 \- \- \[31/Jul/2026 09:00:15\] "POST /callback HTTP/1.1" 200 \-  
Received message: Hello  
Reply sent: Hello

The LINE Bot is now accessible securely over HTTPS using Cloudflare Tunnel without requiring ngrok.

---

Assignment : Chatbot สำหรับค้หาสาขา ร้าน Advice   
 [https://www.advice.co.th/wheretobuy?srsltid=AfmBOoqgONhKTiIr9J-wCinaW\_d9OJhKh3rVBiPWNJO087cjuv\_nsp9t](https://www.advice.co.th/wheretobuy?srsltid=AfmBOoqgONhKTiIr9J-wCinaW_d9OJhKh3rVBiPWNJO087cjuv_nsp9t)

Here is a basic Selenium script in Python to:

1. Open the Advice.co.th website,

2. Click on the "สาขาใกล้ฉัน" (Find Store Near Me) button,

3. Extract and print out the store names shown.

---

### **✅ Prerequisites**

Install dependencies:

`pip install selenium`

You also need the ChromeDriver that matches your browser version.  
---

### **🧠 Selenium Script**

import time  
from bs4 import BeautifulSoup  
from selenium import webdriver  
import chromedriver\_autoinstaller  
from selenium.webdriver.common.by import By  
from selenium.webdriver.support.ui import WebDriverWait  
from selenium.webdriver.support import expected\_conditions as EC

\# Auto-install compatible ChromeDriver  
chromedriver\_autoinstaller.install()

\# Setup Chrome options (uncomment headless if you want no browser UI)  
chrome\_options \= webdriver.ChromeOptions()  
\# chrome\_options.add\_argument('--headless')  
\# chrome\_options.add\_argument('--no-sandbox')  
\# chrome\_options.add\_argument('--disable-dev-shm-usage')

\# Initialize the WebDriver  
driver \= webdriver.Chrome(options=chrome\_options)

try:  
    \# Step 1: Go to Advice website  
    driver.get("https://www.advice.co.th/wheretobuy?srsltid=AfmBOoqgONhKTiIr9J-wCinaW\_d9OJhKh3rVBiPWNJO087cjuv\_nsp9t")

    \# Step 2: Wait for page to load  
    time.sleep(5)  \# Adjust or replace with WebDriverWait if needed

    \# Optional: save the page HTML to a local file  
    with open("advice\_page01.html", "w", encoding="utf-8") as f:  
        f.write(driver.page\_source)  
    \# Step 3: Click on "สาขาใกล้ฉัน"  
    \# Locate the input box by ID and type text  
    search\_input \= driver.find\_element(By.ID, "shop\_find")  
    search\_input.clear()  
    search\_input.send\_keys("หาดใหญ่")  \# Type the location name in Thai

    \# Optional: press Enter (if needed)  
    search\_input.submit()

    time.sleep(3)

    \# Optional: save the page HTML to a local file  
    with open("advice\_page02.html", "w", encoding="utf-8") as f:  
        f.write(driver.page\_source)

      
    \# Step 4: Scrape store names  
    \# Get HTML directly from Selenium (no need to read file)  
    html \= driver.page\_source  
    soup \= BeautifulSoup(html, "html.parser")

    \# Extract branch names and links  
    branches \= soup.select(".list-items-branch h3 \> a")

    for idx, branch in enumerate(branches, 1):  
        name \= branch.text.strip()  
        link \= branch\["href"\]  
        print(f"{idx}. {name} \--\> {link}")  

finally:  
    driver.quit()

