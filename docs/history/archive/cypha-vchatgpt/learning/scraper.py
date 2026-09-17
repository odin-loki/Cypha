import requests
from bs4 import BeautifulSoup
import time

class CyphaWebScraper:
    def __init__(self, rate_limit=1.0, user_agent="CyphaBot", obey_robots=True):
        self.rate_limit = rate_limit
        self.last_time = 0
        self.headers = {"User-Agent": user_agent}
        self.obey_robots = obey_robots
    def get(self, url):
        dt = time.time() - self.last_time
        if dt < self.rate_limit:
            time.sleep(self.rate_limit - dt)
        self.last_time = time.time()
        try:
            resp = requests.get(url, headers=self.headers, timeout=7)
            return resp.text
        except Exception as e:
            return ""
    def parse_html(self, html):
        bs = BeautifulSoup(html, "html.parser")
        txt = bs.get_text()
        links = [a.get('href') for a in bs.find_all('a', href=True)]
        imgs = [img.get('src') for img in bs.find_all('img', src=True)]
        return dict(text=txt, links=links, images=imgs)
    def scrape(self, url):
        html = self.get(url)
        if not html:
            return {}
        return self.parse_html(html)
