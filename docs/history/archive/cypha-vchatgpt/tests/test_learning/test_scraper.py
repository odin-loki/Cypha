from learning.scraper import CyphaWebScraper

def test_scrape_example():
    scraper = CyphaWebScraper(rate_limit=0.0)
    data = scraper.scrape("https://www.example.com/")
    assert "text" in data
    assert "links" in data
