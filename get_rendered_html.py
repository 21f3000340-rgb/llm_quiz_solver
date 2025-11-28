from langchain.tools import tool
from playwright.sync_api import sync_playwright

@tool
def get_rendered_html(url: str) -> str:
    """Render a webpage with JavaScript and return final HTML."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
        return html
