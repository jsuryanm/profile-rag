import logging 
import requests 
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright

from src.config.settings import settings 

logger = logging.getLogger(__name__)

mcp = FastMCP("linkedin-scraper",port=settings.mcp_server_port)

@mcp.tool()
async def scrape_linkedin(url: str, use_mock: bool = False) -> dict:
    """
    Scrapes a LinkedIn profile and returns structured profile data. 

    Args:
        url: Full LinkedIn profile URL e.g. https://www.linkedin.com/in/username/
        use_mock: If True, returns pre-saved mock data (recommended for development
                  to avoid LinkedIn bot detection).
    Returns:
        Dictionary with profile fields: full_name, headline, location,
        about, experience, education, skills, source_url.
    """
    if use_mock:
        return _load_mock_data()
    return await _scrape_profile(url)

def _load_mock_data() -> dict:
    """
    Fetches pre-saved mock LinkedIn JSON from IBM Cloud. 
    Use during development to avoid hitting LinkedIn's bot detection.
    """
    logger.info("Loading mock LinkedIn data")
    try:
        response = requests.get(settings.mock_data_url,timeout=30)
        response.raise_for_status()
        data = response.json()
        return {
            k:v for k,v in data.items()
            if v not in ([],"",None)
            and k not in ["people_also_viewed","certifications"]}
    
    except Exception as e:
        logger.error(f"Mock data loading fail:{e}")
        return {"error":str(e)}
    
async def _scrape_profile(url: str) -> dict:
    """
    Headless Chromium scraper via Playwright.

    ⚠️  LinkedIn limitation:
        Logged-out visitors see limited data.
        Reliably scraped: name, headline, location.
        May scrape: partial experience, some education.
        Won't scrape: skills, full history, recommendations.

        Use use_mock=True for development.
    """
    logger.info(f"Scrapping LinkedIn profile: {url}")

    async with async_play