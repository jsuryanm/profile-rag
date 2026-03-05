import requests 
import logging 
from src.config.settings import MCP_SERVER_URL

logger = logging.getLogger(__name__)

class ProfileService:

    def fetch_profile(self,profile_url: str):
        """Fetch profile  data from MCP server"""

        try:
            response = requests.post(f"{MCP_SERVER_URL}/profile",
                                     json={"profile_url":profile_url},
                                     timeout=30)
            
            response.raise_for_status()
            
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch profile: {e}")
            return None