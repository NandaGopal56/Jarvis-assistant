import requests
import os
import json
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

# Load environment variables from .env file
load_dotenv()


class SearchProvider(ABC):
    """Abstract base class for search providers."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self.get_api_key_from_env()
    
    @staticmethod
    @abstractmethod
    def get_api_key_from_env() -> str:
        """Retrieve the API key from environment variables."""
        pass

    @abstractmethod
    def search(self, query: str) -> Optional[Dict[str, Any]]:
        """Perform a search using the provider's API."""
        pass

    @abstractmethod
    def extract_results(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract search results from the provider's API response."""
        pass


class BraveSearchProvider(SearchProvider):
    """Implementation of Brave Search provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        country: str = "IN",
        search_lang: str = "en",
        count: int = 20,
        safesearch: str = "strict",
        result_filter: str = "discussions,faq,infobox,news,query,summarizer,web,locations"
    ):
        """
        Initialize BraveSearchProvider with optional parameters.
        
        Args:
            api_key (Optional[str]): Brave API key (defaults to environment variable).
            country (str): Country code for search (default: "IN").
            search_lang (str): Search language (default: "en").
            count (int): Number of results to fetch (default: 20).
            safesearch (str): Safe search filter level (default: "strict").
            result_filter (str): Types of results to include (default: all types).
        """
        super().__init__(api_key)
        self.url = "https://api.search.brave.com/res/v1/web/search"

        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key
        }
        # Individual parameters
        self.country = country
        self.search_lang = search_lang
        self.count = count
        self.safesearch = safesearch
        self.result_filter = result_filter

    @staticmethod
    def get_api_key_from_env() -> str:
        """Retrieve the Brave API key from environment variables."""
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            raise ValueError("Error: BRAVE_API_KEY not found in environment variables.")
        return api_key

    def search(self, query: str) -> Optional[Dict[str, Any]]:
        """Perform a search using the Brave Search API."""
        search_params = {
            "q": query,
            "country": self.country,
            "search_lang": self.search_lang,
            "count": self.count,
            "safesearch": self.safesearch,
            "result_filter": self.result_filter
        }

        try:
            response = requests.get(self.url, headers=self.headers, params=search_params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error during Brave Search API request: {e}")
            return None

    def extract_results(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Extract search results specific to Brave Search.
        
        Args:
            data (Dict[str, Any]): JSON response from Brave API.
        
        Returns:
            List[Dict[str, str]]: Extracted search results with title and URL.
        """
        extracted_results = []
        web = data.get("web", {})
        results = web.get("results", [])
        for index, result in enumerate(results):
            title = result.get("title")
            url = result.get("url")
            if title and url:
                extracted_results.append({"index": index, "title": title, "url": url})
        return extracted_results


class SearchEngine:
    """Main class to manage different search providers."""

    def __init__(self, provider: SearchProvider):
        self.provider = provider

    def perform_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Perform a search using the configured provider."""
        return self.provider.search(query)

    def extract_results(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract results using the provider's specific method."""
        return self.provider.extract_results(data)


def save_results_to_file(results: List[Dict[str, str]], output_file: str) -> None:
    """
    Save extracted search results to a JSON file.

    Args:
        results (List[Dict[str, str]]): Extracted search results.
        output_file (str): Path to the output JSON file.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)
        print(f"Search results saved to {output_file}")
    except Exception as e:
        print(f"Error while saving results to file: {e}")


if __name__ == "__main__":
    # User input for query and optional parameters
    query = input("Enter your search query: ").strip() or "Django serializer"
    params = {
        "country": input("Enter country code (default: IN): ").strip() or "IN",
        "search_lang": input("Enter search language (default: en): ").strip() or "en",
        "count": int(input("Enter number of results (default: 20): ").strip() or 20),
        "safesearch": input("Enter safesearch level (default: strict): ").strip() or "strict",
        "result_filter": input("Enter result filters (default: all types): ").strip() or 
        "discussions,faq,infobox,news,query,summarizer,web,locations"
    }

    try:
        # Initialize BraveSearchProvider with optional parameters
        brave_provider = BraveSearchProvider()

        # Use SearchEngine with BraveSearchProvider
        search_engine = SearchEngine(brave_provider)

        # Perform the search
        search_results = search_engine.perform_search(query)

        if search_results:
            # Extract relevant search results
            extracted_results = search_engine.extract_results(search_results)

            # Save results to file
            output_file = "test/output_file.json"
            save_results_to_file(extracted_results, output_file)
        else:
            print("No results found or an error occurred.")
    except ValueError as e:
        print(e)
