from abc import ABC, abstractmethod
from typing import List, Union
from langchain_community.document_loaders import WebBaseLoader, PythonLoader, PythonLoader
from langchain.docstore.document import Document
import asyncio


class LoaderProvider(ABC):
    """
    Abstract base class for loaders.
    """

    def __init__(self):
        self.docs: List[Document] = []

    @abstractmethod
    async def load_documents(self, *args, **kwargs) -> List[Document]:
        """
        Abstract method to load documents and return a list of Documents.
        """
        pass


class WebLoader(LoaderProvider):
    """
    Concrete implementation of LoaderProvider for web-based loaders.
    """

    def __init__(self, requests_per_second: int = 1):
        super().__init__()
        self.requests_per_second = requests_per_second
        self._loader: Union[WebBaseLoader, None] = None  # Define the loader as None initially

    def _create_loader(self, urls: List[str], *args, **kwargs) -> None:
        """
        Private method to create a WebBaseLoader instance with the specified URLs.
        """
        self._loader = WebBaseLoader(urls, **kwargs)
        self._loader.requests_per_second = self.requests_per_second

    async def load_documents(self, urls: List[str], *args, **kwargs) -> List[Document]:
        """
        Asynchronously loads documents from the web using the loader.
        """
        if not self._loader:
            self._create_loader(urls, **kwargs)

        async for doc in self._loader.alazy_load():
            self.docs.append(doc)

        return self.docs


class PythonFileLoader(LoaderProvider):
    """
    Concrete implementation of LoaderProvider for file-based loaders.
    """

    def __init__(self):
        super().__init__()
        self._loader: Union[PythonFileLoader, None] = None  # Define the loader as None initially

    def _create_loader(self, file_path: str, *args, **kwargs) -> None:
        """
        Private method to create a PythonFileLoader instance with the specified file path.
        """
        self._loader = PythonLoader(file_path, **kwargs)

    async def load_documents(self, file_path: str, *args, **kwargs) -> List[Document]:
        """
        Asynchronously loads documents from a file using the loader.
        """
        if not self._loader:
            self._create_loader(file_path, **kwargs)

        # Assuming PythonFileLoader has an async method to load documents
        self.docs.extend(self._loader.load())
        return self.docs


class Loader:
    """
    Main class to manage different types of loaders.
    """

    def __init__(self, provider: LoaderProvider):
        self.provider = provider

    async def load_documents(self, *args, **kwargs) -> List[Document]:
        """
        Delegates the document loading to the provider.
        """
        return await self.provider.load_documents(*args, **kwargs)



async def main():
    """
    Main function to orchestrate document loading and text splitting.
    """
    
    # web_loader_provider = WebLoader(requests_per_second=10)
    # web_loader = Loader(web_loader_provider)
    # urls = ["https://en.wikipedia.org/wiki/Shah_Rukh_Khan"]
    # documents = await web_loader.load_documents(urls=urls)
    # print(f"Loaded {len(documents)} documents from web.")

    # Test PythonFileLoader
    print("\nTesting PythonFileLoader...")
    python_loader_provider = PythonFileLoader()
    python_loader = Loader(python_loader_provider)
    file_path = "test/sample_doc.py"
    python_doc = await python_loader.load_documents(file_path=file_path)
    print(f"Loaded {len(python_doc)} documents from file.")
    print(python_doc)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
