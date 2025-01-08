from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List



class TextSplitter:
    """
    Handles text splitting using the provided splitter configuration.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initializes the TextSplitter with specific configurations.

        Args:
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The number of overlapping characters between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )


    def split_documents(self, documents: List[Document]) -> List[str]:
        """
        Splits the documents into smaller chunks.

        Args:
            documents (List[Document]): List of documents to split.

        Returns:
            List[str]: List of text chunks.
        """
        return self.splitter.split_documents(documents)



if __name__ == "__main__":
    documents = []
    
    text_splitter = TextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(documents)

    # Print text chunks
    for chunk in text_chunks:
        print(chunk)