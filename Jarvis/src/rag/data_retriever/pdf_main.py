import asyncio
from dotenv import load_dotenv
from typing import List
from unstructured.partition.pdf import partition_pdf
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm.llm_manager import LanguageModelFactory
from src.globals.configs import ModelProvider, GroqModelName

load_dotenv()

class PDFProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_content(self):
        chunks = partition_pdf(
            filename=self.file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )
        
        tables, texts, images = [], [], []
        
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk.metadata.text_as_html)
            elif "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
                images.extend(self._extract_images(chunk))
                
        return texts, tables, images
    
    def _extract_images(self, chunk) -> List[str]:
        images_b64 = []
        chunk_els = chunk.metadata.orig_elements
        for el in chunk_els:
            if "Image" in str(type(el)):
                images_b64.append(el.metadata.image_base64)
        return images_b64


class TextSummarizer:
    def __init__(self, model):
        self.model = model
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an assistant tasked with summarizing tables and text.
            Give a concise summary of the table or text.
            Respond only with the summary, no additional comment.
            Table or text chunk: {element}
            """
        )
        self.summarize_chain = {"element": lambda x: x} | self.prompt | self.model | StrOutputParser()
    
    def summarize(self, elements: List[str]) -> List[str]:
        return self.summarize_chain.batch(elements, {"max_concurrency": 3})


class ImageAnalyzer:
    def __init__(self, model):
        self.model = model
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("user", [
                {"type": "text", "text": "Describe the image in detail. For context, the image is part of a research paper explaining the transformers architecture. Be specific about graphs, such as bar plots."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}}
            ])
        ])
        self.chain = self.prompt_template | self.model | StrOutputParser()
    
    def analyze(self, images: List[str]) -> List[str]:
        return self.chain.batch(images)


async def main(file_path: str):
    pdf_processor = PDFProcessor(file_path)
    texts, tables, images = pdf_processor.extract_content()

    # Create language model instance
    model = await LanguageModelFactory.create_model(
                        provider=ModelProvider.GROQ,
                        model_name=GroqModelName.LLAMA_3_2_1B
                    )
    
    messages = [
        ("system", "You are a helpful assistant that knows about Indian history. Answer the question under 10 words."),
        ("human", "When did India get its independence?")
    ]
    
    # response = await model.generate_response(messages)
    # print(f"Response: {response}\n")

    text_summarizer = TextSummarizer(model)
    table_summarizer = TextSummarizer(model)
    # image_analyzer = ImageAnalyzer(ChatOpenAI(model="gpt-4o-mini"))
    
    text_summaries = text_summarizer.summarize(texts)
    table_summaries = table_summarizer.summarize(tables)
    # image_summaries = image_analyzer.analyze(images)
    
    return text_summaries + table_summaries #+ image_summaries


if __name__ == "__main__":
    file_path = "/Users/nnandagopal/Desktop/personal_projects/Jarvis/Jarvis/src/rag/data_retriever/attention.pdf"
    summaries = asyncio.run(main(file_path))
    for summary in summaries:
        print(summary)
