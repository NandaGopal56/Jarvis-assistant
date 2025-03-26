import asyncio
from dotenv import load_dotenv
from typing import List
from unstructured.partition.pdf import partition_pdf
from langchain_core.output_parsers import StrOutputParser
from src.llm.llm_manager import LanguageModelFactory
from src.globals.configs import ModelProvider, GroqModelName
from dataclasses import dataclass, field

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

@dataclass
class TextSummarizer:
    llm: any = field(init=False)
    output_parser: StrOutputParser = field(default_factory=StrOutputParser)
    prompt_template: List[tuple] = field(default_factory=lambda: [
        ("system", "Summarize the table or text concisely."),
        ("human", "Table or text chunk: {element}")
    ])

    def __post_init__(self):
        asyncio.create_task(self._initialize())

    async def _initialize(self):
        self.llm = await LanguageModelFactory.create_model(ModelProvider.GROQ, GroqModelName.LLAMA_3_2_1B)

    async def summarize(self, elements: List[str]) -> List[str]:
        return await asyncio.gather(*(self._invoke_llm(e) for e in elements))

    async def _invoke_llm(self, element: str) -> str:
        response = await self.llm.generate_response([(r, m.format(element=element)) for r, m in self.prompt_template])
        return self.output_parser.parse(response)

@dataclass
class ImageAnalyzer:
    llm: any = field(init=False)
    output_parser: StrOutputParser = field(default_factory=StrOutputParser)
    prompt_template: List[tuple] = field(default_factory=lambda: [
        ("system", "Describe the image in detail. Be specific about all the details in the image like any statistics, graphs, or anything."),
        ("human", "{image}")
    ])

    def __post_init__(self):
        asyncio.create_task(self._initialize())

    async def _initialize(self):
        self.llm = await LanguageModelFactory.create_model(ModelProvider.GROQ, GroqModelName.LLAMA_3_2_90b_VISION_PREVIEW)
    
    async def _invoke_llm(self, image: str) -> str:
        formatted_prompt = [(role, msg.format(image=f"data:image/jpeg;base64,{image}")) for role, msg in self.prompt_template]
        response = await self.llm.generate_response(formatted_prompt)
        return self.output_parser.parse(response)

    async def analyze(self, images: List[str]) -> List[str]:
        return await asyncio.gather(*(self._invoke_llm(image) for image in images))

async def main(file_path: str, process_text_flag: bool, process_tables_flag: bool, process_images_flag: bool):
    pdf_processor = PDFProcessor(file_path)
    texts, tables, images = pdf_processor.extract_content()

    text_summarizer = TextSummarizer() if process_text_flag else None
    table_summarizer = TextSummarizer() if process_tables_flag else None
    image_analyzer = ImageAnalyzer() if process_images_flag else None
    
    text_summaries = await text_summarizer.summarize(texts) if process_text_flag else texts
    table_summaries = await table_summarizer.summarize(tables) if process_tables_flag else []
    image_summaries = await image_analyzer.analyze(images) if process_images_flag else []
    
    return text_summaries + table_summaries + image_summaries

def get_feature_flags():
    # Fetch from a feature flag service, database, or config
    return {
        "process_text_flag": True,
        "process_tables_flag": True,
        "process_images_flag": False,
    }

if __name__ == "__main__":
    file_path = "/Users/nnandagopal/Desktop/personal_projects/Jarvis/Jarvis/src/rag/data_retriever/PwC_Wikipedia.pdf"
    
    feature_flags = get_feature_flags()
    
    summaries = asyncio.run(main(
        file_path, 
        feature_flags.get('process_text_flag'), 
        feature_flags.get('process_tables_flag'), 
        feature_flags.get('process_images_flag')
    ))

    for summary in summaries:
        print(summary)
        print(f'-'*30)
