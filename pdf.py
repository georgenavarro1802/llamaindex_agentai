import os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core.indices import VectorStoreIndex, load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.readers.file.docs.base import PDFReader

# load env
load_dotenv()

# OpenAI api key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def get_index(data, index_name):
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


# get documents from data loader
loader = PDFReader()
pdf_path = Path("./data/Canada.pdf")
canada_pdf = loader.load_data(file=pdf_path)

# engine
canada_index = get_index(canada_pdf, "canada_data")
canada_engine = canada_index.as_query_engine()

