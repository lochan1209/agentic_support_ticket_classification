from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Define Parent (Large) and Child (Small) Splitters
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# 2. Setup Vector DB for Children and Key-Value Store for Parents
vectorstore = Chroma(collection_name="split_parents", embedding_function=OpenAIEmbeddings())
docstore = InMemoryStore()

# 3. Create the Retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 4. Add documents (This automatically creates parents, splits children, and links them)
retriever.add_documents(documents)