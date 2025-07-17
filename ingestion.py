from dotenv import load_dotenv

load_dotenv()


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# nested list
docs = [
    WebBaseLoader(url).load() for url in urls
]  # a list of lists of one single article

# flatten the list
docs_list = [item for sublist in docs for item in sublist]

# define splitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250,
    chunk_overlap=0,
)

# split the documents
docs_split = text_splitter.split_documents(docs_list)

# ### comment this after first run because we don't want to index this everytime we run the program
# print(f"Going to add {len(docs_split)} documents to Pinecone...")
#
# # create vectorstore
# vectorstore = PineconeVectorStore(
#     index_name="agentic-rag",
#     embedding=embeddings,
# )
#
# # define upserting's batch size
# batch_size = 10
#
# # loop through documents in batches
# for i in range(0, len(docs_split), batch_size):
#     batch = docs_split[i:i + batch_size]
#     # convert into embeddings and upsert to pinecone vectorstore
#     vectorstore.add_documents(batch)
#     print(f"Uploaded batch {i // batch_size + 1} with {len(batch)} documents...")
#
# print("::::: Loading to Pinecone vectorstore done!!! :::::")


# create retriever object
retriever = PineconeVectorStore(
    index_name="agentic-rag",
    embedding=embeddings,
).as_retriever()
