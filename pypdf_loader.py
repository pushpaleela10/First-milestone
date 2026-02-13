from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('foods.pdf')
docs = loader.load()
print(len(docs))
print(docs[0].metadata)