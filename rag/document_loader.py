from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("data/Vantis_Energy_Solutions_Knowledge_Base.pdf")

docs = data.load()
