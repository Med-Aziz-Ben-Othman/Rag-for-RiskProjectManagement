import os
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI configuration
OPENAI_KEY = os.getenv('OPENAI_KEY')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')

MONGO_URI = os.getenv('MONGO_URI')
# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'