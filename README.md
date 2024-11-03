# Risk Chatbot

## Project Summary

The Risk project provides automated assistance with Risk queries based on Projects. It comprises two primary processes:

1. **Indexer**: Handles the ingestion and indexing of Risk documents into a searchable format using Pinecone and MongoDB.
2. **Query Bot**: Manages user interactions, retrieves relevant Risk information, and generates responses using advanced language models.

## Indexer

The Indexer processes Risk documents by chunking them based on descriptions, creating a hierarchical structure of chunks for efficient retrieval. Metadata such as book and chapter are also included to enhance search accuracy.

### Key Components

- **Document Processing**: Chunks Risk documents based on descriptions, with a hierarchical structure for Auto Merging Retrieval.
- **Metadata Handling**: Includes metadata like book and chapter.
- **Vector Store**: Stores document vectors in Pinecone for fast similarity search.
- **Document Store**: Stores the original documents and their metadata in MongoDB.

### Input Schema

The input is organized in a directory containing data with the following structure:

- **data**: Contains folders with books names.
  - **books**: Contains CSV files with the following column:
    - `Sentances`: the set of sentances of each chapter are gathered in a single csv file.

### Configuration

To configure the Indexer, ensure the following environment variables are set:

- `OPENAI_KEY`: API key for Azure OpenAI.
- `MONGO_URI`: URI for MongoDB.
- `PINECONE_API_KEY`: API key for Pinecone.

### Usage

Run the Indexer using the following command: 
```bash
python indexer_data_loader.py
```

## Query Bot

The Query Bot manages user interactions by rewriting queries, generating sub-queries, and retrieving relevant Risk information. It uses advanced techniques such as query rewriting, sub-query generation (similar to Rag Fusion), and reranking combined with Auto Merging Retrieval. The final responses are synthesized by merging the top K matched results. It supports interfaces for CLI and web-based interactions using Streamlit.

### Key Components

- **Query Rewriting**: Refines user queries using language models to improve specificity.
- **Sub-Query Generation**: Creates additional sub-queries to enhance retrieval, similar to Rag Fusion.
- **Metadata Extraction**: Extracts and utilizes metadata from queries to improve accuracy.
- **Similarity Query**: Performs similarity queries to retrieve relevant documents.
- **Auto Merging Retrieval**: Merge Smaller Chunks into to original source if a good percentage appears of it
- **Reranking**: Uses Cohere reranking to refine the retrieval results.
- **Final Merge**: Combines results using Top K matched documents.
- **Interfaces**: Provides CLI and Streamlit web-based interfaces for user interactions.

### Configuration

To configure the Query Bot, ensure the following environment variables are set:

- `OPENAI_KEY`: API key for Azure OpenAI.
- `MONGO_URI`: URI for MongoDB.
- `PINECONE_API_KEY`: API key for Pinecone.

### Usage

Run the Query Bot using the following command:
```bash
python query_bot.py <interface>
```

For Streamlit Use 
```bash
streamlit run query_bot.py
```