import os
import csv
from typing import Generator, List
from llama_index.core import Document
from utils.utils import get_hash
from utils.logger import Logger

class DataLoader:
    def __init__(self, folder_path: str):
        """Initialize the DataLoader with the specified folder path."""
        self.folder_path = folder_path
        if not os.path.exists(self.folder_path):
            raise ValueError(f"The folder path '{self.folder_path}' does not exist.")
    
    def load_data(self) -> Generator[List[Document], None, None]:
        """Load data from CSV files in the specified folder and yield Document instances."""
        for root, dirs, files in os.walk(self.folder_path):
            book_name = os.path.basename(root)  # Name of each book folder
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    chapter_name = file.split(".csv")[0]  # Chapter name from filename
                    rows = []

                    with open(file_path, mode='r', encoding='utf-8') as f:
                        try:
                            reader = csv.DictReader(f)
                            for row in reader:
                                if "Sentence" not in row:
                                    Logger.get_root_logger("legal_bot_indexer").error(
                                        f"'Sentence' column missing in {file_path}."
                                    )
                                    continue
                                sentence_text = row["Sentence"]
                                doc_id = get_hash(sentence_text)
                                rows.append(Document(
                                    text=sentence_text,
                                    doc_id=doc_id,
                                    metadata={
                                        "book": book_name,
                                        "chapter": chapter_name
                                    }
                                ))
                        except Exception as e:
                            Logger.get_root_logger("legal_bot_indexer").error(
                                f"Error reading {file_path}: {str(e)}"
                            )

                    yield rows  # Yields list of Documents per CSV file
