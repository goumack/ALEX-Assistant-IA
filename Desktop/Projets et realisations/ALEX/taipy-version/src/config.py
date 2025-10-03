"""
Configuration et utilitaires pour ALEX RAG Chatbot
"""
import os
from dotenv import load_dotenv
from typing import Optional

# Charger les variables d'environnement
load_dotenv()

class Config:
    """Configuration centrale pour ALEX"""
    
    # Configuration Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "mistral")
    OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    
    # Configuration ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # Configuration Application
    APP_TITLE: str = os.getenv("APP_TITLE", "ALEX - Assistant IA avec RAG")
    APP_DESCRIPTION: str = os.getenv("APP_DESCRIPTION", "Chatbot intelligent avec recherche dans la base de connaissances")
    
    # Limites
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    @classmethod
    def validate_config(cls) -> bool:
        """Valide la configuration"""
        required_vars = [
            cls.OLLAMA_BASE_URL,
            cls.OLLAMA_CHAT_MODEL,
            cls.OLLAMA_EMBEDDING_MODEL
        ]
        return all(var for var in required_vars)

def get_config() -> Config:
    """Retourne l'instance de configuration"""
    return Config()