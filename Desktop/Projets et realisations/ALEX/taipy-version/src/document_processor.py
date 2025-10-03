"""
Gestionnaire de documents pour ALEX RAG
Traitement et indexation des documents
"""
import os
import io
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import pypdf
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import get_config
from src.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processeur de documents pour le RAG"""
    
    def __init__(self):
        self.config = get_config()
        self.ollama_client = OllamaClient()
        
        # Configuration ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        # Collection ChromaDB pour les documents
        try:
            self.collection = self.chroma_client.get_collection("alex_documents")
        except:
            self.collection = self.chroma_client.create_collection(
                name="alex_documents",
                metadata={"description": "Documents pour ALEX RAG"}
            )
        
        # Text splitter pour découper les documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extrait le texte d'un fichier"""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
                    
            elif file_extension == '.pdf':
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = pypdf.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text
                
            elif file_extension in ['.docx', '.doc']:
                return docx2txt.process(file_path)
                
            else:
                raise ValueError(f"Type de fichier non supporté: {file_extension}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du texte de {file_path}: {e}")
            return ""
    
    def extract_text_from_uploaded_file(self, uploaded_file) -> str:
        """Extrait le texte d'un fichier uploadé (Streamlit)"""
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == '.txt':
                return str(uploaded_file.read(), "utf-8")
                
            elif file_extension == '.pdf':
                text = ""
                pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
                
            elif file_extension in ['.docx', '.doc']:
                return docx2txt.process(io.BytesIO(uploaded_file.read()))
                
            else:
                raise ValueError(f"Type de fichier non supporté: {file_extension}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du texte: {e}")
            return ""
    
    def process_and_store_document(self, file_path: str, filename: str) -> bool:
        """Traite et stocke un document dans ChromaDB"""
        try:
            logger.info(f"Traitement du document: {filename}")
            
            # Extraire le texte
            text = self.extract_text_from_file(file_path)
            if not text.strip():
                logger.warning(f"Aucun texte extrait de {filename}")
                return False
            
            # Découper le texte en chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Document découpé en {len(chunks)} chunks")
            
            # Générer les embeddings
            embeddings = self.ollama_client.generate_embeddings(chunks)
            
            # Préparer les métadonnées
            metadatas = [
                {
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            # Générer les IDs uniques
            ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
            
            # Stocker dans ChromaDB
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Document {filename} traité et stocké avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du document {filename}: {e}")
            return False
    
    def process_and_store_uploaded_file(self, uploaded_file) -> bool:
        """Traite et stocke un fichier uploadé"""
        try:
            logger.info(f"Traitement du fichier uploadé: {uploaded_file.name}")
            
            # Extraire le texte
            text = self.extract_text_from_uploaded_file(uploaded_file)
            if not text.strip():
                logger.warning(f"Aucun texte extrait de {uploaded_file.name}")
                return False
            
            # Découper le texte en chunks
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Document découpé en {len(chunks)} chunks")
            
            # Générer les embeddings
            embeddings = self.ollama_client.generate_embeddings(chunks)
            
            # Préparer les métadonnées
            metadatas = [
                {
                    "filename": uploaded_file.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            # Générer les IDs uniques
            ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]
            
            # Stocker dans ChromaDB
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Fichier {uploaded_file.name} traité et stocké avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier {uploaded_file.name}: {e}")
            return False
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Recherche dans les documents stockés"""
        try:
            # Générer l'embedding de la requête
            query_embedding = self.ollama_client.generate_embeddings([query])[0]
            
            # Rechercher dans ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Formatter les résultats
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {e}")
            return []
    
    def get_stored_documents(self) -> List[str]:
        """Récupère la liste des documents stockés"""
        try:
            # Obtenir toutes les métadonnées
            results = self.collection.get(include=['metadatas'])
            
            # Extraire les noms de fichiers uniques
            filenames = set()
            for metadata in results['metadatas']:
                if 'filename' in metadata:
                    filenames.add(metadata['filename'])
            
            return list(filenames)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des documents: {e}")
            return []
    
    def delete_document(self, filename: str) -> bool:
        """Supprime un document de la base"""
        try:
            # Trouver tous les chunks de ce document
            results = self.collection.get(
                where={"filename": filename},
                include=['ids']
            )
            
            if results['ids']:
                # Supprimer tous les chunks
                self.collection.delete(ids=results['ids'])
                logger.info(f"Document {filename} supprimé avec succès")
                return True
            else:
                logger.warning(f"Document {filename} non trouvé")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du document {filename}: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """Supprime tous les documents de la base"""
        try:
            # Réinitialiser la collection
            self.chroma_client.delete_collection("alex_documents")
            self.collection = self.chroma_client.create_collection(
                name="alex_documents",
                metadata={"description": "Documents pour ALEX RAG"}
            )
            logger.info("Tous les documents ont été supprimés")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de tous les documents: {e}")
            return False