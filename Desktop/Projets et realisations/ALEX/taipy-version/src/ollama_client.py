"""
Client Ollama pour ALEX
Gestion de la connexion avec Ollama sur OpenShift
"""
import requests
import json
from typing import List, Dict, Any, Optional
import logging
from src.config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient:
    """Client pour interagir avec Ollama"""
    
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config.OLLAMA_BASE_URL.rstrip('/')
        self.chat_model = self.config.OLLAMA_CHAT_MODEL
        self.embedding_model = self.config.OLLAMA_EMBEDDING_MODEL
        
    def test_connection(self) -> bool:
        """Test la connexion avec Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Erreur de connexion Ollama: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Liste les modèles disponibles"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                return response.json().get('models', [])
            return []
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des modèles: {e}")
            return []
    
    def check_model_exists(self, model_name: str) -> bool:
        """Vérifie si un modèle existe"""
        models = self.list_models()
        return any(model['name'].startswith(model_name) for model in models)
    
    def pull_model(self, model_name: str) -> bool:
        """Télécharge un modèle"""
        try:
            logger.info(f"Téléchargement du modèle {model_name}...")
            
            payload = {"name": model_name}
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                stream=True,
                timeout=300
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'status' in data:
                                logger.info(f"Status: {data['status']}")
                            if data.get('status') == 'success':
                                return True
                        except json.JSONDecodeError:
                            continue
                return True
            else:
                logger.error(f"Erreur lors du téléchargement: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement du modèle {model_name}: {e}")
            return False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Génère des embeddings pour une liste de textes"""
        embeddings = []
        
        for text in texts:
            try:
                payload = {
                    "model": self.embedding_model,
                    "prompt": text
                }
                
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    embeddings.append(data['embedding'])
                else:
                    logger.error(f"Erreur embedding: {response.status_code}")
                    embeddings.append([0.0] * 384)  # Embedding par défaut
                    
            except Exception as e:
                logger.error(f"Erreur lors de la génération d'embedding: {e}")
                embeddings.append([0.0] * 384)
                
        return embeddings
    
    def chat_completion(self, messages: List[Dict[str, str]], context: Optional[str] = None) -> str:
        """Génère une réponse de chat"""
        try:
            # Construire le prompt complet
            prompt = ""
            
            # Ajouter le contexte RAG si fourni
            if context:
                prompt += f"""Tu es ALEX, un assistant IA intelligent. 
Utilise le contexte suivant pour répondre à la question de l'utilisateur.
Si l'information n'est pas dans le contexte, dis-le clairement.

Contexte: {context}

"""
            
            # Construire le prompt à partir des messages
            for message in messages:
                if message['role'] == 'user':
                    prompt += f"Utilisateur: {message['content']}\n"
                elif message['role'] == 'assistant':
                    prompt += f"Assistant: {message['content']}\n"
                elif message['role'] == 'system':
                    prompt = message['content'] + "\n" + prompt
            
            prompt += "Assistant: "
            
            payload = {
                "model": self.chat_model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return data['response']
            else:
                return f"Erreur de génération: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Erreur lors du chat: {e}")
            return f"Erreur: {str(e)}"
    
    def setup_embedding_model(self) -> bool:
        """Configure le modèle d'embedding"""
        logger.info("Configuration du modèle d'embedding...")
        
        if not self.check_model_exists(self.embedding_model):
            logger.info(f"Le modèle {self.embedding_model} n'est pas disponible. Téléchargement...")
            return self.pull_model(self.embedding_model)
        else:
            logger.info(f"Le modèle {self.embedding_model} est déjà disponible.")
            return True