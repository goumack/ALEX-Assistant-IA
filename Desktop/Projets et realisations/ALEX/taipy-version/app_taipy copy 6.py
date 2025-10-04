"""
ALEX - Version simplifiée avec interface Flask moderne
Version entreprise élégante et professionnelle
"""
import os
from flask import Flask, render_template_string, request, jsonify
import requests
import json
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import logging
import threading
import time
import hashlib
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger la configuration
load_dotenv()

class ALEXProConfig:
    """Configuration ALEX"""
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollamaaccel-chatbotaccel.apps.senum.heritage.africa")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral:7b")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    WATCH_DIRECTORY = os.getenv("WATCH_DIRECTORY", "./documents")  # Répertoire à surveiller
    SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md', '.json', '.csv', '.odt']

class DocumentWatcherHandler(FileSystemEventHandler):
    """Gestionnaire de surveillance automatique en arrière-plan"""
    
    def __init__(self, alex_client):
        self.alex_client = alex_client
        self.processing_queue = []
        self.last_processed = {}
        super().__init__()
    
    def on_created(self, event):
        """Nouveau fichier créé - Indexation automatique en arrière-plan"""
        if not event.is_directory:
            file_path = event.src_path
            
            # Ignorer les fichiers temporaires
            if Path(file_path).name.startswith(('~$', '.')):
                return
                
            logger.info(f"� [AUTO] Nouveau fichier détecté: {Path(file_path).name}")
            
            # Traitement asynchrone en arrière-plan
            import threading
            def delayed_process():
                try:
                    time.sleep(2)  # Attendre que le fichier soit complètement écrit
                    if self.alex_client.is_supported_file(file_path):
                        self.alex_client.process_new_file_background(file_path)
                        logger.info(f" [AUTO] Fichier indexé automatiquement: {Path(file_path).name}")
                    else:
                        logger.debug(f"⏭ [AUTO] Fichier ignoré (format non supporté): {Path(file_path).name}")
                except Exception as e:
                    logger.error(f" [AUTO] Erreur indexation automatique {Path(file_path).name}: {e}")
            
            # Lancer en thread séparé pour ne pas bloquer le système
            thread = threading.Thread(target=delayed_process, daemon=True)
            thread.start()
    
    def on_modified(self, event):
        """Fichier modifié - Réindexation automatique si nécessaire"""
        if not event.is_directory:
            file_path = event.src_path
            
            # Ignorer les fichiers temporaires et éviter les doublons rapides
            if Path(file_path).name.startswith(('~$', '.')):
                return
                
            # Éviter le traitement en boucle (limitation par temps)
            current_time = time.time()
            if file_path in self.last_processed:
                if current_time - self.last_processed[file_path] < 5:  # 5 secondes minimum
                    return
            
            self.last_processed[file_path] = current_time
            logger.info(f"[AUTO] Modification détectée: {Path(file_path).name}")
            
            # Traitement asynchrone en arrière-plan
            import threading
            def delayed_reprocess():
                try:
                    time.sleep(1)  # Attendre la fin de l'écriture
                    if self.alex_client.is_supported_file(file_path):
                        self.alex_client.process_modified_file_background(file_path)
                        logger.info(f" [AUTO] Fichier réindexé automatiquement: {Path(file_path).name}")
                except Exception as e:
                    logger.error(f" [AUTO] Erreur réindexation automatique {Path(file_path).name}: {e}")
            
            # Lancer en thread séparé
            thread = threading.Thread(target=delayed_reprocess, daemon=True)
            thread.start()

class ALEXProClient:
    """Client ALEX optimisé avec surveillance automatique"""
    
    def __init__(self):
        self.config = ALEXProConfig()
        self.indexed_files = {}  # Cache des fichiers indexés {path: hash}
        self.observer = None  # Référence au watcher
        self.setup_chroma()
        self.setup_watch_directory()
        
        # Démarrer automatiquement la surveillance en arrière-plan
        try:
            self.start_file_watcher()
        except Exception as e:
            logger.warning(f"  Surveillance automatique désactivée: {e}")
        
        logger.info("   ALEX initialisé - Surveillance automatique active")
    
    def setup_chroma(self):
        """Initialise ChromaDB"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.CHROMA_PERSIST_DIRECTORY,
                settings=Settings(allow_reset=True, anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.chroma_client.get_collection("alex_documents")
            except:
                try:
                    self.collection = self.chroma_client.get_collection("alex_pro_docs")
                except:
                    self.collection = self.chroma_client.create_collection(
                        name="alex_pro_docs",
                        metadata={"description": "Documents ALEX"}
                    )
            
            # Charger la liste des fichiers déjà indexés
            self.load_indexed_files_cache()
            
        except Exception as e:
            logger.error(f"Erreur ChromaDB: {e}")
            self.collection = None
    
    def create_vector_store(self):
        """Crée une nouvelle collection ChromaDB"""
        try:
            # Supprimer l'ancienne collection si elle existe
            try:
                self.chroma_client.delete_collection("alex_documents")
            except:
                pass
            
            # Créer une nouvelle collection
            collection = self.chroma_client.create_collection(
                name="alex_documents",
                metadata={"hnsw:space": "cosine", "description": "Documents ALEX"}
            )
            self.collection = collection
            return collection
        except Exception as e:
            logger.error(f"Erreur création collection: {e}")
            return self.collection
    
    def setup_watch_directory(self):
        """Configure le répertoire à surveiller"""
        self.watch_dir = Path(self.config.WATCH_DIRECTORY)
        self.watch_dir.mkdir(exist_ok=True)
        logger.info(f"   Répertoire surveillé: {self.watch_dir.absolute()}")
    
    def start_file_watcher(self):
        """Démarre la surveillance automatique du répertoire"""
        try:
            if not self.watch_dir.exists():
                logger.warning(f"  Répertoire de surveillance introuvable: {self.watch_dir}")
                return False
                
            # Arrêter l'ancien observer s'il existe
            if self.observer and self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
            
            # Créer et démarrer le nouvel observer
            from watchdog.observers import Observer
            self.observer = Observer()
            handler = DocumentWatcherHandler(self)
            self.observer.schedule(handler, str(self.watch_dir), recursive=True)
            self.observer.daemon = True  # Thread daemon pour ne pas bloquer l'arrêt
            self.observer.start()
            
            logger.info(f"   [AUTO] Surveillance automatique active: {self.watch_dir}")
            logger.info("   [AUTO] Les nouveaux fichiers seront indexés automatiquement en arrière-plan")
            
            # Scan initial en mode intelligent (respecte le cache)
            import threading
            def initial_scan():
                try:
                    time.sleep(1)  # Petite pause pour laisser le système s'initialiser
                    self.scan_existing_files()
                except Exception as e:
                    logger.warning(f"  [AUTO] Scan initial différé: {e}")
            
            # Scan initial en arrière-plan
            scan_thread = threading.Thread(target=initial_scan, daemon=True)
            scan_thread.start()
            
            return True
            
        except Exception as e:
            logger.warning(f"  Impossible de démarrer la surveillance automatique: {e}")
            logger.info("📚 Fonctionnement en mode manuel - utilisez les boutons pour indexer")
            self.observer = None
            return False
    
    def load_indexed_files_cache(self):
        """Charge le cache des fichiers indexés depuis ChromaDB"""
        try:
            if self.collection:
                # Récupérer tous les documents avec leurs métadonnées
                results = self.collection.get(include=['metadatas'])
                if results and results['metadatas']:
                    for metadata in results['metadatas']:
                        if metadata and 'file_path' in metadata and 'file_hash' in metadata:
                            self.indexed_files[metadata['file_path']] = metadata['file_hash']
                logger.info(f"📚 Cache chargé: {len(self.indexed_files)} fichiers indexés")
        except Exception as e:
            logger.error(f"Erreur chargement cache: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """Calcule le hash MD5 d'un fichier"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5()
                for chunk in iter(lambda: f.read(4096), b""):
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except Exception as e:
            logger.error(f"Erreur calcul hash pour {file_path}: {e}")
            return ""
    
    def is_supported_file(self, file_path: str) -> bool:
        """Vérifie si le fichier est supporté et pas temporaire"""
        file_name = Path(file_path).name
        
        # Ignorer les fichiers temporaires
        if file_name.startswith('~$') or file_name.startswith('.'):
            return False
            
        return Path(file_path).suffix.lower() in self.config.SUPPORTED_EXTENSIONS
    
    def is_file_already_indexed(self, file_path: str) -> bool:
        """Vérifie si le fichier est déjà indexé (même contenu)"""
        if file_path not in self.indexed_files:
            return False
        
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.indexed_files.get(file_path, "")
        
        return current_hash == stored_hash and current_hash != ""
    
    def read_file_content(self, file_path: str) -> str:
        """Lit le contenu d'un fichier"""
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.txt' or file_ext == '.md':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2, ensure_ascii=False)
            
            elif file_ext == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path)
                return df.to_string()
            
            elif file_ext == '.odt':
                try:
                    # Essayer d'extraire le texte des fichiers ODT
                    import zipfile
                    import xml.etree.ElementTree as ET
                    
                    with zipfile.ZipFile(file_path, 'r') as zip_file:
                        if 'content.xml' in zip_file.namelist():
                            content_xml = zip_file.read('content.xml')
                            root = ET.fromstring(content_xml)
                            
                            # Extraire tout le texte
                            text_content = []
                            for elem in root.iter():
                                if elem.text:
                                    text_content.append(elem.text.strip())
                                if elem.tail:
                                    text_content.append(elem.tail.strip())
                            
                            return ' '.join(filter(None, text_content))
                    return ""
                except Exception as odt_error:
                    logger.warning(f"Erreur lecture ODT {file_path}: {odt_error}")
                    return ""
            
            elif file_ext == '.pdf':
                try:
                    # Extraction PDF avec PyPDF2 ou pdfplumber
                    try:
                        import PyPDF2
                        with open(file_path, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)
                            text_content = []
                            for page in pdf_reader.pages:
                                text_content.append(page.extract_text())
                            return '\n'.join(text_content)
                    except ImportError:
                        try:
                            import pdfplumber
                            with pdfplumber.open(file_path) as pdf:
                                text_content = []
                                for page in pdf.pages:
                                    text_content.append(page.extract_text() or '')
                                return '\n'.join(text_content)
                        except ImportError:
                            logger.warning(f"📄 PyPDF2 et pdfplumber non installés pour: {file_path}")
                            return f"Fichier PDF détecté: {Path(file_path).name} - Contenu non extractible"
                except Exception as pdf_error:
                    logger.error(f"Erreur extraction PDF {file_path}: {pdf_error}")
                    # Retourner au moins les métadonnées du fichier
                    return f"Document PDF: {Path(file_path).name} - Fichier détecté mais extraction échouée. Contient probablement des informations sur la détection de fraude."
            
            elif file_ext == '.docx':
                try:
                    import docx
                    doc = docx.Document(file_path)
                    text_content = []
                    for paragraph in doc.paragraphs:
                        text_content.append(paragraph.text)
                    return '\n'.join(text_content)
                except ImportError:
                    logger.warning(f"📄 python-docx non installé pour: {file_path}")
                    return f"Fichier DOCX détecté: {Path(file_path).name} - Contenu non extractible"
                except Exception as docx_error:
                    logger.error(f"Erreur extraction DOCX {file_path}: {docx_error}")
                    return f"Fichier DOCX: {Path(file_path).name}"
            
            # Pour d'autres types de fichiers, essayer la lecture basique
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Erreur lecture fichier {file_path}: {e}")
            return ""
    
    def process_new_file(self, file_path: str):
        """Traite un nouveau fichier"""
        if not self.is_supported_file(file_path):
            logger.info(f"⏭️ Type de fichier non supporté: {file_path}")
            return
        
        if self.is_file_already_indexed(file_path):
            logger.info(f"✅ Fichier déjà indexé: {Path(file_path).name}")
            return
        
        self.index_file(file_path)
    
    def process_new_file_background(self, file_path: str):
        """Traite un nouveau fichier en arrière-plan (ne bloque pas le chatbot)"""
        try:
            if not self.is_supported_file(file_path):
                return
            
            if self.is_file_already_indexed(file_path):
                logger.debug(f"⏭️ [AUTO] Fichier déjà indexé: {Path(file_path).name}")
                return
            
            logger.info(f"🔄 [AUTO] Indexation en arrière-plan: {Path(file_path).name}")
            self.index_file(file_path)
            
        except Exception as e:
            logger.error(f"  [AUTO] Erreur traitement nouveau fichier {file_path}: {e}")
    
    def process_modified_file_background(self, file_path: str):
        """Traite un fichier modifié en arrière-plan"""
        try:
            if not self.is_supported_file(file_path):
                return
            
            current_hash = self.get_file_hash(file_path)
            stored_hash = self.indexed_files.get(file_path, "")
            
            if current_hash != stored_hash:
                logger.info(f"🔄 [AUTO] Réindexation automatique: {Path(file_path).name}")
                # Supprimer l'ancienne version
                self.remove_file_from_index(file_path)
                # Réindexer
                self.index_file(file_path)
            else:
                logger.debug(f"⏭️ [AUTO] Fichier inchangé: {Path(file_path).name}")
                
        except Exception as e:
            logger.error(f"  [AUTO] Erreur traitement fichier modifié {file_path}: {e}")
    
    def process_modified_file(self, file_path: str):
        """Traite un fichier modifié (version manuelle)"""
        if not self.is_supported_file(file_path):
            return
        
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.indexed_files.get(file_path, "")
        
        if current_hash != stored_hash:
            logger.info(f"🔄 Réindexation du fichier modifié: {Path(file_path).name}")
            # Supprimer l'ancienne version
            self.remove_file_from_index(file_path)
            # Réindexer
            self.index_file(file_path)
    
    def index_file(self, file_path: str):
        """Indexe un fichier dans ChromaDB avec optimisations"""
        try:
            content = self.read_file_content(file_path)
            if not content.strip():
                logger.warning(f"  Fichier vide: {file_path}")
                return
            
            # Découper le contenu en chunks plus grands
            chunks = self.chunk_text(content)
            if not chunks:
                return
            
            logger.info(f"🔄 Indexation de {Path(file_path).name} ({len(chunks)} chunks)")
            
            # Générer les embeddings en batch pour optimiser
            embeddings = []
            valid_chunks = []
            
            # Traitement par petits groupes pour éviter les timeouts
            batch_size = 3
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                
                for chunk in batch_chunks:
                    embedding = self.generate_embeddings(chunk)
                    if embedding:
                        embeddings.append(embedding)
                        valid_chunks.append(chunk)
                
                # Petit délai entre les batches pour éviter de surcharger Ollama
                if i + batch_size < len(chunks):
                    time.sleep(0.1)
            
            if not embeddings:
                logger.error(f"  Impossible de générer les embeddings pour: {file_path}")
                return
            
            # Ajouter à ChromaDB
            file_hash = self.get_file_hash(file_path)
            file_name = Path(file_path).name
            
            ids = [f"{file_name}_{i}_{file_hash[:8]}" for i in range(len(valid_chunks))]
            metadatas = [
                {
                    "file_path": file_path,
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "chunk_index": i,
                    "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                for i in range(len(valid_chunks))
            ]
            
            self.collection.add(
                embeddings=embeddings,
                documents=valid_chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            # Mettre à jour le cache
            self.indexed_files[file_path] = file_hash
            
            logger.info(f"✅ Fichier indexé: {file_name} ({len(valid_chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"  Erreur indexation {file_path}: {e}")
    
    def remove_file_from_index(self, file_path: str):
        """Supprime un fichier de l'index"""
        try:
            # Trouver tous les documents de ce fichier
            results = self.collection.get(
                where={"file_path": file_path},
                include=['metadatas']
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Ancien index supprimé pour: {Path(file_path).name}")
                
        except Exception as e:
            logger.error(f"Erreur suppression index: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 100) -> List[str]:
        """Découpe le texte en chunks plus grands pour optimiser l'indexation"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Essayer de couper à un point naturel (phrase)
            if end < len(text):
                # Chercher le dernier point ou saut de ligne
                last_sentence = max(
                    text.rfind('.', start, end),
                    text.rfind('\n', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end)
                )
                
                if last_sentence > start + 200:  # Au moins 200 caractères
                    end = last_sentence + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def scan_existing_files(self):
        """Scanne les fichiers existants au démarrage avec optimisations"""
        logger.info("� Scan optimisé des fichiers existants...")
        
        try:
            import concurrent.futures
            
            # Collecter tous les fichiers et classifier
            files_to_index = []
            already_indexed = []
            
            for file_path in self.watch_dir.rglob('*'):
                if file_path.is_file() and self.is_supported_file(str(file_path)):
                    if self.is_file_already_indexed(str(file_path)):
                        already_indexed.append(str(file_path))
                    else:
                        files_to_index.append(str(file_path))
            
            # Afficher le statut
            if already_indexed:
                logger.info(f"⏭️ {len(already_indexed)} fichiers déjà indexés (ignorés):")
                for file_path in already_indexed:
                    logger.info(f"   ⏭️ {Path(file_path).name}")
            
            if not files_to_index:
                logger.info("✅ Tous les fichiers sont déjà indexés - Aucun nouveau fichier à traiter")
                return
            
            logger.info(f"📚 Indexation de {len(files_to_index)} fichiers en parallèle...")
            
            # Traitement parallèle avec ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Soumettre tous les fichiers pour traitement
                future_to_file = {
                    executor.submit(self.index_file, file_path): file_path 
                    for file_path in files_to_index
                }
                
                # Collecter les résultats
                count = 0
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        future.result()
                        count += 1
                        if count % 5 == 0:  # Progress indicator
                            logger.info(f"⏳ {count}/{len(files_to_index)} fichiers traités...")
                    except Exception as e:
                        logger.error(f"  Erreur avec {file_path}: {e}")
            
            logger.info(f"🎯 {count} fichiers indexés avec succès!")
            
        except Exception as e:
            logger.warning(f"  Erreur lors du scan initial: {e}")
            logger.info("📚 Fallback: indexation séquentielle")
            # Fallback vers méthode séquentielle
            count = 0
            for file_path in self.watch_dir.rglob('*'):
                if file_path.is_file() and self.is_supported_file(str(file_path)):
                    if not self.is_file_already_indexed(str(file_path)):
                        self.index_file(str(file_path))
                        count += 1

    def generate_embeddings(self, text: str) -> List[float]:
        """Génère des embeddings avec optimisations"""
        try:
            payload = {
                "model": self.config.OLLAMA_EMBEDDING_MODEL,
                "prompt": text
            }
            
            # Timeout réduit et session réutilisable
            if not hasattr(self, '_session'):
                self._session = requests.Session()
                self._session.headers.update({'Connection': 'keep-alive'})
            
            response = self._session.post(
                f"{self.config.OLLAMA_BASE_URL}/api/embeddings",
                json=payload,
                timeout=15  # Timeout augmenté pour OpenShift
            )
            
            if response.status_code == 200:
                return response.json()['embedding']
            return []
        except Exception as e:
            logger.error(f"Erreur embedding: {e}")
            return []
    
    def search_context(self, query: str, limit: int = 5) -> str:
        """Recherche le contexte dans les documents indexés"""
        if not self.collection:
            logger.warning("  Aucune collection ChromaDB disponible")
            return ""
        
        try:
            # Générer embedding de la requête
            query_embedding = self.generate_embeddings(query)
            if not query_embedding:
                logger.warning("  Impossible de générer embedding pour la requête")
                return ""
            
            # Rechercher dans ChromaDB
            logger.info(f"🔍 Recherche dans ChromaDB: {query[:50]}...")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=['documents', 'metadatas']
            )
            
            if results and results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                
                # Construire le contexte avec sources
                context_parts = []
                for i, doc in enumerate(documents):
                    source = ""
                    if metadatas and i < len(metadatas) and metadatas[i]:
                        source_path = metadatas[i].get('source', '')
                        if source_path:
                            source = f"[Source: {Path(source_path).name}]"
                    
                    context_parts.append(f"{source}\n{doc}")
                
                logger.info(f"✅ Contexte trouvé: {len(documents)} documents")
                return "\n\n".join(context_parts)
            else:
                logger.warning("  Aucun contexte trouvé dans ChromaDB")
                return ""
        except Exception as e:
            logger.error(f"  Erreur recherche ChromaDB: {e}")
            return ""
    
    def is_greeting_or_general(self, message: str) -> bool:
        """Détecte si le message est une salutation ou question générale"""
        message_lower = message.lower().strip()
        
        # Debug: log du message pour voir ce qui se passe
        logger.info(f"🔍 Analyse message salutation: '{message_lower}'")
        
        # Mots-clés techniques qui indiquent une question technique (pas une salutation)
        technical_keywords = [
            'deployer', 'déployer', 'deployment', 'openshift', 'kubernetes', 'docker',
            'modele', 'modèle', 'model', 'ai', 'ia', 'machine learning', 'ml',
            'comment faire', 'procédure', 'étapes', 'installation', 'configuration',
            'serveur', 'base de données', 'api', 'endpoint', 'framework'
        ]
        
        # Si le message contient des mots techniques, ce n'est PAS une salutation
        for tech_word in technical_keywords:
            if tech_word in message_lower:
                logger.info(f"❌ Mot technique détecté: '{tech_word}' - Pas une salutation")
                return False
        
        # Mots-clés de salutation
        greeting_words = [
            'salut', 'bonjour', 'bonsoir', 'hello', 'hi', 'hey', 
            'coucou', 'yo', 'wesh'
        ]
        
        # Expressions de politesse (uniquement si pas de contenu technique)
        polite_expressions = [
            'ça va', 'comment ça va', 'comment allez-vous', 'comment allez vous',
            'comment vous allez', 'ça roule', 'quoi de neuf', 'comment tu vas'
        ]
        
        # Questions générales sur ALEX (courtes et sans technique)
        general_questions = [
            'qui es-tu', 'que fais-tu', 'aide', 'help', 
            'présente-toi', 'tu es qui', 'c\'est quoi'
        ]
        
        # Vérifier si le message contient UNIQUEMENT des mots de salutation
        for word in greeting_words:
            if word in message_lower:
                # Vérifier que c'est bien une salutation simple (pas "bonjour, comment déployer...")
                if len(message_lower.split()) <= 4:  # Maximum 4 mots pour une salutation
                    logger.info(f"✅ Salutation simple détectée: '{word}'")
                    return True
            
        # Vérifier les expressions de politesse COURTES
        for expr in polite_expressions:
            if expr in message_lower and len(message_lower) <= 20:  # Expressions courtes seulement
                logger.info(f"✅ Expression de politesse courte détectée: '{expr}'")
                return True
            
        # Vérifier si c'est une question générale sur ALEX
        for q in general_questions:
            if q in message_lower and len(message_lower) <= 15:
                logger.info(f"✅ Question générale courte détectée: '{q}'")
                return True
        
        logger.info(f"❌ Pas de salutation détectée pour: '{message_lower}'")
        return False
    
    def generate_greeting_response(self, message: str) -> str:
        """Génère une réponse appropriée pour les salutations et questions générales"""
        message_lower = message.lower().strip()
        
        # Réponses aux salutations avec "comment allez-vous" ou similaire
        if any(pattern in message_lower for pattern in ['comment allez', 'comment ça va', 'ça va']):
            return """Bonjour ! Je vais très bien, merci ! Je suis ALEX, votre assistant IA d'Accel Tech.

Je suis là pour vous aider à explorer et analyser vos documents de façon conversationnelle.

Comment m'utiliser :
• Posez-moi des questions sur le contenu de vos documents
• Je peux analyser, résumer et extraire des informations
• J'ai accès aux documents dans votre dossier surveillé

Exemples de questions :
• "Quelles sont les mesures de détection de fraude ?"
• "Résume-moi le document sur la stratégie"
• "Trouve les informations sur les recettes budgétaires"

Et vous, comment puis-je vous aider aujourd'hui ?"""
        
        # Réponses aux salutations simples
        elif any(greeting in message_lower for greeting in ['salut', 'bonjour', 'hello', 'hi', 'hey', 'coucou']):
            return """Salut ! Je suis ALEX, votre assistant IA d'Accel Tech !

Je suis là pour vous aider à explorer et analyser vos documents de façon conversationnelle.

Comment m'utiliser :
• Posez-moi des questions sur le contenu de vos documents
• Je peux analyser, résumer et extraire des informations
• J'ai accès aux documents dans votre dossier surveillé

Exemples de questions :
• "Quelles sont les mesures de détection de fraude ?"
• "Résume-moi le document sur la stratégie"
• "Trouve les informations sur les recettes budgétaires"

N'hésitez pas à me poser vos questions !"""

        # Questions sur ALEX
        elif any(q in message_lower for q in ['qui es-tu', 'présente-toi', 'tu es qui']):
            return """Je suis ALEX (Assistant Learning and eXpert)

Développé par Accel Tech, je suis un assistant IA spécialisé dans l'analyse de documents avec la technologie RAG (Retrieval-Augmented Generation).

Mes capacités :
• Analyse et recherche dans vos documents PDF, DOCX, TXT, ODT
• Réponses contextuelles basées sur vos fichiers
• Surveillance automatique des nouveaux documents
• Interface moderne et intuitive

Ma technologie :
• Modèle Mistral 7B pour la génération de réponses
• ChromaDB pour la recherche vectorielle
• Embeddings Nomic pour la compréhension sémantique

Posez-moi des questions spécifiques sur vos documents !"""

        # Questions d'aide
        elif any(q in message_lower for q in ['aide', 'help', 'comment']):
            return """Guide d'utilisation d'ALEX

Comment poser des questions :
• Soyez spécifique : "Quels sont les indicateurs de fraude ?"
• Utilisez des mots-clés pertinents de vos documents
• Demandez des analyses : "Résume le chapitre sur..."

Types de recherches possibles :
• Recherche d'informations précises
• Résumés de documents ou sections
• Comparaisons entre différents documents
• Extraction de données chiffrées

Astuces pour de meilleures réponses :
• Utilisez le vocabulaire de vos domaines (finance, juridique, etc.)
• Posez des questions ouvertes pour plus de détails
• Précisez le document si vous en cherchez un en particulier

Essayez maintenant avec une question sur vos documents !"""

        # Réponse par défaut
        else:
            return """Salut ! Je suis ALEX, votre assistant IA personnel.

Posez-moi des questions sur vos documents et je vous aiderai à trouver les informations que vous cherchez !

Exemple: "Quels sont les principaux points abordés dans mes documents ?" """

    def generate_natural_greeting_response(self, message: str) -> str:
        """Génère une réponse naturelle aux salutations en utilisant Mistral directement"""
        try:
            # Prompt pour que Mistral réponde naturellement aux salutations
            greeting_prompt = f"""Tu es ALEX, un assistant chatbot d'Accel Tech.

L'utilisateur te dit: "{message}"

IMPORTANT: Tu es un chatbot spécialisé d'Accel Tech et tu ne réponds QUE aux questions liées à Accel Tech et ses services.

Réponds de façon naturelle et professionnelle:
- Présente-toi comme ALEX, le chatbot assistant d'Accel Tech
- Précise que tu réponds uniquement aux questions concernant Accel Tech
- Mentionne brièvement les services d'Accel Tech (modernisation, innovation, consulting)
- Reste professionnel et concis (maximum 3-4 lignes)
- Invite l'utilisateur à poser des questions sur Accel Tech

Réponse:"""

            payload = {
                "model": self.config.OLLAMA_CHAT_MODEL,
                "prompt": greeting_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Plus de créativité pour les salutations
                    "top_p": 0.9,
                    "max_tokens": 200
                }
            }
            
            response = requests.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                natural_response = response.json()['response']
                logger.info(f"🤖 Réponse naturelle de salutation générée")
                return natural_response.strip()
            else:
                # Fallback vers réponse prédéfinie
                return self.generate_greeting_response(message)
                
        except Exception as e:
            logger.error(f"Erreur génération réponse naturelle: {e}")
            # Fallback vers réponse prédéfinie
            return self.generate_greeting_response(message)

    def chat(self, message: str) -> str:
        """Génère une réponse de chat basée UNIQUEMENT sur les documents indexés"""
        try:
            # Vérifier si c'est une salutation ou question générale
            if self.is_greeting_or_general(message):
                # Pour les salutations, utiliser directement Mistral sans RAG
                return self.generate_natural_greeting_response(message)
            
            # Rechercher le contexte avec plusieurs stratégies
            context = ""
            
            # Première recherche directe
            context1 = self.search_context(message, limit=3)
            
            # Recherche avec mots-clés extraits
            keywords = [word for word in message.split() if len(word) > 3]
            if keywords:
                keyword_query = " ".join(keywords)
                context2 = self.search_context(keyword_query, limit=3)
                
                # Combiner les contextes en évitant les doublons
                combined_context = context1
                for chunk in context2.split('\n\n'):
                    if chunk not in context1:
                        combined_context += f"\n\n{chunk}"
                
                context = combined_context
            else:
                context = context1
            
            # FORCER l'utilisation du contexte des documents
            if context and context.strip():
                # Extraire les mots-clés de la question
                question_keywords = message.lower().split()
                context_lower = context.lower()
                
                # Vérifier si le contexte est pertinent
                keyword_found = any(kw in context_lower for kw in question_keywords if len(kw) > 3)
                
                if keyword_found or "srmt" in context_lower or "stratégie" in context_lower:
                    prompt = f"""INSTRUCTION ABSOLUE: Tu DOIS répondre UNIQUEMENT en utilisant les informations du CONTEXTE ci-dessous.

CONTEXTE DES DOCUMENTS:
{context}

QUESTION: {message}

RÉPONSE: Basez votre réponse EXCLUSIVEMENT sur le contexte ci-dessus. Citez les sources."""
                else:
                    # Si le contexte ne semble pas pertinent, forcer une recherche plus large
                    return f"""  Le contexte trouvé ne semble pas correspondre à votre question sur "{message}".

🔍 Contexte trouvé: {context[:200]}...

💡 Essayez de reformuler votre question ou utilisez des mots-clés plus spécifiques."""
            else:
                return f"""  Cette information n'est pas disponible dans les documents indexés.

   Documents disponibles: DEPLOIEMENT MODELE FRAUDE, Rapport DGD, Documentation technique, etc.

💡 Essayez des questions comme:
- "Comment déployer un modèle de fraude sur OpenShift AI ?"
- "Que dit le rapport DGD ?"
- "Quelle est la documentation technique disponible ?"

🔄 Si vous pensez que cette information devrait être disponible, cliquez sur 'Indexer dossier fraude' pour réindexer."""
            
            # Debug: Logger le prompt et le contexte
            logger.info(f"   PROMPT ENVOYÉ À OLLAMA:")
            logger.info(f"Contexte: {context[:200]}..." if context else "AUCUN CONTEXTE!")
            logger.info(f"Prompt: {prompt[:300]}...")
            
            payload = {
                "model": self.config.OLLAMA_CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Réduire la créativité
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            response = requests.post(
                f"{self.config.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                ollama_response = response.json()['response']
                logger.info(f"   RÉPONSE OLLAMA: {ollama_response[:200]}...")
                return ollama_response
            else:
                return "Désolé, je rencontre un problème technique. Veuillez réessayer."
                
        except Exception as e:
            logger.error(f"Erreur chat: {e}")
            return "Une erreur s'est produite. Veuillez réessayer dans un moment."

# Template HTML ultra moderne et responsif avec effets
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>   ALEX - Assistant IA Professionnel</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #f5f7fa;
            margin: 0;
            padding: 0;
            position: relative;
        }



        /* Bouton flottant d'ouverture */
        .chat-toggle-btn {
            position: fixed;
            bottom: 80px;
            right: 20px;
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #1e40af 100%);
            border: none;
            border-radius: 50%;
            color: white;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(30, 58, 138, 0.3);
            z-index: 1000;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .chat-toggle-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 8px 30px rgba(30, 58, 138, 0.4);
        }

        .chat-toggle-btn.active {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }

        /* Container de chat en widget */
        .chat-widget {
            position: fixed;
            bottom: 150px;
            right: 20px;
            width: 380px;
            height: 600px;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
            z-index: 999;
            display: none;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid rgba(30, 58, 138, 0.1);
            animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(30px) scale(0.9);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        .chat-widget.open {
            display: flex;
        }

        .container {
            display: flex;
            flex-direction: column;
            height: 100%;
            padding: 0;
            margin: 0;
            width: 100%;
            max-width: none;
            background: transparent;
            border-radius: 0;
            box-shadow: none;
            animation: none;
        }



        .chat-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #1e40af 100%);
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 20px 20px 0 0;
            position: relative;
        }

        .chat-header h1 {
            margin: 0;
            font-size: 1.5em;
            font-weight: 600;
        }

        .chat-header p {
            margin: 5px 0 0 0;
            font-size: 0.9em;
            opacity: 0.9;
        }

        .close-chat-btn {
            position: absolute;
            top: 15px;
            right: 15px;
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s ease;
        }

        .close-chat-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: rotate(90deg);
        }

        .chat-container {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
            border-radius: 0;
            border: none;
            box-shadow: none;
            margin: 0;
        }



        /* Scrollbar personnalisée */
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }

        .chat-container::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }

        .chat-container::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #1e3a8a, #3b82f6);
            border-radius: 4px;
            transition: all 0.3s ease;
        }

        .chat-container::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #1e40af, #2563eb);
        }

        .message {
            margin-bottom: 20px;
            padding: 18px 24px;
            border-radius: 18px;
            animation: messageSlideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            backdrop-filter: blur(10px);
            transition: all 0.3s ease;
        }

        .message:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }

        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateY(30px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }

        .user-message {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            margin-left: 20%;
            text-align: right;
            box-shadow: 0 2px 10px rgba(30, 58, 138, 0.2);
            border: none;
        }

        .user-message::after {
            display: none;
        }

        .assistant-message {
            background: white;
            margin-right: 20%;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            color: #333;
            border: 1px solid #e9ecef;
        }

        .assistant-message::after {
            display: none;
        }

        .chat-input-section {
            padding: 20px;
            background: white;
            border-top: 1px solid #e9ecef;
            border-radius: 0 0 20px 20px;
        }

        .input-section {
            display: flex;
            gap: 10px;
            margin: 0;
        }

        .input-wrapper {
            flex: 1;
            position: relative;
        }

        #messageInput {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 14px;
            color: #333;
            outline: none;
            transition: border-color 0.3s ease;
        }

        #messageInput::placeholder {
            color: #6c757d;
        }

        #messageInput:focus {
            border-color: #2563eb;
        }



        .send-btn {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #1e40af 100%);
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.3s ease;
            min-width: 80px;
        }

        .send-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(30, 58, 138, 0.3);
        }

        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }



        .loading {
            display: none;
            text-align: center;
            color: rgba(255, 255, 255, 0.9);
            font-style: italic;
            font-weight: 500;
            margin: 15px 0;
            animation: loadingPulse 2s ease-in-out infinite;
        }

        @keyframes loadingPulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }

        .typing {
            display: inline-block;
            width: 24px;
            height: 24px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: rgba(255, 255, 255, 0.9);
            animation: spin 1s linear infinite;
            margin-right: 10px;
            vertical-align: middle;
        }

        .loading-dots {
            display: inline-block;
            margin-left: 10px;
        }

        .loading-dots::after {
            content: '';
            animation: dots 1.5s steps(4, end) infinite;
        }

        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }



        /* Responsive Design */
        @media (max-width: 768px) {
            .chat-widget {
                width: 95%;
                height: 70vh;
                right: 2.5%;
                left: 2.5%;
                bottom: 140px;
            }
            
            .chat-toggle-btn {
                width: 55px;
                height: 55px;
                font-size: 22px;
                bottom: 60px;
            }
            
            .input-section {
                flex-direction: column;
                gap: 10px;
            }
            
            .send-btn {
                width: 100%;
            }
            
            .message {
                margin-left: 5% !important;
                margin-right: 5% !important;
            }
        }

        @media (max-width: 480px) {
            .chat-widget {
                width: 100%;
                height: 100vh;
                right: 0;
                left: 0;
                bottom: 0;
                border-radius: 0;
            }
            
            .chat-header {
                border-radius: 0;
            }
            
            .chat-input-section {
                border-radius: 0;
            }
            
            .message {
                margin-left: 0 !important;
                margin-right: 0 !important;
                padding: 12px 16px;
            }
        }


    </style>
</head>
<body>
    <!-- Page principale simulée -->
    <div style="padding: 50px; text-align: center; color: #333;">
        <h1 style="color: #1e3a8a; margin-bottom: 20px;">Bienvenue sur notre site</h1>
        <p style="font-size: 18px; color: #666; max-width: 600px; margin: 0 auto;">
            Découvrez nos services et n'hésitez pas à utiliser notre assistant ALEX pour toute question.
        </p>
        <div style="margin-top: 40px; padding: 40px; background: #f8f9fa; border-radius: 15px; max-width: 800px; margin: 40px auto;">
            <h2 style="color: #2563eb;">Services Accel Tech</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h3 style="color: #1e3a8a;">Modernisation</h3>
                    <p>Transformation digitale de votre entreprise</p>
                </div>
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h3 style="color: #1e3a8a;">Innovation</h3>
                    <p>Solutions technologiques avancées</p>
                </div>
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h3 style="color: #1e3a8a;">Consulting</h3>
                    <p>Accompagnement personnalisé</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Bouton flottant pour ouvrir le chat -->
    <button class="chat-toggle-btn" id="chatToggle" onclick="toggleChat()">
        💬
    </button>

    <!-- Widget de chat -->
    <div class="chat-widget" id="chatWidget">
        <div class="container">
            <div class="chat-header">
                <button class="close-chat-btn" onclick="toggleChat()">×</button>
                <h1>ALEX by Accel Tech</h1>
                <p>Modernize. Innovate.</p>
            </div>

        <div class="chat-container" id="chatContainer">
            <div class="message assistant-message">
                Bonjour ! Je suis ALEX, l'assistant IA d'Accel Tech. Modernize. Innovate. Comment puis-je vous accompagner dans votre transformation digitale aujourd'hui ?
            </div>
        </div>

            <div class="loading" id="loading">
                <div class="typing"></div>
                <span>ALEX réfléchit<span class="loading-dots"></span></span>
            </div>

            <div class="chat-input-section">
                <div class="input-section">
                    <input type="text" id="messageInput" placeholder="Posez votre question à ALEX..." onkeypress="checkEnter(event)">
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                        Envoyer
                    </button>
                </div>
            </div>
        </div>



    </div>

    <script>
        // Gestion du widget de chat
        function toggleChat() {
            const widget = document.getElementById('chatWidget');
            const toggleBtn = document.getElementById('chatToggle');
            
            if (widget.classList.contains('open')) {
                // Fermer le chat
                widget.classList.remove('open');
                toggleBtn.innerHTML = '💬';
                toggleBtn.classList.remove('active');
                toggleBtn.style.transform = 'scale(1)';
            } else {
                // Ouvrir le chat
                widget.classList.add('open');
                toggleBtn.innerHTML = '✕';
                toggleBtn.classList.add('active');
                toggleBtn.style.transform = 'scale(0.9)';
                
                // Focus sur l'input
                setTimeout(() => {
                    document.getElementById('messageInput').focus();
                }, 300);
            }
        }

        function checkEnter(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;

            const chatContainer = document.getElementById('chatContainer');
            const sendBtn = document.getElementById('sendBtn');
            const loading = document.getElementById('loading');

            // Ajouter le message utilisateur
            const userMessage = document.createElement('div');
            userMessage.className = 'message user-message';
            userMessage.textContent = message;
            chatContainer.appendChild(userMessage);

            // Vider l'input et désactiver le bouton
            input.value = '';
            sendBtn.disabled = true;
            loading.style.display = 'block';
            
            // Faire défiler vers le bas
            chatContainer.scrollTop = chatContainer.scrollHeight;

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();

                // Ajouter la réponse de l'assistant avec effet de frappe
                const assistantMessage = document.createElement('div');
                assistantMessage.className = 'message assistant-message';
                assistantMessage.innerHTML = '';
                chatContainer.appendChild(assistantMessage);
                
                // Effet de frappe
                const textSpan = document.createElement('span');
                assistantMessage.appendChild(textSpan);
                typewriterEffect(textSpan, data.response, 20);
                
                // Ajouter les effets de survol
                setTimeout(() => addMessageEffects(), 1000);

            } catch (error) {
                const errorMessage = document.createElement('div');
                errorMessage.className = 'message assistant-message';
                errorMessage.textContent = 'Désolé, une erreur s\\'est produite. Veuillez réessayer.';
                errorMessage.style.color = '#e74c3c';
                chatContainer.appendChild(errorMessage);
            }

            // Réactiver le bouton et cacher le loading
            sendBtn.disabled = false;
            loading.style.display = 'none';
            
            // Faire défiler vers le bas
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function clearChat() {
            const chatContainer = document.getElementById('chatContainer');
            
            // Animation de fade out
            chatContainer.style.opacity = '0.5';
            chatContainer.style.transform = 'scale(0.95)';
            
            setTimeout(() => {
                chatContainer.innerHTML = `
                    <div class="message assistant-message">
                        Bonjour ! Je suis ALEX, l'assistant IA d'Accel Tech. Comment puis-je vous accompagner aujourd'hui ?
                    </div>
                `;
                
                // Animation de fade in
                chatContainer.style.opacity = '1';
                chatContainer.style.transform = 'scale(1)';
            }, 200);
        }

        // Effets au survol des messages
        function addMessageEffects() {
            const messages = document.querySelectorAll('.message');
            messages.forEach(message => {
                message.addEventListener('mouseenter', function() {
                    this.style.transform = 'translateY(-2px) scale(1.01)';
                });
                
                message.addEventListener('mouseleave', function() {
                    this.style.transform = 'translateY(0) scale(1)';
                });
            });
        }

        // Effet de frappe pour les réponses
        function typewriterEffect(element, text, speed = 30) {
            element.textContent = '';
            let i = 0;
            const timer = setInterval(() => {
                if (i < text.length) {
                    element.textContent += text.charAt(i);
                    i++;
                } else {
                    clearInterval(timer);
                }
            }, speed);
        }

        // Initialisation au chargement
        window.onload = function() {
            addMessageEffects();
            
            // Animation d'entrée du bouton flottant
            const toggleBtn = document.getElementById('chatToggle');
            toggleBtn.style.opacity = '0';
            toggleBtn.style.transform = 'scale(0.5)';
            
            setTimeout(() => {
                toggleBtn.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
                toggleBtn.style.opacity = '1';
                toggleBtn.style.transform = 'scale(1)';
            }, 500);
        };




    </script>
    
    <!-- Footer Accel Tech -->
    <div style="position: fixed; bottom: 15px; right: 20px; 
                color: rgba(255, 255, 255, 0.7); font-size: 12px; 
                background: rgba(0, 0, 0, 0.3); padding: 8px 15px; 
                border-radius: 20px; backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                z-index: 1000;">
        Powered by <strong>Accel Tech</strong> • Modernize. Innovate.
    </div>
</body>
</html>
"""

# Application Flask
app = Flask(__name__)
alex_client = ALEXProClient()

@app.route('/')
def home():
    """Page d'accueil"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint pour le chat"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'response': 'Veuillez saisir un message.'}), 400
        
        response = alex_client.chat(message)
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Erreur chat endpoint: {e}")
        return jsonify({'response': 'Une erreur s\'est produite.'}), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint pour obtenir le statut de l'indexation"""
    try:
        # Vérifier le statut de la surveillance
        surveillance_status = "Inactive"
        auto_indexing = False
        if alex_client.observer:
            if alex_client.observer.is_alive():
                surveillance_status = "🔄 Active (Auto-indexation ON)"
                auto_indexing = True
            else:
                surveillance_status = "⏸️ Arrêté"
        
        status = {
            'indexed_files_count': len(alex_client.indexed_files),
            'watch_directory': str(alex_client.watch_dir.absolute()),
            'supported_extensions': alex_client.config.SUPPORTED_EXTENSIONS,
            'indexed_files': list(alex_client.indexed_files.keys()),
            'surveillance_status': surveillance_status,
            'auto_indexing': auto_indexing
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Erreur status endpoint: {e}")
        return jsonify({'error': 'Erreur récupération statut'}), 500

@app.route('/force_full_reindex', methods=['POST'])
def force_full_reindex():
    """Force la réindexation complète de TOUS les fichiers (ignore le cache)"""
    try:
        # Diagnostic avant indexation
        logger.info(f"🔍 RÉINDEXATION FORCÉE COMPLÈTE")
        
        # Lister tous les fichiers supportés
        supported_files = []
        for file_path in alex_client.watch_dir.rglob('*'):
            if file_path.is_file() and alex_client.is_supported_file(str(file_path)):
                supported_files.append(str(file_path))
        
        # VIDER COMPLÈTEMENT le cache et ChromaDB
        alex_client.indexed_files.clear()
        try:
            if hasattr(alex_client, 'collection') and alex_client.collection:
                alex_client.create_vector_store()
                logger.info("🗑️ Base vectorielle et cache complètement vidés")
        except Exception as e:
            logger.warning(f"  Erreur vidage: {e}")
        
        # Indexation complète
        alex_client.scan_existing_files()
        
        return jsonify({
            'message': f'Réindexation COMPLÈTE terminée: {len(supported_files)} fichiers retraités',
            'indexed_count': len(alex_client.indexed_files),
            'files_found': len(supported_files),
            'cache_cleared': True
        })
    except Exception as e:
        logger.error(f"Erreur force_full_reindex: {e}")
        return jsonify({'error': 'Erreur réindexation complète'}), 500

@app.route('/reindex', methods=['POST'])
def smart_reindex():
    """Réindexation intelligente (respecte le cache des fichiers déjà indexés)"""
    try:
        # Diagnostic avant indexation
        logger.info(f"🔍 Scan du dossier: {alex_client.config.WATCH_DIRECTORY}")
        
        # Lister tous les fichiers supportés
        supported_files = []
        for file_path in alex_client.watch_dir.rglob('*'):
            if file_path.is_file() and alex_client.is_supported_file(str(file_path)):
                supported_files.append(str(file_path))
        
        logger.info(f"   {len(supported_files)} fichiers supportés trouvés:")
        for file_path in supported_files:
            logger.info(f"   - {Path(file_path).name}")
        
        # Vider le cache ChromaDB complètement
        try:
            if hasattr(alex_client, 'collection') and alex_client.collection:
                alex_client.create_vector_store()
                logger.info("🗑️ Base vectorielle vidée complètement")
            else:
                logger.info("🔄 Création nouvelle base vectorielle")
                alex_client.create_vector_store()
        except Exception as e:
            logger.warning(f"  Erreur vidage base: {e}")
            # Fallback : créer une nouvelle collection
            try:
                alex_client.create_vector_store()
            except Exception as e2:
                logger.error(f"  Erreur création base: {e2}")
        
        # NE PAS vider le cache local - garder la mémoire des fichiers indexés
        # alex_client.indexed_files.clear()  # COMMENTÉ pour éviter réindexation
        
        # Relancer le scan avec respect du cache
        try:
            alex_client.scan_existing_files()
            already_indexed = len([f for f in supported_files if alex_client.is_file_already_indexed(f)])
            newly_indexed = len(alex_client.indexed_files) - already_indexed
            message = f'Scan terminé: {already_indexed} déjà indexés, {newly_indexed} nouveaux fichiers traités'
            logger.info(f"✅ Indexation terminée: {len(alex_client.indexed_files)} fichiers au total")
        except Exception as e:
            logger.error(f"Erreur lors de l'indexation: {e}")
            message = f'Réindexation échouée: Vérifiez la connexion Ollama'
        
        return jsonify({
            'message': message,
            'indexed_count': len(alex_client.indexed_files),
            'files_found': len(supported_files),
            'files_list': [Path(f).name for f in supported_files[:5]]  # Top 5 files
        })
    except Exception as e:
        logger.error(f"Erreur reindex endpoint: {e}")
        return jsonify({'error': 'Erreur réindexation'}), 500

@app.route('/start_indexing', methods=['POST'])
def start_indexing():
    """Démarre l'indexation initiale"""
    try:
        alex_client.scan_existing_files()
        return jsonify({
            'message': 'Indexation démarrée',
            'indexed_count': len(alex_client.indexed_files)
        })
    except Exception as e:
        logger.error(f"Erreur start_indexing: {e}")
        return jsonify({'error': f'Erreur indexation: {str(e)}'}), 500

@app.route('/diagnostic', methods=['GET'])
def diagnostic_files():
    """Diagnostic des fichiers indexés"""
    try:
        # Lister tous les fichiers du dossier
        all_files = []
        supported_files = []
        indexed_files = list(alex_client.indexed_files.keys())
        
        for file_path in alex_client.watch_dir.rglob('*'):
            if file_path.is_file():
                all_files.append(str(file_path))
                if alex_client.is_supported_file(str(file_path)):
                    supported_files.append(str(file_path))
        
        # Compter les éléments dans ChromaDB avec diagnostic
        try:
            collection_count = alex_client.vector_store.count()
            logger.info(f"📊 ChromaDB count: {collection_count}")
        except Exception as e:
            logger.error(f"  Erreur ChromaDB count: {e}")
            collection_count = 0
            
        # Vérifier la collection elle-même
        try:
            # Essayer de récupérer quelques documents pour tester
            test_results = alex_client.vector_store.peek(limit=5)
            actual_chunks = len(test_results.get('documents', []))
            logger.info(f"🔍 Documents réels dans ChromaDB: {actual_chunks}")
            if actual_chunks > collection_count:
                collection_count = actual_chunks
        except Exception as e:
            logger.warning(f"  Erreur peek ChromaDB: {e}")
        
        return jsonify({
            'dossier_surveille': alex_client.config.WATCH_DIRECTORY,
            'fichiers_totaux': len(all_files),
            'fichiers_supportes': len(supported_files),
            'fichiers_indexes': len(indexed_files),
            'chunks_chromadb': collection_count,
            'liste_supportes': [Path(f).name for f in supported_files],
            'liste_indexes': [Path(f).name for f in indexed_files],
            'formats_supportes': ['.pdf', '.txt', '.docx', '.odt']
        })
    except Exception as e:
        logger.error(f"Erreur diagnostic: {e}")
        return jsonify({'error': f'Erreur diagnostic: {str(e)}'}), 500

@app.route('/search_fraude', methods=['POST'])
def search_fraude():
    """Recherche spécifique sur la détection de fraude"""
    try:
        data = request.get_json()
        query = data.get('message', '')
        
        if not query:
            return jsonify({'error': 'Query manquante'}), 400
        
        # Rechercher dans les documents avec focus fraude
        context = alex_client.search_context(f"déploiement fraude modèle détection {query}", limit=5)
        
        if not context.strip():
            return jsonify({
                'response': """🚫 Aucun contexte trouvé sur la détection de fraude. 
                
Les documents sur la fraude semblent ne pas être correctement indexés dans ChromaDB. 
Cliquez sur 'Indexer dossier fraude' pour réindexer tous les documents."""
            })
        
        # Générer réponse avec contexte fraude
        prompt = f"""Tu es ALEX, un expert en détection de fraude et déploiement de modèles ML.

Contexte des documents de fraude :
{context}

Question de l'utilisateur : {query}

Réponds de manière détaillée en te basant UNIQUEMENT sur le contexte fourni. 
Mentionne les documents sources quand c'est pertinent."""

        response = alex_client.generate_response(prompt)
        
        return jsonify({
            'response': response,
            'context_found': len(context) > 100,
            'source': 'Documents de détection de fraude'
        })
        
    except Exception as e:
        logger.error(f"Erreur search_fraude: {e}")
        return jsonify({'error': f'Erreur recherche fraude: {str(e)}'}), 500

@app.route('/debug_context', methods=['POST'])
def debug_context():
    """Debug endpoint pour voir le contexte réel"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query manquante'}), 400
        
        # Recherche avec debug
        context = alex_client.search_context(query, limit=3)
        
        # Récupérer aussi quelques documents de ChromaDB
        try:
            sample_docs = alex_client.collection.peek(limit=3)
            sample_content = sample_docs.get('documents', [])[:3] if sample_docs else []
        except:
            sample_content = []
        
        return jsonify({
            'query': query,
            'context_found': context,
            'context_length': len(context) if context else 0,
            'sample_documents': sample_content,
            'collection_count': alex_client.collection.count() if alex_client.collection else 0
        })
        
    except Exception as e:
        logger.error(f"Erreur debug_context: {e}")
        return jsonify({'error': f'Erreur: {str(e)}'}), 500

def cleanup():
    """Nettoyage à la fermeture"""
    try:
        if hasattr(alex_client, 'observer') and alex_client.observer:
            alex_client.observer.stop()
            alex_client.observer.join()
            logger.info("🛑 Surveillance arrêtée proprement")
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt: {e}")

def app_taipy():
    """Lance l'application ALEX"""
    print("   Démarrage d'ALEX...")
    print("=" * 50)
    print(f"🔗 URL Ollama: {ALEXProConfig.OLLAMA_BASE_URL}")
    print(f"   Modèle: {ALEXProConfig.OLLAMA_CHAT_MODEL}")
    print(f"   Répertoire surveillé: {ALEXProConfig.WATCH_DIRECTORY}")
    print("🌐 Démarrage de l'interface...")
    
    try:
        app.run(
            host="0.0.0.0",
            port=8504,
            debug=False
        )
    except KeyboardInterrupt:
        print("\n👋 Arrêt d'ALEX...")
        cleanup()
    except Exception as e:
        print(f"  Erreur: {e}")
        cleanup()

if __name__ == "__main__":
    import atexit
    atexit.register(cleanup)
    app_taipy()