#nlp/knowledge_qa.py
import os
import json
import math
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import time
import traceback
import sys
from .knowledge_graph_entity_linker import KnowledgeGraphEntityLinker
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from config import config as _cfg
from .intents import coerce_intent_by_entities
from .intent_dt import SimpleIntentDecisionTable

# Ensure project root is in path for module imports
try:
    # Determine project root dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Assumes this file is in a subdirectory of root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        # Use logger after basicConfig potentially runs
        # logger.debug(f"Added project root to path: {project_root}")

    # --- Use updated processors ---
    from .bert_processor import BertProcessor
    from .flan_t5_processor import FlanT5Processor
    try:
        from .hydra_client import HydraClient
    except Exception:
        HydraClient = None
    from database.neo4j_connector import Neo4jConnection
    from config import config # Import the global config instance
    from dotenv import load_dotenv

    # Define logger early, before potential import errors in dependencies
    logger = logging.getLogger(__name__) # Use __name__ for better log tracking

except ImportError as e:
    # Define a fallback logger if logger is not initialized
    fallback_logger = logging.getLogger("fallback_logger")
    fallback_logger.setLevel(logging.CRITICAL)
    if not fallback_logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        fallback_logger.addHandler(handler)
    fallback_logger.critical(f"CRITICAL: Failed to import necessary modules: {e}. Ensure all dependencies are installed and PYTHONPATH is correct. Project root: {project_root}", exc_info=True)
    sys.exit(1)
except Exception as e:
    # Define fallback logger if needed
    fallback_logger = logging.getLogger("fallback_logger")
    fallback_logger.setLevel(logging.CRITICAL)
    if not fallback_logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        fallback_logger.addHandler(handler)
    fallback_logger.critical(f"CRITICAL: An unexpected error occurred during imports: {e}", exc_info=True)
    sys.exit(1)

# Setup logging (Now uses config object for level, ensure basicConfig is called only once)
log_level_str = config.get("log_level", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
# Check if root logger already has handlers to avoid duplicate logs
# Configure logging for this specific module
logger.setLevel(log_level)
# Configure handlers specifically for this logger if needed, or rely on basicConfig
if not logger.hasHandlers():
     if not logging.getLogger().hasHandlers():
          logging.basicConfig(
               level=log_level,
               format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
          )
          logger.info("Root logger configured by KnowledgeQASystem.")
     else:
          # If root logger has handlers, just use them
          logger.info("KnowledgeQASystem logger will use existing root handlers.")

    # Confidence V2 Helpers (BEGIN)
def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _norm_t5_logprob(avg_logprob):
    """
    Map avg per-token log-prob (usually in ~[-10, 0]) to [0,1].
    Keeps things monotonic and easy to weight.
    """
    if avg_logprob is None:
        return 0.0
    x = (avg_logprob + 10.0) / 10.0  # [-10..0] -> [0..1]
    return max(0.0, min(1.0, x))

def _estimate_retrieval_score(items):
    """
    Try to derive a retrieval-strength score in [0,1].
    - If items have .score or ['score'], average top-3 (normalize if >1)
    - Else, use a count heuristic that grows with the number of items.
    """
    try:
        if not items:
            return 0.0
        # prefer explicit scores if present
        scores = []
        top = items[:3] if isinstance(items, (list, tuple)) else []
        for p in top:
            s = getattr(p, "score", None)
            if s is None and isinstance(p, dict):
                s = p.get("score")
            if s is not None:
                s = float(s)
                scores.append(s/100.0 if s > 1.0 else s)
        if scores:
            return sum(scores) / len(scores)
        # fallback: count heuristic (cap growth)
        n = min(len(items), 10)
        return 1.0 - math.exp(-0.35 * n)
    except Exception:
        return 0.0

def compute_confidence_v2(signals: dict) -> float:
    """
    Weighted blend of signals into a single [0,1] confidence.
    signals may include:
      retrieval_score, bert_similarity, t5_logprob, kg_score, fallback_flag
    Missing/None values contribute 0.
    """
    w = _cfg.get("confidence_weights", {})
    retr = _to_float(signals.get("retrieval_score"))
    bert = _to_float(signals.get("bert_similarity"))
    t5n  = _norm_t5_logprob(signals.get("t5_logprob"))
    kg   = _to_float(signals.get("kg_score"))
    fb   = _to_float(signals.get("fallback_flag"))

    score = (
        w.get("retrieval", 0.0)       * retr +
        w.get("bert_similarity", 0.0) * bert +
        w.get("t5_logprob", 0.0)      * t5n +
        w.get("kg_link", 0.0)         * kg  -
        w.get("fallback_penalty", 0.0)* fb
    )
    # clamp to [0,1]
    return max(0.0, min(1.0, score))
    # Confidence V2 Helpers (END)

    # CONFIDENCE V2 APPLY (BEGIN)
def apply_confidence_v2(response_data: dict, context: dict) -> None:
    """
    Blend multiple signals into a single confidence, apply post-transform,
    optionally blend with v1, apply a success floor, and always assign.
    """
    try:
        # ---- config toggles ----
        use_t5   = bool(_cfg.get("use_t5_logprob"))
        use_kg   = bool(_cfg.get("use_kg_score"))
        shadow   = bool(_cfg.get("confidence_shadow_mode"))

        # ---- inputs & defaults ----
        v1_conf  = float(response_data.get("confidence", 0.0))

        results_or_passages = context.get("results") or context.get("passages") or []
        retrieval_score = context.get("retrieval_score")
        if retrieval_score is None:
            retrieval_score = _estimate_retrieval_score(results_or_passages)
        
        if results_or_passages:
            retrieval_score = max(retrieval_score, 0.60)

        bert_similarity = context.get("bert_similarity")
        if bert_similarity is None:
            bert_similarity = context.get("bert_confidence", 0.0)

        t5_logprob = context.get("avg_logprob") if use_t5 else None
        # kg_score   = float(context.get("kg_top_score", 0.0)) if use_kg else 0.0
        kg_raw = context.get("kg_top_score")
        kg_score = float(kg_raw) if (use_kg and kg_raw is not None) else 0.0
        used_fallback = bool(context.get("used_fallback"))
        results_count = int(context.get("results_count", 0))

        v2_signals = {
            "retrieval_score": retrieval_score,
            "bert_similarity": bert_similarity,
            "t5_logprob": t5_logprob,
            "kg_score": kg_score,
            "fallback_flag": 1.0 if used_fallback else 0.0,
        }

        # ---- raw v2 ----
        v2_raw = compute_confidence_v2(v2_signals)

        # ---- post-transform (lift & compress to your target band) ----
        base  = float(_cfg.get("confidence_post_offset", 0.15))
        scale = float(_cfg.get("confidence_post_scale", 0.85))
        v2_post = max(0.0, min(1.0, base + scale * v2_raw))

        # ---- hybrid blend with v1 (continuity) ----
        alpha = float(_cfg.get("confidence_mix_alpha", 0.7))  # 0.0 -> all v1, 1.0 -> all v2
        blended = alpha * v2_post + (1.0 - alpha) * v1_conf

        # ---- success floor (if we had results and didnâ€™t fallback) ----
        min_success = float(_cfg.get("confidence_min_success", 0.0))
        if min_success > 0 and results_count > 0 and not used_fallback:
            blended = max(blended, min_success)

        # ---- shadow log for debugging ----
        if shadow:
            logger.info(
                f"[confidence_shadow] v1={v1_conf:.3f} "
                f"v2_raw={v2_raw:.3f} v2_post={v2_post:.3f} "
                f"blend={blended:.3f} "
                f"signals={v2_signals} "
                f"cfg={{'base':{base},'scale':{scale},'alpha':{alpha},'min_success':{min_success}}}"
            )

        # ---- always assign the blended value ----
        response_data["confidence"] = blended

    except Exception as e:
        logger.exception(f"[confidence_v2] Falling back to v1 due to error: {e}")
    # CONFIDENCE V2 APPLY (END)

    
class KnowledgeQASystem:
    """
    Orchestrates the knowledge-based question answering system for medicinal plants.
    Uses config.py for configuration. (v5 - Improved error handling, response generation, fallbacks)
    """
    def __init__(self, custom_config=None):
        logger.info("Initializing Knowledge QA System (v5)...")
        # Load .env file from project root if it exists
        dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path=dotenv_path)
            logger.info(f"Loaded environment variables from: {dotenv_path}")
        else:
             logger.warning(f".env file not found at {dotenv_path}. Relying on environment variables or config defaults.")

        self.config = custom_config if custom_config else config
        self.use_decision_table = bool(self.config.get("use_decision_table", True))
        self.db = None
        self.bert = None
        self.t5 = None
        self.hydra = None
        self.hydra_system_prompt = self.config.get("hydra_system_prompt",
            "You are a concise herbal knowledge assistant. When asked about compounds and plants, "
            "list the most likely plants succinctly. If uncertain, provide best-known candidates."
        )

        try:
            if self.config.get("enable_hydra", True) and HydraClient is not None:
                self.hydra = HydraClient(
                    model=self.config.get("hydra_model", "gemma3:12b")
                    # no default_system_prompt here
                )
                logger.info("Hydra client initialized.")
            else:
                logger.info("Hydra disabled by config or HydraClient unavailable.")
        except Exception as e:
            logger.warning(f"Hydra client initialization failed: {e}")
            self.hydra = None

        # Initialize components with enhanced error handling
        self._initialize_components()

        # Setup log file
        self.log_file = self.config.get("log_file", "logs/qa_interactions.jsonl")
        self._setup_log_file()

        # Cache for optimization
        self.response_cache = {}  # Cache for frequently asked questions
        self.entity_cache = {}    # Cache for frequently queried entities
        self.cache_size = self.config.get("cache_size", 100)  # Maximum cache entries
        self.cache_ttl = self.config.get("cache_ttl", 3600)   # Time-to-live in seconds (1 hour)

        logger.info(f"Knowledge QA System initialization {'succeeded' if self.is_initialized else 'failed'}.")
    def get_plant_info_fallback(self, question: str) -> str:
        """
        Handles questions about plants. Uses BERT for entity detection first.
        If no plant is detected, falls back to querying Neo4j for potential matches.
        Returns a natural language response using FlanT5Processor.
        """
        response = ""
        plants_found = []

        try:
            # 1️⃣ Extract entities and intent with BERT
            extraction = self.bert.extract_entities_and_intent(question) if self.bert else {"entities": {}, "intent": "keyword_search_empty"}
            entities = extraction.get("entities", {})
            intent = extraction.get("intent", "keyword_search_empty")

            # 2️⃣ Check for plants detected by BERT
            plants_found = entities.get("plants", [])

            # 3️⃣ Fallback: query Neo4j if no plant found
            if not plants_found:
                logger.info("No plants detected by BERT; querying Neo4j fallback...")
                neo4j_query = {
                    "intent": "keyword_search", 
                    "parameters": {"query_text": question}
                }
                results = self.query_neo4j(neo4j_query)
                # Assume query_neo4j returns a list of plant dicts with 'name' key
                plants_found = [r["name"] for r in results if "name" in r]
                if plants_found:
                    logger.info(f"Plants found via Neo4j fallback: {plants_found}")
                else:
                    logger.info("No plants found in Neo4j fallback.")

            # 4️⃣ Build response
            if plants_found:
                # For multiple plants, you can summarize or pick the top match
                plant_name = plants_found[0]
                # Retrieve full plant info from Neo4j
                plant_info_query = {
                    "intent": "plant_info",
                    "parameters": {"plant_name": plant_name}
                }
                plant_info = self.query_neo4j(plant_info_query)

                # Format response using FlanT5Processor
                response = self.flan_processor.generate_response(plant_name, plant_info, question)
            else:
                response = "I couldn't find any information about that plant in the database. Please check the spelling or try a different plant."

        except Exception as e:
            logger.error(f"Error in get_plant_info_fallback: {e}", exc_info=True)
            response = "An error occurred while trying to retrieve plant information. Please try again later."

        return response

    def _initialize_components(self):
        """Initializes all system components with comprehensive error handling."""
        initialization_status = {
            "database": False,
            "bert": False,
            "t5": False
        }
        
        #INIT LINKER (BEGIN)
        try:
            if self.config.get("use_kg_score", False):
                logger.info("Initializing KnowledgeGraphEntityLinker...")
                self.kg_linker = KnowledgeGraphEntityLinker()
                logger.info("KnowledgeGraphEntityLinker initialized.")
            else:
                logger.info("KG score disabled by config; linker not initialized.")
        except Exception as e:
            logger.warning(f"KG linker initialization failed: {e}")
            self.kg_linker = None
        #INIT LINKER (END)
        

        # --- Database Connection ---
        # Use config object which should have loaded env vars
        neo4j_uri = self.config.get("neo4j_uri")
        neo4j_user = self.config.get("neo4j_user")
        neo4j_password = self.config.get("neo4j_password")

        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            logger.critical("Neo4j connection details (URI, User, Password) missing in configuration or environment variables (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).")
        else:
            try:
                logger.info(f"Connecting to Neo4j at {neo4j_uri}...")
                self.db = Neo4jConnection(neo4j_uri, neo4j_user, neo4j_password)
                # Test connection with a simple query
                test_result = self.db.query("RETURN 1 AS test")
                if test_result and isinstance(test_result, list) and len(test_result) > 0 and test_result[0].get('test') == 1:
                    logger.info("Successfully connected to Neo4j.")
                    initialization_status["database"] = True
                else:
                    logger.error(f"Neo4j connection test query failed or returned unexpected result: {test_result}")
                    # Keep db object for potential retries? Or set to None? Setting to None for clarity.
                    if self.db: self.db.close()
                    self.db = None
            except Exception as e:
                logger.error(f"Neo4j connection failed: {e}", exc_info=True)
                if self.db: self.db.close() # Ensure connection is closed on error
                self.db = None

        # --- BERT Model Initialization ---
        try:
            logger.info("Initializing BertProcessor...")

            # --- START: Path Determination Logic & DEBUGGING ---
            logger.debug(f"Current Working Directory (CWD): {os.getcwd()}") # Log CWD
            current_script_path = os.path.abspath(__file__)
            logger.debug(f"Full path to knowledge_qa.py: {current_script_path}")
            current_script_dir = os.path.dirname(current_script_path)
            logger.debug(f"Directory containing knowledge_qa.py: {current_script_dir}")

            # Adjust this based on your actual structure.
            # If knowledge_qa.py is in /SmartQA/nlp/, this is correct.
            # If knowledge_qa.py is in /SmartQA/, use: project_root = current_script_dir
            project_root = os.path.dirname(current_script_dir)
            logger.debug(f"Calculated Project Root: {project_root}")

            # Construct the default path relative to the project root
            default_kg_embeddings_path = os.path.join(project_root, "models", "kg_embeddings_trained")
            logger.debug(f"Calculated Default KG Embeddings Path: {default_kg_embeddings_path}")

            # Get path from config, using the dynamically determined default
            config_value = self.config.get("kg_embeddings_dir") # Check if overridden
            logger.debug(f"Value from config for 'kg_embeddings_dir' (None if not set): {config_value}")

            kg_embeddings_dir = config_value if config_value else default_kg_embeddings_path
            logger.info(f"Final path being used for KG embeddings directory: {kg_embeddings_dir}") # Log the actual path being used

            # *** Add existence checks ***
            logger.debug(f"Checking existence of directory: {kg_embeddings_dir}")
            logger.debug(f"Directory exists? {os.path.exists(kg_embeddings_dir)}")
            mappings_file_path = os.path.join(kg_embeddings_dir, 'mappings.pkl')
            proj_file_path = os.path.join(kg_embeddings_dir, 'bert_to_kg_proj.pt')
            logger.debug(f"Checking existence of mappings file: {mappings_file_path}")
            logger.debug(f"Mappings file exists? {os.path.exists(mappings_file_path)}")
            logger.debug(f"Checking existence of projection file: {proj_file_path}")
            logger.debug(f"Projection file exists? {os.path.exists(proj_file_path)}")
            # --- END: Path Determination Logic & DEBUGGING ---

            # Pass the determined (potentially absolute) path to BertProcessor
            self.bert = BertProcessor(kg_embeddings_dir=kg_embeddings_dir)

            # Add more robust check for model/tokenizer presence
            if not hasattr(self.bert, 'model') or not self.bert.model or \
               not hasattr(self.bert, 'tokenizer') or not self.bert.tokenizer:
                raise RuntimeError("BERT model or tokenizer failed to load within BertProcessor.")

            # Check if the KG data actually loaded within BertProcessor (optional but good)
            # This check now relies on BertProcessor logging its own warnings/errors during its init
            # We log a general success/failure message here based on the overall outcome.

            logger.info("BertProcessor initialization sequence completed within try block.") # Add a marker
            initialization_status["bert"] = True

        # --- Corrected single except block ---
        except Exception as e:
            logger.error(f"BertProcessor initialization failed: {e}", exc_info=True)
            self.bert = None

        # Attach Decision Table 
        try:
            if self.bert and self.config.get("use_decision_table", True):
                from os import path as _p

                # choose filename from config, default to "intent_dt.json"
                dt_file = self.config.get("intent_dt_filename", "intent_dt.json")

                # resolve alongside this module (nlp/)
                dt_path = _p.join(_p.dirname(_p.abspath(__file__)), dt_file)

                # build decision table
                try:
                    dt = SimpleIntentDecisionTable(path=dt_path, entity_provider=self.bert.extract_entities)
                except TypeError:
                    dt = SimpleIntentDecisionTable(json_path=dt_path)
                    if hasattr(dt, "set_entity_provider"):
                        dt.set_entity_provider(self.bert.extract_entities)

                # avoid double-attach if someone refactors later
                if getattr(self.bert, "dt_classifier", None) is None and dt.is_ready():
                    self.bert.attach_intent_dt(dt)
                    logger.info(f"Decision Table attached to BertProcessor from {dt_path}.")
                elif not dt.is_ready():
                    logger.warning(f"Decision Table not ready at {dt_path}; continuing with ML-only intent classification.")
                else:
                    logger.info("Decision Table already attached; skipping re-attach.")
            else:
                logger.info("Decision table disabled or BERT unavailable; skipping attach.")
        except Exception as e:
            logger.warning(f"Could not attach Decision Table: {e}")




        # --- T5 Model Initialization ---
        try:
            flan_t5_model_name = self.config.get("flan_t5_model_name", "google/flan-t5-large")
            logger.info(f"Initializing FlanT5Processor with model: {flan_t5_model_name}...")
            self.t5 = FlanT5Processor(model_name=flan_t5_model_name)
            # Check if model and tokenizer are loaded
            if not self.t5.model or not self.t5.tokenizer:
                raise RuntimeError(f"Flan-T5 model or tokenizer for '{flan_t5_model_name}' failed to load.")
            logger.info("FlanT5Processor initialized successfully.")
            initialization_status["t5"] = True
        except Exception as e:
            logger.error(f"FlanT5Processor initialization failed: {e}", exc_info=True)
            self.t5 = None

        # Hydra Client Initialization 
        try:
            if bool(self.config.get("enable_hydra", True)):
                hydra_model = self.config.get("hydra_model", "gemma3:12b")
                hydra_system = self.config.get("hydra_system_prompt",
                    "You are a concise herbal knowledge assistant. If asked about compounds and plants, list likely plants succinctly.")
                self.hydra = HydraClient(model=hydra_model, default_system_prompt=hydra_system)
                logger.info(f"Hydra client initialized (model={hydra_model}).")
            else:
                logger.info("Hydra disabled by config.")
        except Exception as e:
            logger.warning(f"Hydra client initialization failed: {e}")
            self.hydra = None

        # Set overall initialization status
        # Make T5 optional so the system can operate in degraded mode without NLG
        self.is_initialized = initialization_status.get("database", False) and initialization_status.get("bert", False)
        if not self.is_initialized:
            failed_components = [comp for comp, status in initialization_status.items() if not status]
            logger.critical(f"System initialization failed. Failed components: {', '.join(failed_components)}")
        else:
            if not initialization_status.get("t5", False):
                logger.warning("System initialized without FlanT5Processor. Responses will use simplified formatting.")
            else:
                logger.info("All core components initialized successfully.")


    def _setup_log_file(self):
        """Sets up logging file directory."""
        if self.log_file:
            try:
                log_dir = os.path.dirname(self.log_file)
                # Ensure log_dir is not empty before creating directories
                if log_dir and not os.path.exists(log_dir):
                     os.makedirs(log_dir, exist_ok=True)
                     logger.info(f"Log directory '{log_dir}' created.")
                elif log_dir:
                     logger.debug(f"Log directory '{log_dir}' already exists.")
                else:
                     # Handle case where log_file is in the current directory (no dirname)
                     logger.info(f"Using current directory for log file: '{os.path.basename(self.log_file)}'")

                # Test writability
                with open(self.log_file, 'a', encoding='utf-8') as f:
                     pass
                logger.info(f"Interaction log file configured: {self.log_file}")
            except OSError as e:
                logger.warning(f"Could not create log directory or file '{self.log_file}': {e}. Interactions will not be logged to file.")
                self.log_file = None # Disable file logging if setup fails
            except Exception as e:
                 logger.warning(f"Error setting up log file '{self.log_file}': {e}. Interactions will not be logged to file.")
                 self.log_file = None
        else:
            logger.warning("Log file path not configured. Interactions will not be logged to file.")

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Processes a natural language question about medicinal plants with enhanced
        performance, error handling, and response quality.
        """
        start_time = datetime.now()
        log_question = question # Keep original for logging

        # Initial validation checks
        if not self.is_initialized:
            logger.critical("Attempted to answer question, but system is not initialized.")
            return self._create_error_response(
                "The question answering system is currently unavailable due to initialization errors.",
                log_question
            )

        if not self.bert or not self.db:
             logger.critical("Attempted to answer question, but core components (BERT, T5, DB) are missing.")
             return self._create_error_response(
                "The question answering system is missing essential components.",
                log_question
            )

        if not question or not isinstance(question, str) or not question.strip():
            logger.warning("Received empty question.")
            return self._create_error_response("Question cannot be empty.", log_question)

        # Normalize the question for caching (lowercase, extra spaces removed)
        normalized_question = ' '.join(question.lower().split())

        # Check cache
        cached_response = self._check_cache(normalized_question)
        if cached_response:
            logger.info(f"Cache hit for question: \"{log_question}\"")
            # Return a copy to prevent modification of cached object
            final_response = cached_response.copy()
            # Update timestamp if needed, or just return as is
            return self._finalize_response(final_response, start_time, log_question, from_cache=True)


        logger.info(f"Processing question: \"{log_question}\"")

        # Initialize response structure
        response_data = {
            'answer': "",
            'structured_data_sample': None, # Changed from empty string
            'confidence': 0.0,
            'query_type': 'unknown',
            'results_count': 0,
            'entities': {},
            'error': False,
            'error_message': None # Explicitly add error message field
        }

        try:
            # --- STEP 1: Process question with BERT (Intent & Entities) ---
            logger.debug("Step 1: Processing question with BertProcessor...")
            query_info, entities = self._process_question(question)

            # Add synonyms to entities dict for easy access by formatters
            # Ensure _synonyms key exists
            entities['_synonyms'] = {}
            if self.bert: # Check if bert exists before calling methods
                for cat, ents in entities.items():
                     # Exclude internal keys
                     if cat != '_synonyms':
                          for ent in ents:
                               # Check if ent is a non-empty string
                               if ent and isinstance(ent, str):
                                    try:
                                         entities['_synonyms'][ent] = self.bert.get_synonyms(ent)
                                    except Exception as syn_e:
                                         logger.warning(f"Could not get synonyms for entity '{ent}': {syn_e}")
                                         entities['_synonyms'][ent] = [ent.lower()] # Fallback to just the entity itself
                               else:
                                     logger.debug(f"Skipping synonym lookup for non-string or empty entity in category '{cat}': {ent}")


            response_data['entities'] = entities
            response_data['query_type'] = query_info.get('intent', 'unknown')
            response_data['intent_source'] = query_info.get('intent_source', getattr(self.bert, 'last_intent_source', 'unknown'))
            logger.debug(f"Intent: {response_data['query_type']}, Entities (with synonyms): {entities}") # Log extracted entities including synonyms
            kg_top_score = self._get_kg_top_score(question)

            # --- STEP 2: Execute database query ---
            logger.debug("Step 2: Executing database query...")
            results, results_count = self._execute_query(query_info)
            response_data['results_count'] = results_count
            logger.debug(f"Database query returned {results_count} results.")

            # --- STEP 3: Process results ---
            # Only process if results exist
            processed_data = []
            if results:
                logger.debug("Step 3: Processing database results...")
                processed_data = self._process_results(results, response_data['query_type'])
                response_data['structured_data_sample'] = f"Processed {len(processed_data)} items from {results_count} raw results."
                logger.debug(f"Processed {len(processed_data)} results.")
            else:
                response_data['structured_data_sample'] = "No results returned from database query."
                logger.debug("No results to process.")

            # --- STEP 4: Generate Natural Language Response ---
            logger.debug("Step 4: Generating natural language response...")
            try:
                nl_answer = self._generate_response(query_info, entities, processed_data, results_count, question)
            except Exception as e:
                logger.error(f"Failed to generate response: {e}", exc_info=True)
                nl_answer = f"I encountered an error while generating a response to your question. Please try rephrasing or ask about something else."

            # Ensure nl_answer is not None or empty
            if not nl_answer or not isinstance(nl_answer, str) or not nl_answer.strip():
                logger.warning("Generated answer is empty or None. Using fallback.")
                nl_answer = self._fallback_response(query_info.get('intent', 'unknown'), entities, processed_data)
            
            # Ensure we have at least some response text
            if not nl_answer or not nl_answer.strip():
                nl_answer = "I'm sorry, I couldn't generate a response to your question. Please try rephrasing your question or asking about something else."

            # Check if the generated answer is meaningful or just a fallback/error
            is_error_or_fallback = "I couldn't find specific information" in nl_answer or \
                                   "I understand you're interested" in nl_answer or \
                                   "encountered difficulty formatting" in nl_answer or \
                                   "encountered a technical issue" in nl_answer or \
                                   "seems a bit broad or unclear" in nl_answer or \
                                   "tried to find" in nl_answer # Added check for _fallback_response trigger
            
            #T5 AVG LOGPROB (BEGIN)
            avg_logprob = None
            if _cfg.get("use_t5_logprob", False) and self.t5:
                try:
                    avg_logprob = self.t5.score_pair(question, nl_answer)
                except Exception as e:
                    logger.warning(f"Failed to score T5 log-prob: {e}")
            #T5 AVG LOGPROB (END)


            # --- STEP 5: Add Cautions ---
            # Always enhance, even fallbacks, unless it's a direct error message from _create_error_response
            if not response_data['error']:
                 # Ensure T5 is available before calling its methods
                 if self.t5:
                      response_data['answer'] = self.t5.enhance_response_with_cautions(nl_answer)
                 else:
                      logger.warning("T5 processor not available, cannot enhance response with cautions.")
                      response_data['answer'] = nl_answer # Use unenhanced answer
            else:
                 # Keep the specific error message generated by _create_error_response
                 response_data['answer'] = nl_answer


            # --- STEP 6: Calculate Confidence ---
            # Refined confidence logic
            if results_count > 0 and not is_error_or_fallback:
                 confidence = 0.75 # Base confidence for successful retrieval & formatting
                 bert_confidence = self._calculate_confidence(question, response_data['answer'])
                 response_data['confidence'] = max(0.5, (confidence + bert_confidence) / 2) # Ensure confidence >= 0.5 if successful
            elif results_count > 0 and is_error_or_fallback:
                 response_data['confidence'] = 0.4 # Lower confidence if fallback used despite results
            elif results_count == 0 and not is_error_or_fallback:
                 # Case where formatter generated something (e.g., safety warning) without DB results
                 response_data['confidence'] = 0.35
            else: # No results and fallback/error response
                 response_data['confidence'] = 0.2 # Low confidence

            # Ensure confidence is within [0, 1]
            response_data['confidence'] = max(0.0, min(response_data['confidence'], 1.0))
            
            #  CONFIDENCE V2 CALL (BEGIN) 
            apply_confidence_v2(
                response_data,
                {
                    "results": results,
                    "passages": processed_data,
                    "bert_confidence": locals().get("bert_confidence", None),
                    "used_fallback": is_error_or_fallback,
                    "avg_logprob": avg_logprob,
                    "kg_top_score": kg_top_score,
                },
            )
            # CONFIDENCE V2 CALL (END) 
            # --- STEP 7: Finalize and Cache ---
            final_response = self._finalize_response(response_data, start_time, log_question)
            # Cache only successful, non-error responses with reasonable confidence
            if not final_response.get('error') and final_response.get('confidence', 0) >= 0.4:
                self._cache_response(normalized_question, final_response)

            return final_response

        except Exception as e:
            error_message = f"Unexpected error during question processing pipeline: {str(e)}"
            logger.error(error_message, exc_info=True)
            # Ensure a proper error response is created and finalized
            error_response = self._create_error_response(error_message, log_question)
            # Use start_time from the beginning of the function
            return self._finalize_response(error_response, start_time, log_question)

    def _process_question(self, question: str) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """
        Processes the question using BertProcessor to get intent, entities, keywords, and query info.
        Returns:
            - query_info dict (includes intent and parameters)
            - entities dict (including 'keywords')
        """
        if not self.bert:
            logger.error("BertProcessor not initialized. Cannot process question.")
            return (
                {"intent": "error", "query": None, "parameters": {}, "intent_source": "unknown"},
                {"plants": [], "conditions": [], "compounds": [], "regions": [], "keywords": []},
            )

        logger.debug(f'Processing question with BertProcessor: "{question[:100]}..."')
        try:
            # --- Extract Neo4j query info and entities ---
            query_info = self.bert.build_neo4j_query(question)
            entities = self.bert.extract_entities(question) or {}

            # Optional: sanitize entities
            try:
                from .bert_processor import _sanitize_entities_dict
                entities = _sanitize_entities_dict(entities)
            except Exception:
                STOP_PHRASES = {
                    "alternative", "alternatives", "safer alternative", "safer alternatives",
                    "similar", "similar to", "substitute", "substitutes",
                    "replacement", "replacements", "like"
                }
                for k in ("plants", "conditions", "compounds", "regions", "keywords"):
                    vals = entities.get(k) or []
                    entities[k] = [v for v in vals if v and str(v).strip().lower() not in STOP_PHRASES]

            # --- Determine intent ---
            pred_intent = None
            intent_source = "ml"
            try:
                dtc = getattr(self.bert, "dt_classifier", None)
                if dtc and hasattr(dtc, "is_ready") and dtc.is_ready():
                    tmp = dtc.predict(question)
                    if isinstance(tmp, str) and tmp and tmp != "unknown":
                        pred_intent = tmp
                        intent_source = "dt"
            except Exception as e:
                logger.debug(f"Decision Table predict failed; falling back to ML. {e}")

            if not pred_intent:
                pred_intent = self.bert.classify_question_intent(question)
                intent_source = getattr(self.bert, "last_intent_source", "ml")

            query_info['intent'] = coerce_intent_by_entities(pred_intent, entities)
            query_info['intent_source'] = intent_source

            # Ensure parameters dict exists
            if "parameters" not in query_info or query_info["parameters"] is None:
                query_info["parameters"] = {}

            logger.info(
                f"BertProcessor results - Intent: '{query_info.get('intent', 'N/A')}', "
                f"Source: {intent_source}, Entities: {entities}"
            )

            return query_info, entities

        except Exception as e:
            logger.error(f"Error processing question with BERT: {e}", exc_info=True)
            fallback_entities = {"plants": [], "conditions": [], "compounds": [], "regions": [], "keywords": []}
            fallback_query_info = {"intent": "error", "query": None, "parameters": {}, "intent_source": "unknown"}
            return fallback_query_info, fallback_entities



    def _execute_query(self, query_info: Dict[str, Any]) -> Tuple[List[Any], int]:
        """
        Executes the database query based on query_info. Handles potential DB errors.
        Returns results list and result count.
        """
        intent = query_info.get('intent', 'unknown')
        query = query_info.get('query')
        params = query_info.get('parameters', {})

        logger.debug(f"Executing query for intent: {intent}")
        # logger.debug(f"Query: {query}") # Be cautious logging full queries if sensitive
        logger.debug(f"Parameters: {params}")

        # Skip execution if no query is provided (e.g., for error or empty intent)
        no_query_intents = ['error', 'keyword_search_empty', 'general_query', 'unknown']
        if not query or intent in no_query_intents:
            logger.info(f"Skipping database query for intent '{intent}'.")
            return [], 0

        if not self.db:
             logger.error("Database connection is not available. Cannot execute query.")
             return [], 0

        try:
            results = self.db.query(query, parameters=params)
            # Ensure results is a list
            if results is None:
                results = []
            elif not isinstance(results, list):
                 logger.warning(f"Neo4j query returned non-list type: {type(results)}. Converting to empty list.")
                 results = []

            results_count = len(results)
            logger.info(f"Query for intent '{intent}' executed. Found {results_count} results.")
            return results, results_count

        except Exception as e:
            # Catch potential Neo4j driver errors or connection issues
            logger.error(f"Database query execution failed for intent '{intent}': {e}", exc_info=True)
            # Log query details that caused the error for debugging
            logger.error(f"Failed Query Details - Query: {query}")
            logger.error(f"Failed Query Details - Params: {params}")
            return [], 0 # Return empty list on error

    def _process_results(self, results: List[Any], intent: str) -> List[Dict]:
        """
        Processes and transforms Neo4j results into a more structured list of dictionaries.
        Handles potential variations in Neo4j driver return types.
        """
        if not results:
            return []

        processed_results = []
        try:
            for i, record in enumerate(results):
                if record is None: # Check for None record explicitly
                    logger.warning(f"Skipping None record at index {i}")
                    continue

                # Handle different possible record types from Neo4j drivers
                record_dict = {}
                if hasattr(record, 'data'): # Common for neo4j driver records
                    record_dict = record.data()
                elif isinstance(record, dict): # Already a dictionary
                    record_dict = record
                else:
                    logger.warning(f"Unexpected record type at index {i}: {type(record)}. Attempting conversion.")
                    try:
                         # Try converting common types like tuples or lists if they contain dicts
                         if isinstance(record, (list, tuple)) and len(record) > 0 and isinstance(record[0], dict):
                              record_dict = record[0] # Assume first element is the data
                         else:
                              # Attempt direct dict conversion if possible (e.g., for Record objects)
                              record_dict = dict(record)
                    except (TypeError, ValueError) as conversion_e:
                         logger.error(f"Could not convert record at index {i} to dict: {conversion_e}. Record: {record}")
                         continue # Skip this record

                # Basic cleaning: remove None values, strip strings, clean lists
                cleaned_record = {}
                for key, value in record_dict.items():
                     if value is not None:
                          if isinstance(value, str):
                               cleaned_record[key] = value.strip()
                          elif isinstance(value, list):
                               # Clean lists recursively? For now, just clean top-level lists of strings
                               cleaned_record[key] = self._clean_list(value)
                          else:
                               cleaned_record[key] = value
                     # else: keep None out

                if cleaned_record: # Add if not empty after cleaning
                    processed_results.append(cleaned_record)


            logger.debug(f"Processed {len(processed_results)} results into dicts for intent: {intent}")
            return processed_results

        except Exception as e:
            logger.error(f"Error processing results list for intent '{intent}': {e}", exc_info=True)
            # Return whatever was processed so far
            return processed_results

    # Normalization helpers (kept simplified for now, expand if needed)
    def _normalize_record(self, record: Dict, intent: str) -> Optional[Dict]:
         """ Placeholder for intent-specific record normalization if required. """
         # Example: ensure 'name' field exists, effects are always lists, etc.
         # This logic depends heavily on the exact query structures.
         # For now, basic cleaning happens in _process_results.
         return record # Passthrough

    def _clean_list(self, items: Any) -> List[Any]:
        """Safely cleans list-like data from records, preserving non-string types if appropriate."""
        if items is None: return []
        cleaned_list = []
        if isinstance(items, (list, set, tuple)):
            for item in items:
                if isinstance(item, str):
                    stripped_item = item.strip()
                    if stripped_item: # Add non-empty strings
                        cleaned_list.append(stripped_item)
                elif item is not None: # Keep other non-None items as is
                    cleaned_list.append(item)
            return cleaned_list
        elif isinstance(items, str): # Handle single string case
             stripped_item = items.strip()
             return [stripped_item] if stripped_item else []
        else:
            # If it's not list-like or string, but not None, return as single-item list
            logger.debug(f"Treating non-list/non-string item of type {type(items)} as single-item list.")
            return [items]


    def _get_name_field(self, record: Dict) -> str:
         """Extracts a primary name field from a record using common variations."""
         # Prioritized list of common name fields
         name_fields = ['name', 'plant_name', 'p_name', 'p2_name', 'compound_name', 'condition_name', 'region_name', 'common_name']
         for field in name_fields:
              value = record.get(field)
              # Check if value is a non-empty string
              if value and isinstance(value, str) and value.strip():
                   return value.strip()
         # Fallback: check keys containing 'name'
         for key, value in record.items():
              if 'name' in key.lower() and value and isinstance(value, str) and value.strip():
                   return value.strip()
         return "Unknown Entity" # Default if no suitable name found


    def _get_formatter_and_args(self, intent, entities, processed_data, results_count):
        """
        Maps intent to the correct FlanT5Processor formatter and prepares arguments.
        Returns (formatter_fn, args_dict) or (None, None) if no suitable formatter.
        """
        if not self.t5:
            logger.error("FlanT5Processor (self.t5) is not initialized. Cannot get formatter.")
            return None, None

        # Ensure processed_data is a list
        data_list = processed_data if isinstance(processed_data, list) else []
        # Get the first result safely, ensuring it's a dict
        first_result = data_list[0] if data_list and isinstance(data_list[0], dict) else {}

        # Helper to safely get the first entity of a type
        def get_first(key):
             # Use the helper function defined in the class
             return self._get_entity(entities, key)


        # Map intent to (formatter_method, args_dict_builder_lambda)
        formatter_map = {
            'plant_info': (self.t5.format_plant_info, lambda: dict(plant_data=first_result, entities=entities)),
            'condition_plants': (self.t5.format_condition_plants, lambda: dict(condition=get_first('conditions'), plants_data=data_list, entities=entities)),
            'multi_condition_plants': (self.t5.format_multi_condition_plants, lambda: dict(conditions=entities.get('conditions', []), plants_data=data_list, entities=entities)),
            'plant_compounds': (self.t5.format_plant_compounds, lambda: dict(plant_name=get_first('plants'), compounds_data=data_list, entities=entities)),
            'compound_effects': (self.t5.format_compound_effects, lambda: dict(compound_name=get_first('compounds'), compound_data=first_result, entities=entities)),
            'compound_plants': (self.t5.format_compound_plants, lambda: dict(compound_name=get_first('compounds'), results=data_list, entities=entities)),
            'region_plants': (self.t5.format_region_plants, lambda: dict(region=get_first('regions'), plants_data=data_list, entities=entities)),
            'region_condition_plants': (self.t5.format_region_condition_plants, lambda: dict(region=get_first('regions'), condition=get_first('conditions'), plants_data=data_list, entities=entities)),
            'safety_info': (self.t5.format_safety_info, lambda: dict(plant_name=get_first('plants'), safety_data=first_result, context={}, entities=entities)), # Added empty context dict
            # Use the unified preparation formatter
            'plant_preparation': (self.t5.format_preparation_methods, lambda: dict(target_entity=get_first('plants'), prep_data=data_list, is_condition_query=False, entities=entities)),
            'preparation_for_condition': (self.t5.format_preparation_methods, lambda: dict(target_entity=get_first('conditions'), prep_data=data_list, is_condition_query=True, entities=entities)),
            'similar_plants': (self.t5.format_similar_plants, lambda: dict(target_plant=get_first('plants'), similar_plants_data=data_list, entities=entities)),
             # Add fallback/general formatter intents explicitly
            'general_query': (self.t5.generate_general_explanation, lambda: dict(intent=intent, entities=entities, count=results_count)),
            'keyword_search': (self.t5.generate_general_explanation, lambda: dict(intent=intent, entities=entities, count=results_count)), # Map keyword search to general explanation for now
            'keyword_search_empty': (self.t5.generate_general_explanation, lambda: dict(intent=intent, entities=entities, count=results_count)),
            'unknown': (self.t5.generate_general_explanation, lambda: dict(intent=intent, entities=entities, count=results_count)),
            'error': (self.t5.generate_general_explanation, lambda: dict(intent=intent, entities=entities, count=results_count)), # Map error intent too
        }

        formatter_tuple = formatter_map.get(intent)

        if formatter_tuple:
            formatter_fn, args_builder = formatter_tuple
            try:
                args = args_builder()
                # Basic validation of essential args
                if intent == 'plant_info' and not args.get('plant_data'): logger.debug(f"Formatter '{intent}': Missing 'plant_data'.") # Debug level
                if intent == 'condition_plants' and not args.get('condition'): logger.debug(f"Formatter '{intent}': Missing 'condition'.")
                if intent == 'safety_info' and not args.get('plant_name'): logger.debug(f"Formatter '{intent}': Missing 'plant_name'.")
                # Add more checks as needed...
                return formatter_fn, args
            except Exception as e:
                 logger.error(f"Error building arguments for formatter '{intent}': {e}", exc_info=True)
                 # Fallback to general explanation if args building fails
                 if self.t5 and hasattr(self.t5, 'generate_general_explanation'):
                     return self.t5.generate_general_explanation, dict(intent=intent, entities=entities, count=results_count)
                 else:
                     logger.error("T5 not available for fallback. Returning None, None to trigger fallback response.")
                     return None, None
        else:
            logger.warning(f"No specific formatter mapped for intent '{intent}'. Using general explanation.")
            # Fallback: general explanation for unmapped intents
            return self.t5.generate_general_explanation, dict(intent=intent, entities=entities, count=results_count)


    def _calculate_confidence(self, question: str, answer: str) -> float:
        """Calculates confidence score (placeholder using BERT if available)."""
        # This is a placeholder. Real confidence requires a trained model or complex heuristics.
        default_confidence = 0.5 # Default if BERT method unavailable
        if self.bert and hasattr(self.bert, 'get_answer_confidence'):
            try:
                confidence = self.bert.get_answer_confidence(question, answer)
                # Clamp confidence between 0.1 and 0.9 to avoid extremes from simple methods
                return float(max(0.1, min(confidence, 0.9)))
            except Exception as e:
                logger.warning(f"BERT confidence calculation failed: {e}. Using default.")
                return default_confidence
        else:
            # Basic heuristic: longer answers are slightly more confident? (Very weak)
            if len(answer) > 200: return 0.6
            if len(answer) > 100: return 0.55
            if len(answer) > 50: return 0.5
            return 0.3 # Lower confidence for very short answers
        
        #KG SCORE HELPER (BEGIN)
    def _get_kg_top_score(self, question: str):
            """Safely compute KG top score, honoring the config flag."""
            if not (_cfg.get("use_kg_score", False) and getattr(self, "kg_linker", None)):
                return None
            try:
                _, top_score = self.kg_linker.link_entities_with_scores(question)
                return top_score
            except Exception as e:
                logger.warning(f"KG score unavailable: {e}")
                return None
        #KG SCORE HELPER (END)

    def _get_entity(self, entities: Dict, key: str) -> Optional[str]:
        """Safely gets the first entity from a list in the entities dictionary."""
        if not entities or not isinstance(entities, dict): return None
        entity_list = entities.get(key, [])
        # Return the first item if it's a non-empty list and the item is a non-empty string
        if entity_list and isinstance(entity_list, list) and entity_list[0] and isinstance(entity_list[0], str) and entity_list[0].strip():
            return entity_list[0].strip()
        return None


    def _get_main_entity(self, entities: Dict) -> Optional[str]:
        """Gets the most likely primary entity based on priority order."""
        if not entities: return None
        priority = ['plants', 'conditions', 'compounds', 'regions']
        for key in priority:
            entity = self._get_entity(entities, key)
            if entity:
                return entity
        # Fallback: check _synonyms if primary keys failed and _synonyms exists
        if '_synonyms' in entities and isinstance(entities['_synonyms'], dict):
             syn_dict = entities['_synonyms']
             for key in priority:
                  # Find first synonym key that matches priority category (simple check)
                  entity_name = next((ent for ent in syn_dict.keys() if key.rstrip('s') in ent.lower()), None)
                  if entity_name: return entity_name # Return the original entity name
        return None

    def _generate_response(self, query_info, entities, processed_data, results_count, original_question):
        """
        Generates the final answer. If hydra_mode == "always", Hydra authors the answer.
        Otherwise uses local formatters with an optional Hydra fallback.
        """
        intent = query_info.get('intent', 'unknown')
        logger.debug(f"Generating response for intent: {intent}, Results count: {results_count}")

        def _build_hydra_prompt() -> str:
            # Make a compact, helpful context for Hydra
            ent_lines = []
            for k in ("plants", "compounds", "conditions", "regions"):
                vals = (entities or {}).get(k) or []
                if vals:
                    ent_lines.append(f"{k}: {', '.join(vals[:5])}")
            ent_block = "\n".join(ent_lines) if ent_lines else "none"

            # Include up to 3 top names from DB for grounding
            names = []
            for rec in (processed_data or [])[:3]:
                if isinstance(rec, dict):
                    for key in ("plant_name", "p_name", "name", "scientific_name"):
                        v = rec.get(key)
                        if isinstance(v, str) and v.strip():
                            names.append(v.strip()); break
            names_block = ", ".join(names) if names else "none"

            return (
                "User question:\n"
                f"{original_question}\n\n"
                "Detected intent:\n"
                f"{intent}\n\n"
                "Detected entities:\n"
                f"{ent_block}\n\n"
                "Database hints (top names if any):\n"
                f"{names_block}\n\n"
                "Answer directly and succinctly."
            )

        def _answer_via_hydra() -> str:
            if not getattr(self, "hydra", None):
                return ""
            try:
                prompt = _build_hydra_prompt()
                sys_msg = self.config.get("hydra_system_prompt") or \
                        "You are a concise herbal knowledge assistant."
                temp = float(self.config.get("hydra_temperature", 0.4))
                max_tok = int(self.config.get("hydra_max_tokens", 320))
                text = self.hydra.simple_chat(prompt, system=sys_msg, temperature=temp, max_tokens=max_tok, stream=False)
                return (text or "").strip()
            except Exception as e:
                logger.warning(f"Hydra answer failed: {e}")
                return ""

        # --- Hydra-first path ---
        hydra_mode = (self.config.get("hydra_mode") or "fallback").lower()
        if self.config.get("hydra_enable", False) and hydra_mode == "always":
            text = _answer_via_hydra()
            if text:
                # still pass through T5 safety enhancer if available
                if self.t5:
                    try:
                        text = self.t5.enhance_response_with_cautions(text)
                    except Exception as e:
                        logger.debug(f"T5 caution enhancement skipped: {e}")
                return text
            # if Hydra failed, fall through to local generation as a safety net

        # --- Local generation (used when Hydra is off, or in fallback mode, or Hydra failed) ---
        formatter_fn, args = self._get_formatter_and_args(intent, entities, processed_data, results_count)
        if formatter_fn is None or args is None:
            # If hydra_mode == fallback, try Hydra here before default fallback
            if self.config.get("hydra_enable", False) and hydra_mode == "fallback":
                text = _answer_via_hydra()
                if text:
                    if self.t5:
                        try:
                            text = self.t5.enhance_response_with_cautions(text)
                        except Exception:
                            pass
                    return text
            logger.error(f"Could not determine formatter/args for intent '{intent}'. Using local fallback.")
            return self._fallback_response(intent, entities, processed_data)

        try:
            resp = formatter_fn(**args)

            # Guard: ensure string
            if resp is None or not isinstance(resp, str):
                resp = "" if resp is None else str(resp)

            # If result looks empty/negative and hydra_mode == fallback → ask Hydra to improve
            neg_markers = ("i couldn't find", "could not find", "no specific", "no matching", "not listed", "no plants listed", "no results")
            looks_negative = any(m in resp.lower() for m in neg_markers)
            too_short = len(resp.strip()) < 30
            placeholder_phrases = ("Unknown Entity", "Unknown Method", "None listed in database.", "Please specify")
            looks_placeholder = any(p in resp for p in placeholder_phrases)

            if self.config.get("hydra_enable", False) and hydra_mode == "fallback" and (looks_negative or looks_placeholder or too_short):
                text = _answer_via_hydra()
                if text:
                    if self.t5:
                        try:
                            text = self.t5.enhance_response_with_cautions(text)
                        except Exception:
                            pass
                    return text

            # Normal path: enhance with cautions and return
            if self.t5:
                try:
                    resp = self.t5.enhance_response_with_cautions(resp)
                except Exception:
                    pass
            return resp

        except Exception as e:
            logger.error(f"Formatter failed for intent '{intent}': {e}", exc_info=True)
            if intent == "safety_info":
                plant_name = self._get_main_entity(entities) or "the requested plant"
                return (
                    f"I encountered an issue retrieving detailed safety specifics for **{plant_name}**. "
                    "However, it is crucial to exercise caution with any medicinal plant. "
                    "General risks can include unexpected side effects, allergic reactions, "
                    "or interactions with medications. "
                )
            # Last resort: Hydra if in fallback mode, else local fallback
            if self.config.get("hydra_enable", False) and hydra_mode == "fallback":
                text = _answer_via_hydra()
                if text:
                    if self.t5:
                        try:
                            text = self.t5.enhance_response_with_cautions(text)
                        except Exception:
                            pass
                    return text
            return self._fallback_response(intent, entities, processed_data)




    def _create_error_response(self, error_message: str, question: Optional[str]=None) -> Dict[str, Any]:
        """Creates a standardized error response dictionary."""
        user_message = "Sorry, I encountered a technical problem processing your request."
        # Customize user message based on error type if possible
        error_lower = str(error_message).lower() # Ensure it's string
        if "empty" in error_lower and "question" in error_lower:
            user_message = "I need a question to process. Please enter your query."
        elif "connection" in error_lower or "database" in error_lower or "neo4j" in error_lower:
            user_message += " There might be an issue connecting to the knowledge base."
        elif "initialization" in error_lower or "missing essential components" in error_lower:
            user_message = "The question answering system is currently unavailable. Please try again later."
        # Add standard advice
        user_message += " You could try rephrasing your question or asking about something else."

        logger.error(f"Creating error response. Internal Error: {error_message}. Question: '{question}'")
        return {
            'answer': user_message, # User-facing message
            'structured_data_sample': None,
            'confidence': 0.0,
            'query_type': 'error',
            'results_count': 0,
            'entities': {},
            'error': True,
            'error_message': str(error_message) # Internal error detail
        }

    def _finalize_response(self, response_data: Dict[str, Any], start_time: datetime, question: str, from_cache: bool = False) -> Dict[str, Any]:
        """Finalizes and logs the response before returning it."""
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Ensure essential keys exist with default values
        response_data.setdefault('intent_source', 'ml')
        response_data.setdefault('answer', "Error: Response generation failed.")
        response_data.setdefault('entities', {})
        response_data.setdefault('query_type', 'unknown')
        response_data.setdefault('confidence', 0.0)
        response_data.setdefault('error', False)
        response_data.setdefault('results_count', 0)

        # Log the interaction details
        log_data = {
            "timestamp": end_time.isoformat(),
            "question": question,
            "answer": response_data['answer'],
            "intent": response_data['query_type'],
            "intent_source": response_data['intent_source'],
            "entities": response_data['entities'], # Keep full entities dict for logging
            "confidence": response_data['confidence'],
            "duration_seconds": round(processing_time, 3), # Round duration
            "results_count": response_data['results_count'],
            "from_cache": from_cache,
            "error_flag": response_data['error'],
            "error_message": response_data.get('error_message', None) # Log internal message
        }
        self._log_interaction(**log_data)

        # Remove internal error message and synonyms before returning to user/API
        response_data.pop('error_message', None)
        response_data.get('entities', {}).pop('_synonyms', None) # Remove synonyms from final output


        log_level = logging.INFO if not response_data['error'] else logging.ERROR
        logger.log(log_level, f"Question processed in {processing_time:.3f}s. Intent: {response_data['query_type']}. Confidence: {response_data['confidence']:.3f}. Error: {response_data['error']}. Cache: {from_cache}.")

        return response_data
    
    def _log_interaction(self, **kwargs) -> None:
        """Logs interaction details to a JSONL file."""
        if not self.log_file:
            return # Logging disabled or failed setup

        try:
            # Ensure entities are JSON serializable (convert sets to lists)
            entities = kwargs.get('entities', {})
            serializable_entities = {}
            for k, v in entities.items():
                 # Handle sets and also lists containing non-serializable items like sets
                 if isinstance(v, set):
                      serializable_entities[k] = sorted([str(item) for item in v]) # Convert items to str
                 elif isinstance(v, list):
                      serializable_entities[k] = sorted([str(item) for item in v]) # Convert items to str
                 elif isinstance(v, dict) and k == '_synonyms': # Handle synonyms dict
                      serializable_entities[k] = {key: sorted([str(s) for s in syn_list]) for key, syn_list in v.items()}
                 else:
                      try:
                           # Attempt to serialize other types, convert to string if fails
                           json.dumps(v)
                           serializable_entities[k] = v
                      except TypeError:
                           serializable_entities[k] = str(v)


            interaction = {
                'timestamp': kwargs.get('timestamp', datetime.now().isoformat()),
                'duration_seconds': kwargs.get('duration_seconds'),
                'question': kwargs.get('question'),
                'intent': kwargs.get('intent'),
                'intent_source': kwargs.get('intent_source', 'ml'),
                'entities': serializable_entities, # Use serializable version
                'confidence': kwargs.get('confidence'),
                'results_count': kwargs.get('results_count'),
                'answer_length': len(kwargs.get('answer', '')),
                'from_cache': kwargs.get('from_cache', False),
                'error_flag': kwargs.get('error_flag', False),
                'error_message': kwargs.get('error_message', None) # Include error message in log
            }

            # Append to the JSONL file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                # Use ensure_ascii=False for broader character support
                f.write(json.dumps(interaction, ensure_ascii=False) + '\\n')

        except Exception as e:
            # Log failure to write log, but don't crash the main process
            logger.warning(f"Failed to write interaction to log file '{self.log_file}': {e}")
            # Optionally, disable file logging temporarily if errors persist
            # self.log_file = None

    def _check_cache(self, normalized_question: str) -> Optional[Dict[str, Any]]:
        """Checks cache for a valid, non-expired response."""
        if normalized_question in self.response_cache:
            entry = self.response_cache[normalized_question]
            timestamp = entry.get('timestamp')
            if timestamp and isinstance(timestamp, datetime):
                age = (datetime.now() - timestamp).total_seconds()
                if age < self.cache_ttl:
                    logger.debug(f"Cache hit: Found valid entry for '{normalized_question}' (Age: {age:.1f}s).")
                    return entry.get('response') # Return the cached response dict
                else:
                    logger.debug(f"Cache expired: Entry for '{normalized_question}' removed (Age: {age:.1f}s > TTL: {self.cache_ttl}s).")
                    self.response_cache.pop(normalized_question, None) # Remove expired entry
            else:
                 logger.warning(f"Invalid timestamp in cache for '{normalized_question}'. Removing entry.")
                 self.response_cache.pop(normalized_question, None) # Remove invalid entry
        return None

    def _cache_response(self, normalized_question: str, response: Dict[str, Any]) -> None:
        """Caches a successful response, managing cache size."""
        # Only cache non-error responses with decent confidence
        if response.get('error') or response.get('confidence', 0) < 0.4:
            logger.debug(f"Skipping cache for low confidence or error response (Q: '{normalized_question}')")
            return

        # Ensure cache doesn't exceed max size
        if len(self.response_cache) >= self.cache_size:
            try:
                # Find and remove the oldest entry (based on timestamp)
                # Use datetime.max for entries potentially missing timestamp during sort
                oldest_q = min(self.response_cache.keys(),
                               key=lambda q: self.response_cache[q].get('timestamp', datetime.max))
                removed_entry = self.response_cache.pop(oldest_q, None)
                if removed_entry:
                     removed_ts = removed_entry.get('timestamp', 'N/A')
                     logger.debug(f"Cache full ({self.cache_size}). Removed oldest entry for question: '{oldest_q}' (Timestamp: {removed_ts})")
                else:
                     logger.warning("Cache full, but failed to identify/remove oldest entry.")
            except ValueError:
                 logger.warning("Cache is full, but no entries found to remove (potentially empty or error during min key search).")
            except Exception as e:
                 logger.warning(f"Error trimming cache: {e}")

        # Add the new entry with a timestamp
        # Store a copy of the response to prevent modification by reference
        self.response_cache[normalized_question] = {
            'timestamp': datetime.now(),
            'response': response.copy()
        }
        logger.debug(f"Cached response for question: '{normalized_question}'")


    def clear_cache(self) -> None:
        """Clears all cached responses."""
        self.response_cache.clear()
        self.entity_cache.clear() # Assuming entity cache exists
        logger.info("Response and entity caches cleared.")

    def close(self) -> None:
        """Closes database connection and performs cleanup."""
        logger.info("Shutting down Knowledge QA System...")
        if self.db:
            try:
                self.db.close()
                logger.info("Neo4j connection closed.")
            except Exception as e:
                logger.error(f"Error closing Neo4j connection: {e}", exc_info=True)
        self.clear_cache()
        logger.info("Knowledge QA System shutdown completed.")


    def _fallback_response(self, intent: str, entities: Dict[str, List[str]], data: List[Dict]) -> str:
        """
        Generates a more structured and informative fallback response.
        """
        logger.warning(f"Generating fallback response for intent '{intent}'.")
        sections = []

        # 1. Acknowledge the request intent
        intent_map = {
             'plant_info': 'information about a specific plant',
             'condition_plants': 'plants for a specific condition',
             'safety_info': 'safety information about a plant',
             'compound_effects': 'effects of a specific compound',
             'region_plants': 'plants from a specific region',
             'multi_condition_plants': 'plants for multiple conditions',
             'plant_compounds': 'compounds in a specific plant',
             'compound_plants': 'plants containing a specific compound',
             'plant_preparation': 'preparation methods for a plant',
             'preparation_for_condition': 'preparation methods for a condition',
             'similar_plants': 'plants similar to another plant',
             # Add more mappings as needed
        }
        intent_desc = intent_map.get(intent, f"your query (type: {intent})")
        sections.append(f"I tried to find {intent_desc}, but encountered some difficulty generating a detailed response.")

        # 2. Summarize identified entities (excluding synonyms)
        entity_summary = []
        primary_entity = self._get_main_entity(entities) # Use helper

        for cat, ents in entities.items():
             if cat != '_synonyms' and ents:
                  ent_str = f"{cat.capitalize()}: {', '.join(ents[:3])}"
                  if len(ents) > 3: ent_str += "..."
                  entity_summary.append(ent_str)

        if entity_summary:
            sections.append(f"\\nI recognized the following entities in your request: {'; '.join(entity_summary)}.")
        elif primary_entity: # Fallback if entity dict is weird but primary was found
             sections.append(f"\\nI recognized '{primary_entity}' in your request.")
        # Don't say "trouble identifying" if entities *were* identified but formatting failed.

        # 3. Indicate if *any* data was found
        if data:
            sections.append(f"\\nI did find {len(data)} potentially relevant entries in the database, but couldn't format them into a specific answer.")
            # Optional: Mention the name of the first item if available and useful
            # first_item_name = self._get_name_field(data[0]) if isinstance(data[0], dict) else None
            # if first_item_name and first_item_name != "Unknown Entity":
            #      sections.append(f"The first entry relates to '{first_item_name}'.")
        elif intent not in ['error', 'keyword_search_empty', 'general_query', 'unknown']:
             # Only say no data if we actually expected results based on intent
             sections.append("\\nI could not find specific matching data in the database for this query.")


        # 4. Provide actionable suggestions
        suggestions = "\\n\\nTo help me understand better, you could try:"
        suggestions += "\\nâ€¢ Rephrasing your question slightly."
        suggestions += "\\nâ€¢ Asking about a single entity (e.g., just one plant or condition)."
        suggestions += "\\nâ€¢ Specifying the type of information (e.g., 'effects of X', 'safety of Y', 'preparation for Z')."
        sections.append(suggestions)

        # 5. Crucial safety disclaimer will be added by enhance_response_with_cautions

        return "\\n".join(sections)


# --- Example Usage Block (Keep as is for testing) ---
if __name__ == '__main__':
    logger.info("Running KnowledgeQASystem directly...")

    # Check essential config more robustly
    essential_configs = ["neo4j_uri", "neo4j_user", "neo4j_password"]
    missing_configs = [cfg for cfg in essential_configs if not config.get(cfg)]
    if missing_configs:
        logger.critical(f"Missing essential configuration(s): {', '.join(missing_configs)}. Check config.py and .env file (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD). Exiting.")
        sys.exit(1)

    qa_system = None # Initialize to None
    try:
        qa_system = KnowledgeQASystem()

        if not qa_system.is_initialized:
            print("\\nSystem initialization failed. Check logs above. Exiting.")
            # qa_system.close() # close() might fail if not initialized
            sys.exit(1)
        else:
            print("\\nKnowledge QA System Ready. Enter question (or 'quit'):")

            while True:
                try:
                    user_question = input("> ")
                    if user_question.lower() in ('quit', 'exit', 'q'):
                        break
                    if not user_question.strip():
                        print("Please enter a question.")
                        continue

                    # Process the question and time it
                    answer_start_time = time.time()
                    response = qa_system.answer_question(user_question)
                    answer_end_time = time.time()

                    # Display the answer with formatting
                    print("\\n" + "="*15 + " Answer " + "="*15)
                    print(response.get('answer', 'Error: No answer generated.'))
                    print("=" * 38)

                    # Display metadata
                    print("\\n" + "="*15 + " Metadata " + "="*15)
                    print(f"Processing Time: {answer_end_time - answer_start_time:.3f}s")
                    print(f"Confidence: {response.get('confidence', 0.0):.3f}")
                    print(f"Detected Intent: {response.get('query_type', 'N/A')}")

                    # Format entities for display (show without internal synonyms)
                    display_entities = {k: v for k, v in response.get('entities', {}).items() if k != '_synonyms'}
                    entities_str = json.dumps(display_entities, indent=2)
                    print(f"Entities Found: {entities_str}")
                    print(f"DB Results Count: {response.get('results_count', 'N/A')}")

                    if response.get('error'):
                        print(f"Error Flag: True")
                    print("=" * 38)

                    print("\\nEnter question (or 'quit'):")

                except EOFError:  # Handle Ctrl+D
                    print("\\nExiting...")
                    break
                except KeyboardInterrupt:
                    print("\\nExiting...")
                    break

    except Exception as main_e:
         logger.critical(f"An error occurred during the main execution loop: {main_e}", exc_info=True)
    finally:
        if qa_system:
            qa_system.close()
            print("Knowledge QA System closed.")
        else:
             print("Knowledge QA System did not initialize properly.")