#nlp/flan_t5_processor.py
import torch
import re
import logging
import os
import random
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer, util

# from datetime import datetime

# Setup logging
logger = logging.getLogger("FlanT5Processor")
# Ensure logger is configured (this might be handled globally elsewhere)
if not logger.hasHandlers():
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FlanT5Processor:
    """
    Enhanced FlanT5 processor for generating rich, natural responses about medicinal plants.
    Leverages Neo4j relationships to provide comprehensive, contextual information.
    (Improved formatting, data handling, and fallbacks)
    """
    def __init__(self, model_name="google/flan-t5-large"):
        """Initialize the Flan-T5 model and tokenizer."""
        logger.info(f"Initializing FlanT5Processor with model: {model_name}")
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        try:
            # Defer import until needed
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)

            if torch.cuda.is_available():
                self.device = "cuda"
                self.model.to(self.device)
                logger.info(f"Flan-T5 model '{model_name}' loaded successfully on CUDA")
            else:
                logger.info(f"Flan-T5 model '{model_name}' loaded successfully on CPU")
            self._register_decision_table()

        except ImportError:
            logger.error("Required libraries not found. Install: pip install transformers torch sentencepiece")
            # Raise the error to prevent the system from continuing in a broken state
            raise ImportError("transformers library not found. Please install it.")
        except Exception as e:
            logger.error(f"Error loading Flan-T5 model '{model_name}': {e}", exc_info=True)
            # Raise the error
            raise RuntimeError(f"Failed to load Flan-T5 model: {e}")
            
        # Initialize BERT miniLM for category detection
        try:
            self.bert_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.categories = ["side effects", "benefits", "interactions", "contraindications", "safe usage"]
            self.category_embeddings = self.bert_model.encode(self.categories, convert_to_tensor=True)
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}", exc_info=True)
            self.bert_model = None
            self.categories = []
            self.category_embeddings = None  
            
    def detect_categories(self, text: str, threshold: float = 0.6) -> List[str]:
        """
        Detects keywords/categories like 'side effects', 'benefits' from free text using BERT embeddings.
        """
        if not text or not text.strip():
            return []

        sentences = text.split('.')
        extracted = set()
        for sentence in sentences:
            if not sentence.strip():
                continue
            sent_embedding = self.bert_model.encode(sentence, convert_to_tensor=True)
            cosine_scores = util.cos_sim(sent_embedding, self.category_embeddings)
            for i, score in enumerate(cosine_scores[0]):
                if score > threshold:
                    extracted.add(self.categories[i])
        return list(extracted)
    
    def _register_decision_table(self) -> None:
        """Map high-level intents to the existing formatter methods."""
        self._intent_handlers = {
            # exact Ã¢â€ â€™ single-entity formatters
            "plant_info":                  lambda entities, rows: self.format_plant_info(
                                            plant_data=(rows[0] if rows else {}), entities=entities),
            "safety_info":                 lambda entities, rows: self.format_safety_info(
                                            plant_name=self._get_primary_entity(entities, ["plants"]),
                                            safety_data=(rows[0] if rows else {}), context={}, entities=entities),
            "plant_compounds":             lambda entities, rows: self.format_plant_compounds(
                                            plant_name=self._get_primary_entity(entities, ["plants"]),
                                            compounds_data=(rows or []), entities=entities),
            "plant_preparation":           lambda entities, rows: self.format_plant_preparation(
                                            plant_name=self._get_primary_entity(entities, ["plants"]),
                                            preparations_data=(rows or []), entities=entities),

            # condition / region combos
            "condition_plants":            lambda entities, rows: self.format_condition_plants(
                                            condition=self._get_primary_entity(entities, ["conditions"]),
                                            plants_data=(rows or []), entities=entities),
            "region_plants":               lambda entities, rows: self.format_region_plants(
                                            region=self._get_primary_entity(entities, ["regions"]),
                                            plants_data=(rows or []), entities=entities),
            "region_condition_plants":     lambda entities, rows: self.format_region_condition_plants(
                                            region=self._get_primary_entity(entities, ["regions"]),
                                            condition=self._get_primary_entity(entities, ["conditions"]),
                                            plants_data=(rows or []), entities=entities),

            # compounds
            "compound_effects":            lambda entities, rows: self.format_compound_effects(
                                            compound_name=self._get_primary_entity(entities, ["compounds"]),
                                            compound_data=(rows[0] if rows else {}), entities=entities),
            "compound_plants":             lambda entities, rows: self.format_compound_plants(
                                            compound_name=self._get_primary_entity(entities, ["compounds"]),
                                            results=(rows or []), entities=entities),

            # similarity / multi-condition
            "similar_plants":              lambda entities, rows: self.format_similar_plants(
                                            target_plant=self._get_primary_entity(entities, ["plants"]),
                                            similar_plants_data=(rows or []), entities=entities),
            "multi_condition_plants":      lambda entities, rows: self.format_multi_condition_plants(
                                            conditions=(entities.get("conditions") or []),
                                            plants_data=(rows or []), entities=entities),

            # preparations for a condition
            "preparation_for_condition":   lambda entities, rows: self.format_preparation_for_condition(
                                            condition=self._get_primary_entity(entities, ["conditions"]),
                                            preparations_data=(rows or []), entities=entities),

            # graceful fallbacks / broad queries
            "keyword_search":              lambda entities, rows: self.generate_general_explanation(
                                            intent="keyword_search", entities=entities, count=len(rows or [])),
            "keyword_search_empty":        lambda entities, rows: self.generate_general_explanation(
                                            intent="keyword_search_empty", entities=entities, count=0),
            "general_query":               lambda entities, rows: self.generate_general_explanation(
                                            intent="general_query", entities=entities, count=len(rows or [])),
            "unknown":                     lambda entities, rows: self.generate_general_explanation(
                                            intent="unknown", entities=entities, count=len(rows or [])),
            "error":                       lambda entities, rows: self.generate_general_explanation(
                                            intent="error", entities=entities, count=0),
    }

    def score_pair(self, prompt: str, response: str):
        """
        Return a rough avg per-token log-prob in [-10..0].
        Start simple: give decent answers a soft boost.
        """
        try:
            # Heuristic starter: use response length as a proxy
            # (Replace this later with real sequence log-prob.)
            L = max(1, len(response.split()))
            # Map length to a pseudo logprob; longer Ã¢â€ â€™ less negative Ã¢â€ â€™ higher normalized score
            # e.g., -6.0 .. -2.0
            approx = -6.0 + min(4.0, L / 50.0 * 4.0)
            return approx
        except Exception:
            return None



    def format_answer(self, intent: str, entities: Dict[str, list], rows: Optional[list], add_caution: bool = True) -> str:
        """
        Single entry point: give me intent + entities + Neo4j rows, I return final text.
        This does not call the T5 generator; it routes to existing formatters.
        """
        handler = getattr(self, "_intent_handlers", {}).get(intent)
        if not handler:
            # Unknown intent Ã¢â€ â€™ helpful fallback
            text = self.generate_general_explanation(intent or "unknown", entities or {}, len(rows or []))
        else:
            text = handler(entities or {}, rows or [])

        return self.enhance_response_with_cautions(text) if add_caution else text

    def generate_response(self, prompt: str, max_length: int = 768, num_beams: int = 4,
                        temperature: float = 0.75, top_k: int = 50, top_p: float = 0.95,
                        attempt_retry: bool = True, **kwargs) -> str:
        """Generate a response with retry logic for numerical stability."""
        if not self.model or not self.tokenizer:
            logger.error("Model/tokenizer not initialized. Cannot generate response.")
            # Return a more informative error message if possible
            return self._fallback_error_response(context="the system components")

        logger.debug(f"Generating response for prompt (first 100 chars): {prompt[:100]}...")
        try:
            # Ensure prompt is not excessively long before tokenization
            # T5 models typically have a max sequence length (e.g., 512 or 1024)
            # Truncate the *input* prompt if necessary, although the formatter should ideally handle this.
            # Using 1024 as a safe default max input length for T5.
            inputs = self.tokenizer(prompt, max_length=1024, truncation=True, return_tensors="pt").to(self.device)

            generation_params = {
                "max_length": max_length,
                "min_length": 30,  # Reduced min_length slightly for flexibility
                "num_beams": num_beams,
                "no_repeat_ngram_size": 3, # Prevent repetitive phrases
                "early_stopping": True,    # Stop when EOS token is generated
                "do_sample": True,         # Enable sampling for more varied output
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": 1.1, # Slightly discourage repetition
                **kwargs
            }

            with torch.no_grad():
                outputs = self.model.generate(inputs.input_ids, **generation_params)

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.debug(f"Generated response length: {len(response)}")
            return self._clean_response(response)

        except RuntimeError as e:
            # Handle potential CUDA OOM errors or generation instability
            if "CUDA out of memory" in str(e):
                 logger.error(f"CUDA OOM Error during generation: {e}. Try reducing max_length or batch size if applicable.")
                 return self._fallback_error_response(context="due to resource constraints")
            elif "probability tensor contains" in str(e) and attempt_retry:
                logger.warning(f"Generation instability detected: {e}. Retrying without sampling...")
                try:
                    # Fallback to beam search without sampling
                    generation_params["do_sample"] = False
                    generation_params.pop("temperature", None)
                    generation_params.pop("top_k", None)
                    generation_params.pop("top_p", None)
                    generation_params["num_beams"] = max(2, num_beams // 2) # Reduce beams slightly

                    with torch.no_grad():
                        outputs = self.model.generate(inputs.input_ids, **generation_params)

                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.info("Retry generation successful.")
                    return self._clean_response(response)
                except Exception as retry_e:
                    logger.error(f"Retry failed: {retry_e}", exc_info=True)
                    return self._fallback_error_response(context="after encountering an issue")
            else:
                logger.error(f"Unexpected RuntimeError during generation: {e}", exc_info=True)
                return self._fallback_error_response()
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return self._fallback_error_response()

    def _clean_response(self, response: str) -> str:
        """Clean and format the generated response."""
        response = response.replace("<pad>", "").replace("</s>", "").strip()
        # Consolidate whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        # Replace common N/A phrases more naturally
        response = re.sub(r'\b(N/A|None listed|Not specified)\b', 'information not available in the database', response, flags=re.IGNORECASE)
        # Fix spacing around punctuation
        response = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', response) # Add space after punctuation
        response = re.sub(r'\s+\)', ')', response) # Remove space before closing parenthesis
        response = re.sub(r'\(\s+', '(', response) # Remove space after opening parenthesis
        # Capitalize the first letter of the response
        if response:
            response = response[0].upper() + response[1:]
        # Ensure consistent line breaks before bullet points if any exist
        response = re.sub(r'\s*â€¢\s*', r'\nâ€¢ ', response)
        response = re.sub(r'\s*-\s*', r'\n- ', response) # Also handle hyphens as bullets
        return response.strip() # Remove leading/trailing whitespace again

    def _fallback_error_response(self, context: str = "") -> str:
        """Generate a context-aware error message."""
        base_message = "I encountered a technical issue while generating a response"
        if context:
            base_message += f" {context}"
        base_message += ". Please try rephrasing your question or ask about something else."
        # Add the standard caution as errors might prevent it elsewhere
        return self.enhance_response_with_cautions(base_message, force_add=True)

    def enhance_response_with_cautions(self, base_response: str, force_add: bool = False) -> str:
        #Add appropriate cautions to the response if not already present.
        caution_keywords = [
            'consult', 'professional', 'advice', 'doctor', 'healthcare', 'qualified',
            'caution', 'warning', 'interaction', 'side effect', 'risk',
            'pregnant', 'nursing', 'condition', 'medication', 'medical advice'
        ]

        response_lower = base_response.lower()
        # Check if a substantial caution message seems to be present
        has_caution = any(keyword in response_lower for keyword in caution_keywords)
        # Also check for longer phrases
        has_caution_phrase = "consult with a qualified healthcare professional" in response_lower or \
                             "not medical advice" in response_lower

        if force_add or not (has_caution and has_caution_phrase):
            caution = "\n\n**Important Note:** Information provided is based on available data and is for educational purposes only. It is not intended as medical advice. "
            #caution += "Always consult with a qualified healthcare professional before using any medicinal plant, "
            #caution += "especially if you have underlying health conditions, are pregnant or nursing, are taking other medications, or considering combining treatments. "
            #caution += "Individual responses can vary, and proper identification, dosage, and preparation are crucial for safety and effectiveness."

            # Avoid adding duplicate cautions if base_response already ends with one
            if not base_response.strip().endswith(caution.strip()):
                 return base_response.strip() + caution # Ensure no leading/trailing space on base
            else:
                 return base_response # Already has it
        else:
            logger.debug("Caution keywords detected, skipping addition of standard note.")
            return base_response # Already has sufficient caution

    def _format_no_data_response(self, entity_type: str, entity_name: Optional[str]) -> str:
        """Generates a response when no data is found for a specific entity."""
        if entity_name:
            return f"I couldn't find specific information about the {entity_type} '{entity_name}' in my database. This doesn't necessarily mean it's not used or relevant, only that it's not detailed in my current data. You could try asking about related {entity_type}s or broader categories."
        else:
            return f"I couldn't find specific information for the requested {entity_type}. Please try specifying the {entity_type} you're interested in."

    #kylie added
    def _format_list(self, items: List[str], max_items: Optional[int] = None) -> str:
        """Format a list into a human-readable string with commas and 'and'. Optionally cap length."""
        cleaned_items = [item.strip() for item in items if item and str(item).strip()]
        if not cleaned_items:
            return ""
        if isinstance(max_items, int) and max_items > 0 and len(cleaned_items) > max_items:
            cleaned_items = cleaned_items[:max_items]
        if len(cleaned_items) == 1:
            return cleaned_items[0]
        if len(cleaned_items) == 2:
            return f"{cleaned_items[0]} and {cleaned_items[1]}"
        return ", ".join(cleaned_items[:-1]) + ", and " + cleaned_items[-1]


    def _format_list_bullets(self, items: Optional[List[str]], max_items: int = 5) -> str:
        """Formats a list for display, handling empty lists and limiting items."""
        if not items:
            return "None listed in database."
        # Filter out empty strings or None values
        cleaned_items = [str(item).strip() for item in items if item and str(item).strip()]
        if not cleaned_items:
             return "None listed in database."

        if len(cleaned_items) > max_items:
            display_items = cleaned_items[:max_items]
            return "\n".join(f"â€¢ {item.capitalize()}" for item in display_items) + f"\nâ€¢ ... (and {len(cleaned_items) - max_items} more)"
        else:
            return "\n".join(f"â€¢ {item.capitalize()}" for item in cleaned_items)

    #kylie added
    def _format_list_capitalize(self, items: List[str]) -> str:
        """Format a list, but capitalize the first instance"""
        formatted = self._format_list(items)
        if formatted:
            formatted = formatted[0].upper() + formatted[1:]
        return formatted + '.'

    #kylie added
    def _clean_sentence(self, text: str) -> str:
        text = text.strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        if text and not text.endswith('.'):
            text += '.'
        return text

    def _format_with_commas(self, plants):
        if not plants:
            return ""
        if len(plants) == 1:
            return plants[0]
        elif len(plants) == 2:
            return f"{plants[0]} and {plants[1]}"
        else:
            return f"{', '.join(plants[:-1])}, and {plants[-1]}"

    # --- Specific Formatters ---

    def format_plant_info(self, plant_data: dict, entities: dict) -> str:
        """
        Creates a comprehensive response about a medicinal plant.
        Dynamically tailors sections based on extracted keywords (e.g. 'benefits', 'compounds', 'side effects').
        """
        if not plant_data or not plant_data.get('name'):
            queried_plant = self._get_primary_entity(entities, ['plants'])
            return self._format_no_data_response("plant", queried_plant)

        # Extract user keywords
        keywords = [k.lower() for k in entities.get('keywords', [])]
        logger.info(f"[T5 Debug] Keywords for plant_info: {keywords}")

        # Helper to check if any keyword matches a cue group
        def has_kw(*terms):
            return any(any(t in kw for t in terms) for kw in keywords)

        # Extract plant fields safely
        name = plant_data.get('name', 'Unknown Plant')
        sci_name = plant_data.get('scientific_name')
        family = plant_data.get('family')
        morphology = plant_data.get('morphology')
        distribution = plant_data.get('distribution')
        effects = [e for e in plant_data.get('effects', []) if e and str(e).strip()]
        compounds = [c for c in plant_data.get('compounds', []) if c and str(c).strip()]
        preparations = [p for p in plant_data.get('preparations', []) if p and str(p).strip()]
        side_effects = [s for s in plant_data.get('side_effects', []) if s and str(s).strip()]
        traditional_uses = [t for t in plant_data.get('traditional_uses', []) if t and str(t).strip()]
        contraindications = [c for c in plant_data.get('contraindications', []) if c and str(c).strip()]

        # Decide which sections to show
        show_effects = has_kw("benefit", "effect", "property", "use")
        show_general = not (show_effects)

        sections = []
            
        # --- Effects/benefits
        if show_effects and effects:
            sections.append(f"\n**Medicinal Benefits & Effects of {name.title()}:**\n{self._format_list_bullets(effects)}")

        # --- Fallback: show all sections if no keywords guide it
        if show_general:
            # --- Intro section ---
            intro_parts = [f"**{name}**"]
            if sci_name:
                intro_parts[-1] += f" ({sci_name})"
            intro_parts.append(
                f" is a medicinal plant from the {family} family."
                if family else " is a notable medicinal plant."
            )
            sections.append("".join(intro_parts))

            # --- Description section ---
            desc_parts = []
            if morphology:
                desc_parts.append(f"**Description:** {morphology}")
            if distribution:
                desc_parts.append(f"**Distribution:** {distribution}")
            if desc_parts:
                sections.append("\n".join(desc_parts))

            # --- Info sections defined as label â†’ data mapping ---
            info_map = {
                "Medicinal Properties & Effects": effects,
                "Traditional Uses": traditional_uses,
                "Key Active Compounds": compounds,
                "Preparation Methods": preparations,
            }

            for title, data in info_map.items():
                if data:
                    sections.append(f"\n**{title}:**\n{self._format_list_capitalize(data)}")

            # --- Side effects section ---
            if side_effects:
                cleaned = [self._clean_sentence(s) for s in side_effects]
                sections.append(f"\n**Known Side Effects:**\n{self._format_list_bullets(cleaned)}")

        return "\n".join(sections)

    def format_condition_plants(self, condition: str, plants_data: list, entities: dict) -> str:
        """
        Creates an informative response about plants used for a specific condition. (Improved Logic)
        Groups plants by relevance and includes preparation information.
        """
        if not condition:
             return "Please specify the condition you are interested in."
        if not plants_data:
            return self._format_no_data_response("condition", condition)

        condition_lower = condition.lower()
        # Use synonyms for matching effects
        synonyms = entities.get('_synonyms', {}).get(condition, [condition_lower]) # Assuming synonyms are passed via entities dict

        primary_plants = {}
        supportive_plants = {}

        # Process and deduplicate plants_data first
        unique_plants = {}
        for plant in plants_data:
            plant_name = plant.get('p_name') or plant.get('name')
            if not plant_name or not str(plant_name).strip(): continue # Skip if name is missing
            plant_name = str(plant_name).strip()

            # Aggregate effects for the same plant
            if plant_name not in unique_plants:
                unique_plants[plant_name] = {
                    'name': plant_name,
                    'scientific_name': plant.get('p_scientific_name') or plant.get('scientific_name'),
                    'effects': set() # Use a set to store unique effects
                }
            plant_effects = plant.get('effects', [])
            if isinstance(plant_effects, list): # Handle list of effects
                 current_effects = {str(e).lower().strip() for e in plant_effects if e and str(e).strip()}
            elif isinstance(plant_effects, str) and plant_effects.strip(): # Handle single string effect
                 current_effects = {plant_effects.lower().strip()}
            else: # Handle other cases like single effect in 'effect' key
                 effect = plant.get('effect')
                 current_effects = {str(effect).lower().strip()} if effect and str(effect).strip() else set()

            unique_plants[plant_name]['effects'].update(current_effects)


        # Categorize plants based on aggregated effects
        for plant_name, plant_info in unique_plants.items():
            plant_effects = plant_info['effects']
            # Check if any effect directly matches the condition or its synonyms
            is_primary = any(cond_syn in effect for effect in plant_effects for cond_syn in synonyms)

            plant_display = {
                'name': plant_info['name'],
                'scientific_name': plant_info['scientific_name'],
                'effects_sample': sorted(list(plant_effects))[:3] # Show a sample of effects
            }

            if is_primary:
                primary_plants[plant_name] = plant_display
            else:
                # Only add as supportive if it has *any* effects listed
                if plant_effects:
                     supportive_plants[plant_name] = plant_display


        # Build response
        sections = []
        intro = f"Here's information on medicinal plants used to treat **{condition}**:"
        sections.append(intro)

        # Primary plants section
        if primary_plants:
            sections.append("\n**Plants Used:**")

            #Randomly choose 10 plants from database relating to condition
            selected_plants = random.sample(
                list(primary_plants.values()),
                min(10, len(primary_plants))
            )

            selected_plants.sort(key=lambda p: p['name'])
            for plant in selected_plants:
                plant_text = f"â€¢ **{plant['name']}**"
                if plant['scientific_name']: plant_text += f" (*{plant['scientific_name']}*)\n"
                #if plant['effects_sample']: plant_text += f"\n  Known effects include: {', '.join(plant['effects_sample'])}"
                sections.append(plant_text)
        else:
             sections.append(f"\nNo plants specifically listed for directly treating '{condition}' were found in the database based on the available effect descriptions. However, some plants may offer supportive benefits.")

        # Supportive plants section
        if supportive_plants:
            sections.append("\n**Other Potentially Supportive Plants:**")
             # Sort alphabetically by name
            for name in sorted(supportive_plants.keys()):
                plant = supportive_plants[name]
                plant_text = f"\nâ€¢ **{plant['name']}**"
                if plant['scientific_name']: plant_text += f" ({plant['scientific_name']})"
                if plant['effects_sample']: plant_text += f"\n  Related benefits may include: {', '.join(plant['effects_sample'])}"
                sections.append(plant_text)

        # Usage guidelines (General)
        guidelines = "\n**General Usage Considerations:**"
        guidelines += "\nâ€¢ Effectiveness can vary based on the plant part used, preparation, dosage, and individual factors."
        guidelines += "\nâ€¢ Start with a single plant and a low dose to assess your response."
        guidelines += "\nâ€¢ Follow traditional or expert-recommended preparation methods."
        guidelines += "\nâ€¢ Ensure proper identification and use high-quality plant sources."
        #sections.append(guidelines)

        # Safety note is added globally by enhance_response_with_cautions

        return "\n".join(sections)

    def format_region_condition_plants(self, region: str, condition: str, plants_data: list, entities: dict) -> str:
        """
        Creates a detailed response about plants from a specific region used for a specific condition. (Improved)
        Includes traditional uses and cultural context.
        """
        if not region or not condition:
             return "Please specify both the region and the condition you are interested in."
        if not plants_data:
            # Use a more informative no-data response
            return f"While plants from **{region}** are likely used for **{condition}**, I don't have specific records linking plants from this region directly to this condition in my database. You could try asking about plants for '{condition}' generally, or plants found in '{region}'."

        condition_lower = condition.lower()
        region_lower = region.lower()
        synonyms = entities.get('_synonyms', {}).get(condition, [condition_lower])

        # Organize plants data - deduplicate and aggregate
        plants_info = {}
        for plant in plants_data:
            name = plant.get('plant_name') or plant.get('name')
            if not name or not str(name).strip(): continue
            name = str(name).strip()

            if name not in plants_info:
                plants_info[name] = {
                    'scientific_name': plant.get('scientific_name'),
                    'effects': set(),
                    'traditional_uses': set(), # Assuming these might come from the query
                    'preparations': set()      # Assuming these might come from the query
                }

            # Aggregate effects
            effect = plant.get('effect')
            if effect and str(effect).strip():
                plants_info[name]['effects'].add(str(effect).strip())
            # Add other potential fields if they exist in query results
            trad_uses = plant.get('traditional_uses', [])
            if isinstance(trad_uses, list): plants_info[name]['traditional_uses'].update(str(t).strip() for t in trad_uses if t and str(t).strip())
            preps = plant.get('preparation_methods', []) # Or 'preparations'
            if isinstance(preps, list): plants_info[name]['preparations'].update(str(p).strip() for p in preps if p and str(p).strip())


        # Categorize plants based on aggregated effects
        relevant_plants = {}
        for name, info in plants_info.items():
            effects_lower = {e.lower() for e in info['effects']}
            # Check if any effect matches the condition or its synonyms
            if any(cond_syn in effect for effect in effects_lower for cond_syn in synonyms):
                 relevant_plants[name] = info # Keep all info for relevant plants

        # Build response
        sections = []
        if not relevant_plants:
             # If aggregation removed all plants, use the no-data message
             return f"While plants from **{region}** are likely used for **{condition}**, I don't have specific records linking plants from this region directly to this condition in my database after filtering. You could try asking about plants for '{condition}' generally, or plants found in '{region}'."

        intro = f"Here are some medicinal plants found in **{region.title()}** traditionally associated with **{condition}**:"
        sections.append(intro)

        # List relevant plants
        sections.append("\n**Relevant Plants:**")

        if relevant_plants:
            selected_plants = random.sample(
                list(relevant_plants.items()),
                min(10, len(relevant_plants))
            )

            selected_plants.sort(key=lambda item: item[0])

            for name, info in selected_plants:
                line = f"â€¢ **{name}**"
                if info['scientific_name']: line += f" ({info['scientific_name']})"
                # Show effects relevant to the condition first, then others
                matched_effects = sorted([e for e in info['effects'] if any(cond_syn in e.lower() for cond_syn in synonyms)])
                other_effects = sorted([e for e in info['effects'] if e not in matched_effects])[:2] # Limit other effects shown

                if matched_effects: line += f"\n  *Relevant Effects:* {', '.join(matched_effects)}"
                if other_effects: line += f"\n  *Other Effects:* {', '.join(other_effects)}"
                if info['traditional_uses']: line += f"\n  *Traditional Uses:* {self._format_list(list(info['traditional_uses']), max_items=3)}"
                if info['preparations']: line += f"\n  *Preparation Methods:* {self._format_list(list(info['preparations']), max_items=3)}"
                sections.append(line + "\n")

        # Regional context (simplified)
        context = f"\n**Context for {region}:**"
        context += "\nâ€¢ Traditional knowledge in this region often guides the specific use and preparation of these plants."
        context += "\nâ€¢ Local environmental factors can influence the potency and characteristics of plants."
        context += "\nâ€¢ Sustainable harvesting is important to preserve these resources."
        sections.append(context)

        # Considerations (General)
        considerations = "\n**Important Considerations:**"
        considerations += "\nâ€¢ This information is based on available data; local practices may vary."
        considerations += "\nâ€¢ Always ensure correct plant identification."
        considerations += "\nâ€¢ Preparation methods significantly impact effectiveness and safety."
        sections.append(considerations)

        # Safety note added globally

        return "\n".join(sections)
        
    def _debug_keywords_section(self, entities, label="Detected keywords"):
        kws = [k for k in (entities or {}).get("keywords", []) if k]
        return f"\n\n_{label}:_ " + ", ".join(kws) if kws else ""

    def format_safety_info(self, plant_name: str, safety_data: dict, context: dict, entities: dict) -> str:
        """
        Generates a safety/benefit/interactions report for a plant.
        Dynamically filters sections based on extracted keywords from BERT.
        """
        if not plant_name:
            return "Please specify the plant you want safety information for."

        safety_data = safety_data or {}
        keywords = [k.lower() for k in entities.get("keywords", [])]
        logger.info(f"[T5 Debug] Keywords for safety_info: {keywords}")

        # Helper to check keyword intent
        def has_kw(*terms):
            return any(any(t in kw for t in terms) for kw in keywords)

        # --- Decide which categories to display ---
        show_side_effects = has_kw("side effect", "adverse", "risk", "danger", "toxicity", "caution", "warning")
        show_interactions = has_kw("interaction", "combine", "mix", "drug", "medication")
        show_contraindications = has_kw("contraindication", "avoid", "pregnant", "nursing", "not use", "should not")
        show_benefits = has_kw("benefit", "effect", "use", "property", "helps", "good for")
        show_general = not (show_side_effects or show_interactions or show_contraindications or show_benefits)

        # --- Extract fields ---
        side_effects = [s for s in safety_data.get("side_effects", []) if s and str(s).strip()]
        interactions = [i for i in safety_data.get("interactions", []) if i and str(i).strip()]
        contraindications = [c for c in safety_data.get("contraindications", []) if c and str(c).strip()]
        benefits = [b for b in safety_data.get("benefits", []) if b and str(b).strip()]  # optional

        sections = []

        # --- Side Effects ---
        if show_side_effects or show_general:
            header = f"**Side Effects of {plant_name.title()}:**"
            if side_effects:
                formatted = self._format_list_bullets([self._clean_sentence(s) for s in side_effects])
            else:
                formatted = "â€¢ No specific side effects listed, but caution is advised."
            sections.append(f"{header}\n{formatted}")

        # --- Interactions ---
        if show_interactions or show_general:
            header = f"**Known Interactions for {plant_name.title()}:**"
            if interactions:
                formatted = self._format_list_bullets([self._clean_sentence(i) for i in interactions])
            else:
                formatted = "â€¢ No known interactions recorded, but consult a healthcare provider before combining with medications."
            sections.append(f"{header}\n{formatted}")

        # --- Contraindications ---
        if show_contraindications or show_general:
            header = f"**Contraindications (When to Avoid) for {plant_name.title()}:**"
            if contraindications:
                formatted = self._format_list_bullets([self._clean_sentence(c) for c in contraindications])
            else:
                formatted = "â€¢ No specific contraindications listed, but use caution during pregnancy, nursing, or with chronic conditions."
            sections.append(f"{header}\n{formatted}")

        # --- If no sections built ---
        if not sections:
            return (f"Safety information for {plant_name.title()} is limited in the database. "
                    "Always monitor for unusual reactions and consult a qualified healthcare professional before use.")

        return "\n\n".join(sections)

    def format_compound_effects(self, compound_name: str, compound_data: dict, entities: dict) -> str:
        """
        Creates a detailed response about a compound's effects and properties. (Improved)
        Includes scientific context and practical information.
        """
        if not compound_name:
             return "Please specify the compound you are interested in."
        if not compound_data:
            return self._format_no_data_response("compound", compound_name)

        # Extract compound information
        compound_name_display = compound_data.get('compound_name', compound_name) # Use name from data if available
        effects = [e for e in compound_data.get('effects', []) if e and str(e).strip()]
        plants = [p for p in compound_data.get('found_in_plants', []) if p and str(p).strip()]
        interactions = [i for i in compound_data.get('interactions', []) if i and str(i).strip()] # Assuming this might exist
        research_notes = [n for n in compound_data.get('research_notes', []) if n and str(n).strip()] # Assuming this might exist

        sections = []

        # Introduction
        intro = f"**{compound_name_display.title()}** is a bioactive compound found in various medicinal plants."
        

        if plants:
            intro += f" It is notably present in plants such as {self._format_with_commas(plants[:3])}"
            if len(plants) > 3: 
                intro += f", among others."
            else: 
                intro += "."
        sections.append(intro)

        # Therapeutic effects
        if effects:
            sections.append(f"\n**Known Therapeutic Properties & Effects:**\n{self._format_list_capitalize(effects[:10])}")
        else:
             sections.append(f"\n**Known Therapeutic Properties & Effects:** Specific effects associated directly with {compound_name_display} are not detailed in the database, but it contributes to the overall activity of the plants it's found in.")

        # Source plants
        if plants:
            sections.append(f"\n**Natural Plant Sources:**\n{self._format_list_capitalize(plants)}")

        # Scientific context / How it works (General statements)
        science = f"\n**Understanding {compound_name_display.capitalize()}'s Role:**"
        science += f"\nâ€¢ The effects of {compound_name_display.title()} often result from synergy with other compounds in the plant."
        science += "\nâ€¢ Its concentration and bioavailability can vary significantly based on the plant source, part used, and preparation method."
        science += "\nâ€¢ Research is ongoing to fully understand its mechanisms of action."
        sections.append(science)

        # Research insights (If available)
        if research_notes:
            sections.append(f"\n**Research Insights:**\n{self._format_list(research_notes)}")

        # Interactions (If available)
        if interactions:
            sections.append(f"\n**Potential Interactions:**\n{self._format_list(interactions)}")

        # Practical considerations
        practical = "\n**Practical Considerations:**"
        practical += f"\nâ€¢ The amount of {compound_name_display.title()} can differ greatly between plant batches and preparations."
        practical += "\nâ€¢ Extraction methods (e.g., water, alcohol, oil) influence which compounds are present in the final product."
        practical += "\nâ€¢ Consider the plant source as a whole, not just the isolated compound, for traditional use."
        sections.append(practical)

        # Safety note added globally

        return "\n".join(sections)

    def format_compound_plants(self, compound_name: str, results: List[dict], entities: dict) -> str:
        """
        Creates a comprehensive response about plants containing a specific compound. (Improved Formatting)
        Includes concentration information and practical uses.
        """
        if not compound_name:
            return "Please specify the compound you are interested in."
        if not results:
            compound_synonyms = entities.get('_synonyms', {}).get(compound_name, [compound_name])
            if any(c in entities.get('compounds', []) for c in compound_synonyms):
                return f"While **{compound_name}** is a known compound, I couldn't find specific plants listed as containing it in the database."
            else:
                return self._format_no_data_response("compound", compound_name)

        # ---normalize record shape ---
        normalized_results = []
        for r in results:
            # Handle both query formats: some return a list of plants, others one per record
            plants_field = r.get("found_in_plants") or r.get("plant_name") or []
            if isinstance(plants_field, list):
                for plant in plants_field:
                    normalized_results.append({
                        "plant_name": plant,
                        "scientific_name": r.get("scientific_name"),
                        "effects": r.get("effects", []),
                        "concentration": r.get("concentration"),
                        "preparations": r.get("preparations", [])
                    })
            else:
                normalized_results.append({
                    "plant_name": plants_field,
                    "scientific_name": r.get("scientific_name"),
                    "effects": r.get("effects", []),
                    "concentration": r.get("concentration"),
                    "preparations": r.get("preparations", [])
                })

        # --- Aggregate plant data ---
        plant_details = {}
        all_effects = set()

        for result in normalized_results:
            plant_name = result.get("plant_name")
            if not plant_name or not str(plant_name).strip():
                continue
            plant_name = str(plant_name).strip()

            if plant_name not in plant_details:
                plant_details[plant_name] = {
                    "scientific_name": result.get("scientific_name"),
                    "effects": set(),
                    "concentration": result.get("concentration"),
                    "preparations": result.get("preparations", [])
                }

            effects = result.get("effects", [])
            if isinstance(effects, list):
                clean = {str(e).strip() for e in effects if e and str(e).strip()}
                plant_details[plant_name]["effects"].update(clean)
                all_effects.update(clean)
            elif isinstance(effects, str) and effects.strip():
                plant_details[plant_name]["effects"].add(effects.strip())
                all_effects.add(effects.strip())

        # --- Build response ---
        if not plant_details:
            return f"I couldn't find specific plants listed as containing **{compound_name.title()}** in the database."

        sections = [f"Here are medicinal plants known to contain the compound **{compound_name.title()}**:"]
        sections.append(f"\n**Plants Containing {compound_name.title()}:**")

        for name in sorted(plant_details.keys()):
            details = plant_details[name]
            line = f"â€¢ {name}"
            sections.append(line)

        return "\n".join(sections)


    def generate_general_explanation(self, intent: str, entities: Dict[str, List[str]], count: int) -> str:
        """
        Generates a helpful explanation when specific data isn't found or query is too broad. (Improved)
        Provides context and suggestions for further exploration.
        """
        entity_list = []
        for category, items in entities.items():
            # Exclude internal keys like _synonyms
            if items and category != '_synonyms':
                entity_list.append(f"{category}: {', '.join(items)}")
        entity_str = "; ".join(entity_list) if entity_list else "your query"

        response = ""
        if count == 0 and intent != 'error' and intent != 'keyword_search_empty' and intent != 'unknown' and intent != 'general_query':
            # Case: Valid intent/entities, but DB query returned nothing specific
            response = f"I understand you're asking about {entity_str}. "
            response += "While I recognize the topic, I couldn't find specific matching details in my database based on your query. "
            response += "This could mean the specific combination isn't recorded, or the information is structured differently."

        elif intent == 'keyword_search_empty' or intent == 'unknown' or intent == 'general_query':
             # Case: Query was too vague or nonsensical
             response = f"I understand you're interested in {entity_str}, but your query seems a bit broad or unclear. "
             response += "To provide a helpful answer, could you please be more specific?"

        else: # Fallback for other cases (e.g., formatter failed despite having results)
            response = f"I found some information related to {entity_str} ({count} potential entries), but I encountered difficulty formatting a specific answer. "
            response += "This might be due to the complexity of the request or data structure."

        # Add suggestions regardless of the specific case above
        response += "\n\nTo help me find what you need, you could try asking about:"
        response += "\nâ€¢ A specific plant's properties (e.g., 'What are the benefits of Ginger?')"
        response += "\nâ€¢ Plants for a specific condition (e.g., 'Which herbs help with headaches?')"
        response += "\nâ€¢ Safety information for a plant (e.g., 'Is St. John's Wort safe?')"
        response += "\nâ€¢ Preparation methods (e.g., 'How to prepare Echinacea tea?')"
        response += "\nâ€¢ Plants from a specific region (e.g., 'Medicinal plants from the Andes')"

        # Safety note added globally

        return response


    def format_similar_plants(self, target_plant: str, similar_plants_data: list, entities: dict) -> str:
        """
        Creates an engaging response about plants with similar therapeutic effects. (Improved)
        Includes detailed comparisons and usage considerations.
        """
        if not target_plant:
            return "Please specify the plant you want to find similar plants for."
        if not similar_plants_data:
            # Provide a more specific no-data response
            return f"I couldn't find specific plants listed as having significantly similar *therapeutic effects* to **{target_plant.title()}** based on the data available. Plants can be similar in other ways (e.g., appearance, family) not covered here."

        # Process and structure data
        similar_plants_info = {}
        for plant_data in similar_plants_data:
            name = plant_data.get('p2_name')
            if not name or not str(name).strip(): continue
            name = str(name).strip()

            similar_plants_info[name] = {
                'scientific_name': plant_data.get('p2_scientific_name'),
                'shared_effects': [e for e in plant_data.get('shared_effects', []) if e and str(e).strip()],
                'similarity_score': plant_data.get('similarity_score', 0) # Use consistent key name
                # Add preparations if available from query
                # 'preparations': [p for p in plant_data.get('preparations', []) if p]
            }

        # Build response
        sections = []
        intro = f"Here are some plants that share similar **therapeutic effects** with **{target_plant.title()}**:"
        sections.append(intro)

        # List similar plants, potentially sorted by similarity score
        sorted_plants = sorted(similar_plants_info.items(), key=lambda item: item[1]['similarity_score'], reverse=True)
        
        display_plants = random.sample(sorted_plants, min(10, len(sorted_plants)))

        sections.append("\n**Similar Plants:**")
        for name, info in display_plants:
            line = f"â€¢ **{name}**"
            if info['scientific_name']: line += f" (*{info['scientific_name']}*)"
            if info['shared_effects']:
                line += f"\n  *Shared Effects Include:* {', '.join(info['shared_effects'][:3])}\n" # Limit displayed effects
            # Could add similarity score if meaningful: line += f" (Similarity Score: {info['similarity_score']})"
            sections.append(line)

        # Important differences
        differences = "\n**Important Differences to Consider:**"
        differences += f"\nâ€¢ While sharing some effects, these plants have unique chemical profiles and other distinct properties not shared with {target_plant.title()}."
        differences += "\nâ€¢ Potency, optimal preparation methods, and safety profiles (side effects, interactions) can differ significantly."
        differences += "\nâ€¢ Traditional uses might vary even if some effects overlap."
       # sections.append(differences)

        # Selection guidelines
        guidelines = f"\n**Guidelines for Choosing:**"
        guidelines += "\nâ€¢ Base your choice on your specific health goals and the *full* profile of the plant, not just the overlap."
        guidelines += "\nâ€¢ Consider availability, ease of preparation, and your individual sensitivities."
        guidelines += f"\nâ€¢ Research each plant individually, including its specific safety information, before use."
        #sections.append(guidelines)

        # Safety note added globally

        return "\n".join(sections)

    def format_preparation_methods(self, target_entity: str, prep_data: list, is_condition_query: bool, entities: dict) -> str:
        """
        Creates a detailed response about preparation methods. (Improved Formatting)
        Includes traditional techniques, tips, and safety considerations.
        """
        if not target_entity:
             return "Please specify the plant or condition you want preparation methods for."
        if not prep_data:
            entity_type = "condition" if is_condition_query else "plant"
            # More specific no-data message
            return f"I couldn't find specific preparation methods listed in the database for the {entity_type} **{target_entity}**."

        # Organize preparation methods
        methods = {}
        for method_info in prep_data:
            name = method_info.get('preparation_method')
            # Filter out invalid or placeholder names more strictly
            if not name or not str(name).strip() or str(name).strip().lower() in ['unknown method', 'none', 'n/a']: continue
            name = str(name).strip()

            # Aggregate example plants for the same method
            if name not in methods:
                methods[name] = {
                    'description': method_info.get('description'), # Assuming this might exist
                    'plant_count': method_info.get('plant_count', 0), # Assuming this might exist
                    'example_plants': set()
                }
            # Add example plants safely
            plants = method_info.get('example_plants', [])
            if isinstance(plants, list):
                 methods[name]['example_plants'].update(str(p).strip() for p in plants if p and str(p).strip())

        # Build response
        sections = []
        # Introduction
        if is_condition_query:
            intro = f"Here are traditional preparation methods commonly associated with treating **{target_entity}**:"
        else:
            intro = f"Here are traditional preparation methods used for **{target_entity.title()}**:"
        sections.append(intro)

        # List Methods
        sections.append("\n**Preparation Methods:**")
        if not methods: # Check if aggregation removed all methods
             return f"I couldn't find specific preparation methods listed in the database for **{target_entity}** after processing."

        for method_name in sorted(methods.keys()):
            details = methods[method_name]
            line = f"â€¢ {method_name.capitalize()}"
            if details['description'] and str(details['description']).strip():
                 line += f": {str(details['description']).strip()}"
            # Display example plants if available
            if details['example_plants']:
                 example_list = sorted(list(details['example_plants']))
                 line += f"\n  *Example Plants:* {', '.join(example_list[:5])}" # Limit examples shown
                 if len(example_list) > 5: line += f", ..."
            # Could add plant count if meaningful: if details['plant_count'] > 0: line += f" (Used with ~{details['plant_count']} plants)"
            sections.append(line)

        # Key Factors
        quality = "\n**Key Factors for Preparation:**"
        quality += "\nâ€¢ **Plant Material:** Use high-quality, correctly identified plants from a reliable source."
        quality += "\nâ€¢ **Plant Part:** Different parts (root, leaf, flower) often require different methods."
        quality += "\nâ€¢ **Solvent:** Water, alcohol, oil, or vinegar extract different compounds."
        quality += "\nâ€¢ **Technique:** Proper temperature, timing, and equipment are crucial."
        quality += "\nâ€¢ **Storage:** Store preparations correctly (e.g., cool, dark place) to maintain potency."
        sections.append(quality)

        # Method Selection (General advice)
        selection = "\n**Choosing the Right Method:**"
        selection += "\nâ€¢ The best method depends on the specific plant, the desired effects, and the compounds being targeted."
        selection += "\nâ€¢ Infusions (teas) are common for leaves and flowers; decoctions (simmering) for roots and barks; tinctures (alcohol extraction) for broader compound extraction."
        selection += "\nâ€¢ Consult reliable herbal resources or practitioners for guidance specific to the plant or condition."
        #sections.append(selection)

        # Safety note added globally

        return "\n".join(sections)

    # --- Simpler Formatters (Minor improvements) ---

    def format_multi_condition_plants(self, conditions: list, plants_data: list, entities: dict) -> str:
        """
        Creates a response about plants that address multiple conditions simultaneously. (Improved)
        Groups plants by which conditions they address and highlights overlaps.
        """
        if not conditions or len(conditions) < 2:
             return "Please specify at least two conditions."
        if not plants_data:
            return f"I couldn't find specific plants listed in the database that are known to address **{', '.join(conditions)}** simultaneously based on the available data."

        conditions_lower = [c.lower() for c in conditions]
        sections = []
        intro = f"Here are plants potentially relevant to **{self._format_with_commas(conditions)}**:"
        sections.append(intro)
        sections.append("\n**Relevant Plants:**")

        plant_info = {}
        for plant in plants_data:
             name = plant.get('p_name') or plant.get('name')
             if not name or not str(name).strip(): continue
             name = str(name).strip()

             if name not in plant_info:
                  plant_info[name] = {
                       'scientific_name': plant.get('p_scientific_name') or plant.get('scientific_name'),
                       'effects': set()
                  }
             # Aggregate effects
             effects = plant.get('effects', [])
             if isinstance(effects, list):
                  plant_info[name]['effects'].update(str(e).lower().strip() for e in effects if e and str(e).strip())
             elif isinstance(effects, str) and effects.strip():
                  plant_info[name]['effects'].add(effects.lower().strip())

        selected_plants = sorted(
            random.sample(
                list(plant_info.items()),
                min(10, len(plant_info))
            ),
            key=lambda item: item[0]
        )

        found_plants = False
        for name, info in selected_plants:
             # Find which of the queried conditions are matched by the plant's effects
             relevant_conditions_matched = {
                 cond for cond in conditions
                 if any(cond.lower() in effect for effect in info['effects'])
             }

             if relevant_conditions_matched: # Only list plants that match at least one condition
                 found_plants = True
                 line = f"\nâ€¢ **{name}**"
                 if info['scientific_name']: line += f" ({info['scientific_name']})"
                 line += f"\n  *Relevant Conditions Addressed:* {', '.join(sorted(list(relevant_conditions_matched)))}"
                 # Optionally list a few general effects
                 other_effects = sorted([e for e in info['effects'] if not any(c.lower() in e for c in conditions)])[:2]
                 if other_effects: line += f"\n  *Other Effects:* {', '.join(other_effects)}"
                 sections.append(line)

        if not found_plants:
             return f"I couldn't find specific plants listed in the database that demonstrably address **{', '.join(conditions)}** simultaneously based on the available effect data."

        sections.append("\n\n**Note:** Using multiple herbs requires careful consideration. Consult a professional.")
        # Safety note added globally
        return "\n".join(sections)

    def format_plant_compounds(self, plant_name: str, compounds_data: list, entities: dict) -> str:
        """
        Creates a response about compounds found in a specific plant, including their known effects. (Improved)
        """
        if not plant_name:
             return "Please specify the plant you want compound information for."
        if not compounds_data:
            return f"I couldn't find specific active compounds listed in the database for **{plant_name}**."

        sections = [f"Here are some key active compounds found in **{plant_name.title()}** and their generally associated effects:"]
        sections.append("\n**Compounds & Associated Effects:**")

        found_compounds = False
        for compound_info in compounds_data:
            cname = compound_info.get('compound_name')
            # Filter out invalid names
            if not cname or not str(cname).strip() or str(cname).strip().lower() in ['unknown compound', 'none', 'n/a']: continue
            cname = str(cname).strip()
            found_compounds = True

            effects = [e for e in compound_info.get('associated_effects', []) if e and str(e).strip()]
            line = f"â€¢ **{cname.title()}**"
            if effects:
                line += f"\n  *Associated Effects:* {self._format_list(effects[:3])}\n" # Limit effects shown
            else:
                 line += "\n  *Associated Effects:* Specific effects not detailed in database."
            sections.append(line)

        if not found_compounds:
             return f"I couldn't find specific active compounds listed in the database for **{plant_name}** after processing."
 
        #sections.append(f"**Note:** The overall effect of {plant_name.title()} comes from the complex interaction of many compounds.")
        # Safety note added globally
        return "\n".join(sections)

    def format_preparation_for_condition(self, condition: str, preparations_data: list, entities: dict) -> str:
        """
        Creates a response about preparation methods used for a specific condition, with example plants. (Improved)
        """
        # This formatter might be redundant if format_preparation_methods handles the 'is_condition_query' flag well.
        # We can delegate to the main formatter.
        logger.debug(f"Delegating format_preparation_for_condition for '{condition}' to format_preparation_methods.")
        return self.format_preparation_methods(target_entity=condition, prep_data=preparations_data, is_condition_query=True, entities=entities)


    def format_plant_preparation(self, plant_name: str, preparations_data: list, entities: dict) -> str:
        """
        Creates a response about preparation methods for a specific plant. (Improved)
        """
         # This formatter might be redundant if format_preparation_methods handles the 'is_condition_query' flag well.
        # We can delegate to the main formatter.
        logger.debug(f"Delegating format_plant_preparation for '{plant_name}' to format_preparation_methods.")
        return self.format_preparation_methods(target_entity=plant_name, prep_data=preparations_data, is_condition_query=False, entities=entities)

    def format_region_plants(self, region: str, plants_data: list, entities: dict = None) -> str:
        """
        Generates a concise response about plants found in a specific region.
        Randomly selects up to 10 plants, then lists them alphabetically.
        """
        if not region:
            return "Please specify the region you are interested in."
        if not plants_data:
            return (
                f"I couldn't find specific medicinal plants listed as primarily growing in the **{region}** region "
                "in the database. This doesn't mean the region lacks medicinal flora, only that it's not detailed in my current data."
            )

        # --- Collect and clean unique plants ---
        plant_info = {}
        for plant in plants_data:
            name = plant.get('plant_name') or plant.get('name')
            if not name or not str(name).strip():
                continue
            name = str(name).strip()
            name_lower = name.lower()
            if name_lower in plant_info:
                continue  # avoid duplicates

            sci_name = plant.get('scientific_name', '').strip()
            uses_data = plant.get('effects') or plant.get('uses') or plant.get('traditional_uses')

            # Normalize uses list
            if isinstance(uses_data, list):
                uses = [str(u).strip() for u in uses_data if u and str(u).strip()]
            elif isinstance(uses_data, str) and uses_data.strip():
                uses = [uses_data.strip()]
            else:
                uses = []

            plant_info[name_lower] = {
                'display_name': name,
                'scientific_name': sci_name,
                'uses': uses
            }

        if not plant_info:
            return (
                f"I couldn't find specific medicinal plants listed as primarily growing in the **{region}** region "
                "after processing."
            )

        # --- Randomly sample up to 10 plants ---
        all_plants = list(plant_info.values())
        sampled_plants = random.sample(all_plants, min(10, len(all_plants)))

        # --- Sort the selected plants alphabetically ---
        sampled_plants.sort(key=lambda p: p['display_name'].lower())

        # --- Build the response ---
        sections = [
            f"Here are some medicinal plants associated with the **{region.title()}** region based on available data:",
            "\n**Plants Found:**"
        ]

        for plant in sampled_plants:
            line = f"â€¢ **{plant['display_name']}**"
            if plant['scientific_name']:
                line += f" (*{plant['scientific_name']}*)"
            if plant['uses']:
                line += f"\n  *Common Uses/Effects:* {', '.join(plant['uses'][:3])}\n"
            sections.append(line)

        # --- Footer note ---
        sections.append(
            f"*Note: * Showing randomly selected plants out of {len(all_plants)} available in the database."
        )

        return "\n".join(sections)


    def _get_primary_entity(self, entities: Dict[str, List[str]], priority: List[str]) -> Optional[str]:
         """Helper to get the first entity found based on a priority list."""
         if not entities: return None # Handle case where entities dict might be None or empty
         for key in priority:
              entity_list = entities.get(key)
              if entity_list and isinstance(entity_list, list): # Check if it's a list
                   # Find the first valid string in the list
                   for entity in entity_list:
                        if entity and isinstance(entity, str) and str(entity).strip():
                             return str(entity).strip()
         return None # Return None if no valid entity found in priority list
