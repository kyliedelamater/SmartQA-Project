import torch
import re
import logging
import os
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

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

        except ImportError:
            logger.error("Required libraries not found. Install: pip install transformers torch sentencepiece")
            # Raise the error to prevent the system from continuing in a broken state
            raise ImportError("transformers library not found. Please install it.")
        except Exception as e:
            logger.error(f"Error loading Flan-T5 model '{model_name}': {e}", exc_info=True)
            # Raise the error
            raise RuntimeError(f"Failed to load Flan-T5 model: {e}")

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
        response = re.sub(r'\s*•\s*', r'\n• ', response)
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
        """Add appropriate cautions to the response if not already present."""
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
            caution += "Always consult with a qualified healthcare professional before using any medicinal plant, "
            caution += "especially if you have underlying health conditions, are pregnant or nursing, are taking other medications, or considering combining treatments. "
            caution += "Individual responses can vary, and proper identification, dosage, and preparation are crucial for safety and effectiveness."

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
    def _format_list(self, items: List[str]) -> str:
        """Format a list into a human-readable string with commas and 'and'."""
        cleaned_items = [item.strip() for item in items if item and item.strip()]

        if not cleaned_items:
            return ""
        if len(cleaned_items) == 1:
            return cleaned_items[0]
        if len(cleaned_items) == 2:
            return f"{cleaned_items[0]} and {cleaned_items[1]}"

        return ", ".join(cleaned_items[:-1]) + ", and " + cleaned_items[-1]

    #kylie added
    def _natural_join(self, items: List[str]) -> str:
         if not items:
             return ""
         if len(items) == 1:
             return items[0]
         return ", ".join(items[:-1]) + " and " + items[-1]

    # --- Specific Formatters ---

    def format_plant_info(self, plant_data: dict, entities: dict) -> str:
        """
        Creates a comprehensive response about a medicinal plant. (Improved Handling)
        Includes all available information in a structured format.
        """
        if not plant_data or not plant_data.get('name'):
            queried_plant = self._get_primary_entity(entities, ['plants'])
            return self._format_no_data_response("plant", queried_plant)

        # Extract all data fields safely
        name = plant_data.get('name', 'Unknown Plant')
        sci_name = plant_data.get('scientific_name')
        family = plant_data.get('family')
        morphology = plant_data.get('morphology')
        distribution = plant_data.get('distribution')
        # Clean lists immediately
        effects = [e for e in plant_data.get('effects', []) if e and str(e).strip()]
        compounds = [c for c in plant_data.get('compounds', []) if c and str(c).strip()]
        preparations = [p for p in plant_data.get('preparations', []) if p and str(p).strip()]
        side_effects = [s for s in plant_data.get('side_effects', []) if s and str(s).strip()]
        traditional_uses = [t for t in plant_data.get('traditional_uses', []) if t and str(t).strip()]
        contraindications = [c for c in plant_data.get('contraindications', []) if c and str(c).strip()] # Added

        sections = []

        # Introduction
        intro = f"**{name}**"
        if sci_name: intro += f" ({sci_name})"
        if family: intro += f" is a medicinal plant from the {family} family."
        else: intro += " is a notable medicinal plant."
        sections.append(intro)

        # Physical description
        desc_section = ""
        if morphology: desc_section += f"\n**Description:** {morphology}"
        if distribution: desc_section += f"\n**Distribution:** {distribution}"
        if desc_section: sections.append(desc_section.strip())

        # Medicinal properties
        if effects:
            sections.append(f"\n**Medicinal Properties & Effects:**\n{self._format_list(effects)}")

        # Traditional uses
        if traditional_uses:
            sections.append(f"\n**Traditional Uses:**\n{self._format_list(traditional_uses)}")

        # Active compounds
        if compounds:
            sections.append(f"\n**Key Active Compounds:**\n{self._format_list(compounds)}")

        # Preparation methods
        if preparations:
            sections.append(f"\n**Traditional Preparation Methods:**\n{self._format_list(preparations)}")

        # Safety information
        safety_section = "\n**Safety Considerations:**"
        if side_effects:
            safety_section += f"\n*Potential Side Effects:*\n{self._format_list(side_effects)}"
        else:
            safety_section += f"\n*Potential Side Effects:* Specific side effects are not listed in the database for {name}, but caution is always advised."

        if contraindications:
             safety_section += f"\n*Contraindications (Should Avoid):*\n{self._format_list(contraindications)}"

        safety_section += "\n*General Precautions:*"
        safety_section += "\n• Always start with a low dose to assess tolerance."
        safety_section += "\n• Ensure correct plant identification and use high-quality sources."
        safety_section += "\n• Be aware of potential interactions with medications (consult your doctor)."
        safety_section += "\n• Use with caution if pregnant, nursing, or have pre-existing health conditions."
        sections.append(safety_section)

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
        intro = f"Here's information on medicinal plants traditionally associated with **{condition}**:"
        sections.append(intro)

        # Primary plants section
        if primary_plants:
            sections.append("\n**Plants Primarily Used:**")
            # Sort alphabetically by name
            for name in sorted(primary_plants.keys()):
                plant = primary_plants[name]
                plant_text = f"\n• **{plant['name']}**"
                if plant['scientific_name']: plant_text += f" ({plant['scientific_name']})"
                if plant['effects_sample']: plant_text += f"\n  Known effects include: {', '.join(plant['effects_sample'])}"
                sections.append(plant_text)
        else:
             sections.append(f"\nNo plants specifically listed for directly treating '{condition}' were found in the database based on the available effect descriptions. However, some plants may offer supportive benefits.")

        # Supportive plants section
        if supportive_plants:
            sections.append("\n**Other Potentially Supportive Plants:**")
             # Sort alphabetically by name
            for name in sorted(supportive_plants.keys()):
                plant = supportive_plants[name]
                plant_text = f"\n• **{plant['name']}**"
                if plant['scientific_name']: plant_text += f" ({plant['scientific_name']})"
                if plant['effects_sample']: plant_text += f"\n  Related benefits may include: {', '.join(plant['effects_sample'])}"
                sections.append(plant_text)

        # Usage guidelines (General)
        guidelines = "\n**General Usage Considerations:**"
        guidelines += "\n• Effectiveness can vary based on the plant part used, preparation, dosage, and individual factors."
        guidelines += "\n• Start with a single plant and a low dose to assess your response."
        guidelines += "\n• Follow traditional or expert-recommended preparation methods."
        guidelines += "\n• Ensure proper identification and use high-quality plant sources."
        sections.append(guidelines)

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

        intro = f"Here are some medicinal plants found in **{region}** traditionally associated with **{condition}**:"
        sections.append(intro)

        # List relevant plants
        sections.append("\n**Relevant Plants:**")
        for name in sorted(relevant_plants.keys()):
            info = relevant_plants[name]
            line = f"\n• **{name}**"
            if info['scientific_name']: line += f" ({info['scientific_name']})"
            # Show effects relevant to the condition first, then others
            matched_effects = sorted([e for e in info['effects'] if any(cond_syn in e.lower() for cond_syn in synonyms)])
            other_effects = sorted([e for e in info['effects'] if e not in matched_effects])[:2] # Limit other effects shown

            if matched_effects: line += f"\n  *Relevant Effects:* {', '.join(matched_effects)}"
            if other_effects: line += f"\n  *Other Effects:* {', '.join(other_effects)}"
            if info['traditional_uses']: line += f"\n  *Traditional Uses:* {self._format_list(list(info['traditional_uses']), max_items=3)}"
            if info['preparations']: line += f"\n  *Preparation Methods:* {self._format_list(list(info['preparations']), max_items=3)}"
            sections.append(line)

        # Regional context (simplified)
        context = f"\n**Context for {region}:**"
        context += "\n• Traditional knowledge in this region often guides the specific use and preparation of these plants."
        context += "\n• Local environmental factors can influence the potency and characteristics of plants."
        context += "\n• Sustainable harvesting is important to preserve these resources."
        sections.append(context)

        # Considerations (General)
        considerations = "\n**Important Considerations:**"
        considerations += "\n• This information is based on available data; local practices may vary."
        considerations += "\n• Always ensure correct plant identification."
        considerations += "\n• Preparation methods significantly impact effectiveness and safety."
        sections.append(considerations)

        # Safety note added globally

        return "\n".join(sections)

    def format_safety_info(self, plant_name: str, safety_data: dict, context: dict, entities: dict) -> str:
        """
        Creates a comprehensive safety response about a plant. (Improved Fallback & Clarity)
        Focuses on side effects, interactions, and safe usage guidelines.
        """
        if not plant_name:
             return "Please specify the plant you want safety information for."

        # Even if safety_data is empty, provide a response structure with general warnings.
        safety_data = safety_data or {} # Ensure safety_data is a dict

        # Extract safety information safely
        side_effects = [s for s in safety_data.get('side_effects', []) if s and str(s).strip()]
        contraindications = [c for c in safety_data.get('contraindications', []) if c and str(c).strip()]
        interactions = [i for i in safety_data.get('interactions', []) if i and str(i).strip()]
        # Context effects are just for framing, not safety per se
        effects_context = [e for e in safety_data.get('effects_context', []) if e and str(e).strip()]

        sections = []

        # Introduction
        intro = f"Here is safety information regarding **{plant_name}**."
        if effects_context:
            intro += f" While it's known for effects like {self._format_list(effects_context[:2])}, understanding potential risks is crucial."
        sections.append(intro)

        # Side effects
        sections.append("\n**Potential Side Effects:**")
        if side_effects:
            sections.append(self._format_list(side_effects))
        else:
            sections.append(f"Specific side effects for {plant_name} are not listed in the database. However, this does not mean it is without risk. Monitor for any unusual reactions.")

        # Contraindications
        sections.append("\n**Contraindications (Who Should Use With Caution or Avoid):**")
        if contraindications:
            sections.append(self._format_list(contraindications))
        else:
            sections.append(f"Specific contraindications for {plant_name} are not listed. General caution applies, especially for vulnerable groups (see below).")

        # Interactions
        sections.append("\n**Potential Interactions:**")
        if interactions:
            sections.append(self._format_list(interactions))
        else:
            sections.append(f"Specific drug or herb interactions for {plant_name} are not detailed in the database. **It is critical to assume interactions are possible, especially with prescription medications.** Discuss use with your doctor.")

        # Safe usage guidelines (Always include)
        guidelines = "\n**General Safe Usage Guidelines:**"
        guidelines += "\n• **Consult First:** Always talk to a qualified healthcare professional before using this plant, especially if combining with other treatments."
        guidelines += "\n• **Start Low:** Begin with the smallest effective dose to assess your individual reaction."
        guidelines += "\n• **Quality Source:** Use high-quality, correctly identified plant material from reputable sources."
        guidelines += "\n• **Proper Prep:** Follow recommended preparation methods carefully."
        guidelines += "\n• **Monitor:** Pay attention to how your body responds and stop use if adverse effects occur."
        guidelines += "\n• **Interactions:** Be aware of potential interactions with foods, supplements, or medications."
        sections.append(guidelines)

        # Special populations (Always include)
        special = "\n**Extra Caution Needed For:**"
        special += "\n• Pregnant or nursing individuals."
        special += "\n• Children and the elderly."
        special += "\n• People with chronic health conditions (e.g., liver, kidney disease)."
        special += "\n• Those taking prescription medications (especially blood thinners, antidepressants, immunosuppressants)."
        special += "\n• Individuals with known allergies or sensitivities."
        sections.append(special)

        # Professional guidance is integrated into the guidelines now.
        # Safety note added globally

        return "\n".join(sections)

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
        intro = f"**{compound_name_display}** is a bioactive compound found in various medicinal plants."
        if plants:
            intro += f" It is notably present in plants such as {', '.join(plants[:3])}"
            if len(plants) > 3: intro += f", among others."
            else: intro += "."
        sections.append(intro)

        # Therapeutic effects
        if effects:
            sections.append(f"\n**Known Therapeutic Properties & Effects:**\n{self._format_list(effects)}")
        else:
             sections.append(f"\n**Known Therapeutic Properties & Effects:** Specific effects associated directly with {compound_name_display} are not detailed in the database, but it contributes to the overall activity of the plants it's found in.")

        # Source plants
        if plants:
            sections.append(f"\n**Natural Plant Sources:**\n{self._format_list(plants)}")

        # Scientific context / How it works (General statements)
        science = f"\n**Understanding {compound_name_display}'s Role:**"
        science += f"\n• The effects of {compound_name_display} often result from synergy with other compounds in the plant."
        science += "\n• Its concentration and bioavailability can vary significantly based on the plant source, part used, and preparation method."
        science += "\n• Research is ongoing to fully understand its mechanisms of action."
        sections.append(science)

        # Research insights (If available)
        if research_notes:
            sections.append(f"\n**Research Insights:**\n{self._format_list(research_notes)}")

        # Interactions (If available)
        if interactions:
            sections.append(f"\n**Potential Interactions:**\n{self._format_list(interactions)}")

        # Practical considerations
        practical = "\n**Practical Considerations:**"
        practical += f"\n• The amount of {compound_name_display} can differ greatly between plant batches and preparations."
        practical += "\n• Extraction methods (e.g., water, alcohol, oil) influence which compounds are present in the final product."
        practical += "\n• Consider the plant source as a whole, not just the isolated compound, for traditional use."
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
            # Check if the compound itself was recognized but no plants found
            compound_synonyms = entities.get('_synonyms', {}).get(compound_name, [compound_name])
            # Check if the compound or its synonyms were in the original entity list
            if any(c in entities.get('compounds', []) for c in compound_synonyms):
                 return f"While **{compound_name}** is a known compound, I couldn't find specific plants listed as containing it in the database."
            else:
                 return self._format_no_data_response("compound", compound_name)


        # Organize plant data - deduplicate and aggregate effects
        plant_details = {}
        all_effects = set()

        for result in results:
            plant_name = result.get('plant_name')
            if not plant_name or not str(plant_name).strip(): continue # Skip if name is missing
            plant_name = str(plant_name).strip()

            if plant_name not in plant_details:
                plant_details[plant_name] = {
                    'scientific_name': result.get('scientific_name'),
                    'effects': set(),
                    'concentration': result.get('concentration'), # Assuming this might exist
                    'preparations': result.get('preparations', []) # Assuming this might exist
                }

            # Aggregate effects
            effects = result.get('effects', [])
            if isinstance(effects, list):
                current_effects = {str(e).strip() for e in effects if e and str(e).strip()}
                plant_details[plant_name]['effects'].update(current_effects)
                all_effects.update(current_effects)
            elif isinstance(effects, str) and effects.strip(): # Handle single effect string
                 plant_details[plant_name]['effects'].add(effects.strip())
                 all_effects.add(effects.strip())


        # Build response
        sections = []
        intro = f"Here are medicinal plants known to contain the compound **{compound_name}**:"
        sections.append(intro)

        # List all plants found
        sections.append(f"\n**Plants Containing {compound_name}:**")
        if not plant_details: # Should not happen if results existed, but check anyway
             return f"I couldn't find specific plants listed as containing **{compound_name}** in the database."

        for name in sorted(plant_details.keys()):
            details = plant_details[name]
            line = f"\n• **{name}**"
            if details['scientific_name']: line += f" ({details['scientific_name']})"
            # Show associated effects concisely
            if details['effects']: line += f"\n  *Associated Effects:* {', '.join(sorted(list(details['effects']))[:3])}" # Limit shown effects
            # Add concentration/prep info if available
            if details['concentration']: line += f"\n  *Concentration Info:* {details['concentration']}"
            if details['preparations'] and isinstance(details['preparations'], list) and any(p for p in details['preparations']):
                 line += f"\n  *Common Preparations:* {self._format_list(details['preparations'], max_items=3)}"
            sections.append(line)

        # Group by common therapeutic effects (optional, can be complex)
        # Let's skip grouping by effect for now to keep it simpler and less prone to errors.
        # If needed later, requires careful aggregation.

        # Practical information
        practical = f"\n**Key Points About {compound_name} in Plants:**"
        practical += f"\n• The concentration of {compound_name} varies significantly between plants, plant parts, and growing conditions."
        practical += "\n• Preparation methods (e.g., infusion, decoction, tincture) affect how much of the compound is extracted."
        practical += "\n• The overall effect of the plant comes from the synergy of all its compounds, not just one."
        sections.append(practical)

        # Usage guidelines
        guidelines = "\n**Usage Considerations:**"
        guidelines += "\n• Choose plants based on the full range of desired effects and traditional uses, not just the presence of one compound."
        guidelines += "\n• Follow recommended preparation methods for the specific plant being used."
        guidelines += "\n• Consider individual sensitivities and potential interactions."
        sections.append(guidelines)

        # Safety note added globally

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
        response += "\n• A specific plant's properties (e.g., 'What are the benefits of Ginger?')"
        response += "\n• Plants for a specific condition (e.g., 'Which herbs help with headaches?')"
        response += "\n• Safety information for a plant (e.g., 'Is St. John's Wort safe?')"
        response += "\n• Preparation methods (e.g., 'How to prepare Echinacea tea?')"
        response += "\n• Plants from a specific region (e.g., 'Medicinal plants from the Andes')"

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
            return f"I couldn't find specific plants listed as having significantly similar *therapeutic effects* to **{target_plant}** based on the data available. Plants can be similar in other ways (e.g., appearance, family) not covered here."

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
        intro = f"Here are some plants that share similar **therapeutic effects** with **{target_plant}**, based on available data:"
        sections.append(intro)

        # List similar plants, potentially sorted by similarity score
        sorted_plants = sorted(similar_plants_info.items(), key=lambda item: item[1]['similarity_score'], reverse=True)

        sections.append("\n**Similar Plants:**")
        for name, info in sorted_plants:
            line = f"\n• **{name}**"
            if info['scientific_name']: line += f" ({info['scientific_name']})"
            if info['shared_effects']:
                line += f"\n  *Shared Effects Include:* {', '.join(info['shared_effects'][:3])}" # Limit displayed effects
            # Could add similarity score if meaningful: line += f" (Similarity Score: {info['similarity_score']})"
            sections.append(line)

        # Important differences
        differences = "\n**Important Differences to Consider:**"
        differences += f"\n• While sharing some effects, these plants have unique chemical profiles and other distinct properties not shared with {target_plant}."
        differences += "\n• Potency, optimal preparation methods, and safety profiles (side effects, interactions) can differ significantly."
        differences += "\n• Traditional uses might vary even if some effects overlap."
        sections.append(differences)

        # Selection guidelines
        guidelines = f"\n**Guidelines for Choosing:**"
        guidelines += "\n• Base your choice on your specific health goals and the *full* profile of the plant, not just the overlap."
        guidelines += "\n• Consider availability, ease of preparation, and your individual sensitivities."
        guidelines += f"\n• Research each plant individually, including its specific safety information, before use."
        sections.append(guidelines)

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
            intro = f"Here are traditional preparation methods used for **{target_entity}**:"
        sections.append(intro)

        # List Methods
        sections.append("\n**Preparation Methods:**")
        if not methods: # Check if aggregation removed all methods
             return f"I couldn't find specific preparation methods listed in the database for **{target_entity}** after processing."

        for method_name in sorted(methods.keys()):
            details = methods[method_name]
            line = f"\n• **{method_name}**"
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
        quality += "\n• **Plant Material:** Use high-quality, correctly identified plants from a reliable source."
        quality += "\n• **Plant Part:** Different parts (root, leaf, flower) often require different methods."
        quality += "\n• **Solvent:** Water, alcohol, oil, or vinegar extract different compounds."
        quality += "\n• **Technique:** Proper temperature, timing, and equipment are crucial."
        quality += "\n• **Storage:** Store preparations correctly (e.g., cool, dark place) to maintain potency."
        sections.append(quality)

        # Method Selection (General advice)
        selection = "\n**Choosing the Right Method:**"
        selection += "\n• The best method depends on the specific plant, the desired effects, and the compounds being targeted."
        selection += "\n• Infusions (teas) are common for leaves and flowers; decoctions (simmering) for roots and barks; tinctures (alcohol extraction) for broader compound extraction."
        selection += "\n• Consult reliable herbal resources or practitioners for guidance specific to the plant or condition."
        sections.append(selection)

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
        intro = f"Here are plants potentially relevant to **{', '.join(conditions)}**:"
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


        found_plants = False
        for name in sorted(plant_info.keys()):
             info = plant_info[name]
             # Find which of the queried conditions are matched by the plant's effects
             relevant_conditions_matched = {
                 cond for cond in conditions
                 if any(cond.lower() in effect for effect in info['effects'])
             }

             if relevant_conditions_matched: # Only list plants that match at least one condition
                 found_plants = True
                 line = f"\n• **{name}**"
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

        sections = [f"Here are some key active compounds found in **{plant_name}** and their generally associated effects:"]
        sections.append("\n**Compounds & Associated Effects:**")

        found_compounds = False
        for compound_info in compounds_data:
            cname = compound_info.get('compound_name')
            # Filter out invalid names
            if not cname or not str(cname).strip() or str(cname).strip().lower() in ['unknown compound', 'none', 'n/a']: continue
            cname = str(cname).strip()
            found_compounds = True

            effects = [e for e in compound_info.get('associated_effects', []) if e and str(e).strip()]
            line = f"\n• **{cname}**"
            if effects:
                line += f"\n  *Associated Effects:* {self._format_list(effects[:5])}" # Limit effects shown
            else:
                 line += "\n  *Associated Effects:* Specific effects not detailed in database."
            sections.append(line)

        if not found_compounds:
             return f"I couldn't find specific active compounds listed in the database for **{plant_name}** after processing."

        sections.append(f"\n\n**Note:** The overall effect of {plant_name} comes from the complex interaction of many compounds.")
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
        Generates a response about plants found in a specific region. (Improved Formatting & Direct Formatting)
        """
        if not region:
            return "Please specify the region you are interested in."
        if not plants_data:
            return f"I couldn't find specific medicinal plants listed as primarily growing in the **{region}** region in the database. This doesn't mean the region lacks medicinal flora, only that it's not detailed in my current data."

        plant_lines = []
        mentioned = set()
        for plant in plants_data:
            # Prioritize specific keys if available
            name = plant.get('plant_name') or plant.get('name')
            if not name or not str(name).strip(): continue
            name = str(name).strip()

            # Avoid duplicates based on lowercase name
            name_lower = name.lower()
            if name_lower in mentioned:
                continue
            mentioned.add(name_lower)

            sci_name = plant.get('scientific_name', '')
            # Aggregate effects/uses from potential keys
            uses = []
            uses_data = plant.get('effects') or plant.get('uses') or plant.get('traditional_uses')
            if isinstance(uses_data, list):
                uses = [str(u).strip() for u in uses_data if u and str(u).strip()]
            elif isinstance(uses_data, str) and uses_data.strip():
                uses = [uses_data.strip()]


            line = f"• **{name}**"
            if sci_name and str(sci_name).strip():
                line += f" ({str(sci_name).strip()})"
            if uses:
                line += f"\n  *Common Uses/Effects:* {', '.join(uses[:3])}" # Limit uses shown
            plant_lines.append(line)

        if not plant_lines: # Check if filtering removed all plants
             return f"I couldn't find specific medicinal plants listed as primarily growing in the **{region}** region in the database after processing."

        # Format directly instead of using LLM prompt
        response_parts = []
        response_parts.append(f"Here are some medicinal plants associated with the **{region}** region based on available data:")
        response_parts.append("\n**Plants Found:**")
        response_parts.extend(plant_lines)
        response_parts.append(f"\n\n**Note:** This list reflects information in the database and may not be exhaustive for the entire {region}. Plant distribution can be complex.")

        # Safety note added globally by enhance_response_with_cautions
        return "\n".join(response_parts)

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
