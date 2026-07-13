import logging
import re
from typing import Any, Dict

# This import assumes vishustra_core is installed and available in the Python path.
# The BaseNode definition is provided in the project context.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node designed to classify the intent of a given text input.

    This node simulates intent classification based on a predefined set of keywords
    and regular expression patterns. It is intended to process user queries
    and identify their primary intention (e.g., greeting, order status, product inquiry).

    The node expects a string as input data and returns a structured dictionary
    containing the classified intent, a confidence score, and the original query.
    """

    def __init__(self):
        """
        Initializes the IntentClassifierNode with a predefined intent mapping.
        Each intent is associated with keywords and optional regex patterns,
        along with a default confidence level for that intent.
        In a production environment, this mapping would typically be loaded
        from a configuration file, a dedicated service, or a machine learning model.
        """
        self._intent_map = {
            "greeting": {
                "keywords": ["hello", "hi", "hey", "good morning", "good evening"],
                "patterns": [r"^(hi|hello|hey)\b", r"\bgood (morning|evening)\b"],
                "confidence": 0.95
            },
            "farewell": {
                "keywords": ["bye", "goodbye", "see you", "farewell"],
                "patterns": [r"^(bye|goodbye)\b", r"\bsee you (later)?\b"],
                "confidence": 0.90
            },
            "order_status": {
                "keywords": ["order status", "my order", "where is", "track my"],
                "patterns": [r"^(where is|what is the status of) my order", r"\btrack my order\b"],
                "confidence": 0.98
            },
            "account_info": {
                "keywords": ["my account", "account details", "change password", "update profile"],
                "patterns": [r"\bmy account\b", r"\baccount details\b", r"\bchange my password\b"],
                "confidence": 0.85
            },
            "product_inquiry": {
                "keywords": ["product", "item", "about", "information on", "tell me about"],
                "patterns": [r"\btell me about (.+)\b", r"\binformation on (.+)\b", r"\bproduct (.+)\b"],
                "confidence": 0.80
            },
            "unclassified": {
                "keywords": [],
                "patterns": [],
                "confidence": 0.10 # Default low confidence for unclassified queries
            }
        }
        logger.debug("IntentClassifierNode initialized with predefined intent map.")

    @property
    def node_name(self) -> str:
        """
        Returns the programmatic name of this node, 'IntentClassifier'.
        This name can be used for identification within the Vishustra framework.
        """
        return "IntentClassifier"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data (a user query string) to classify its intent.

        The classification is performed by matching the query against a set of
        predefined keywords and regular expression patterns associated with
        various intents. Pattern matches are generally given higher priority
        than keyword matches due to their specificity and typically higher
        configured confidence.

        Args:
            data (Any): The input data, which is expected to be a string
                        representing the user's query.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current processing flow.
                                       This node does not currently utilize the context,
                                       but it is passed through as part of the Vishustra
                                       node interface.

        Returns:
            Dict[str, Any]: A dictionary containing the classification result:
                            - "intent": The identified intent (e.g., "greeting", "order_status").
                            - "confidence": A float representing the confidence score (0.0 to 1.0).
                            - "original_query": The original input query string.

        Raises:
            ValueError: If the input `data` is not a string, indicating an invalid
                        input type for intent classification.
        """
        if not isinstance(data, str):
            error_msg = f"IntentClassifierNode received invalid data type. Expected string, got {type(data).__name__}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        query = data.lower().strip()
        
        best_intent = "unclassified"
        best_confidence = self._intent_map["unclassified"]["confidence"]
        
        # Store potential matches with their confidence and source (pattern/keyword)
        # Patterns are generally more specific, so they can influence confidence weighting.
        potential_matches: Dict[str, Dict[str, Any]] = {}

        # Evaluate patterns first, as they often indicate stronger intent
        for intent_name, config in self._intent_map.items():
            for pattern_str in config.get("patterns", []):
                if re.search(pattern_str, query):
                    # If this intent hasn't been matched yet, or if this pattern match
                    # offers higher confidence than a previous match for the same intent
                    current_match = potential_matches.get(intent_name)
                    if not current_match or config["confidence"] > current_match["confidence"]:
                        potential_matches[intent_name] = {"confidence": config["confidence"], "source": "pattern"}
                        logger.debug(f"Pattern match for intent '{intent_name}': '{pattern_str}' in '{query}'")

        # Then, evaluate keywords. A keyword match for an intent should only override
        # an existing match for that same intent if it provides strictly higher confidence,
        # or if the existing match was also a keyword match with lower confidence.
        # It should not easily override a pattern match for the same intent unless specifically configured.
        for intent_name, config in self._intent_map.items():
            for keyword in config.get("keywords", []):
                if keyword in query: # Simple substring match for keywords
                    current_match = potential_matches.get(intent_name)
                    
                    if not current_match: # No previous match for this intent
                        potential_matches[intent_name] = {"confidence": config["confidence"], "source": "keyword"}
                        logger.debug(f"Keyword match for intent '{intent_name}': '{keyword}' in '{query}' (initial)")
                    elif current_match["source"] == "keyword" and config["confidence"] > current_match["confidence"]:
                        # Existing match was also keyword, and this one is stronger
                        potential_matches[intent_name] = {"confidence": config["confidence"], "source": "keyword"}
                        logger.debug(f"Keyword match for intent '{intent_name}': '{keyword}' in '{query}' (overriding existing keyword)")
                    elif current_match["source"] == "pattern" and config["confidence"] > current_match["confidence"]:
                        # Existing match was a pattern, but this keyword is configured with higher confidence
                        potential_matches[intent_name] = {"confidence": config["confidence"], "source": "keyword"}
                        logger.debug(f"Keyword match for intent '{intent_name}': '{keyword}' in '{query}' (overriding existing pattern due to higher confidence)")

        # Determine the best intent from all potential matches
        if potential_matches:
            # Sort matches:
            # Primary sort key: confidence (descending).
            # Secondary sort key: source type (pattern preferred over keyword if confidences are equal).
            # Assigning numerical values (1 for pattern, 0 for keyword) allows for simple sorting.
            sorted_matches = sorted(
                potential_matches.items(),
                key=lambda item: (item[1]["confidence"], 1 if item[1]["source"] == "pattern" else 0),
                reverse=True
            )
            best_intent, match_info = sorted_matches[0]
            best_confidence = match_info["confidence"]
        
        logger.info(f"Classified intent for query '{data}': '{best_intent}' with confidence {best_confidence:.2f}")

        return {
            "intent": best_intent,
            "confidence": best_confidence,
            "original_query": data
        }
