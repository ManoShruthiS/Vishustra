import logging
from typing import Any, Dict, List, Literal, Optional

# Assuming 'vishustra_core.nodes.base_node' is the correct module path for BaseNode
from vishustra_core.nodes.base_node import BaseNode 

logger = logging.getLogger(__name__)

class IntentClassifierNode(BaseNode):
    """
    A Vishustra processing node that classifies the intent of a given text input.
    
    This node simulates intent classification based on simple keyword matching.
    It takes an utterance and attempts to categorize it into one of the 
    predefined intents based on the presence of associated keywords.
    """

    def __init__(self, 
                 intent_keywords: Dict[str, List[str]], 
                 default_intent: str = "unrecognized_intent",
                 case_sensitive: bool = False):
        """
        Initializes the IntentClassifierNode with a mapping of intents to keywords.

        Args:
            intent_keywords: A dictionary where keys are intent names (str) and values
                             are lists of keywords (str) associated with that intent.
                             Example: {"greeting": ["hello", "hi"], "order_status": ["track", "status"]}
            default_intent: The intent to assign if no keywords for any defined intent are found.
                            Defaults to "unrecognized_intent".
            case_sensitive: If True, keyword matching will be case-sensitive. Defaults to False,
                            meaning both the utterance and keywords will be lowercased before matching.

        Raises:
            TypeError: If `intent_keywords`, `default_intent`, or `case_sensitive` are of incorrect types.
        """
        if not isinstance(intent_keywords, dict) or not all(
            isinstance(k, str) and isinstance(v, list) and all(isinstance(kw, str) for kw in v) 
            for k, v in intent_keywords.items()
        ):
            raise TypeError("`intent_keywords` must be a dictionary mapping intent names (str) to lists of keywords (str).")
        if not isinstance(default_intent, str):
            raise TypeError("`default_intent` must be a string.")
        if not isinstance(case_sensitive, bool):
            raise TypeError("`case_sensitive` must be a boolean.")

        self._intent_keywords = {
            intent: [kw.lower() if not case_sensitive else kw for kw in keywords]
            for intent, keywords in intent_keywords.items()
        }
        self._default_intent = default_intent
        self._case_sensitive = case_sensitive
        logger.debug(f"IntentClassifierNode initialized with intents: {list(self._intent_keywords.keys())}, "
                     f"default: '{self._default_intent}', case-sensitive: {self._case_sensitive}")


    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "IntentClassifierNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies the intent of the input text data based on configured keywords.

        Expected `data` input: A string representing the user utterance.

        Args:
            data: The input data, expected to be a string utterance.
            context: A dictionary containing contextual information for processing.
                     (This implementation passes context through but does not
                     directly use it for classification logic.)

        Returns:
            A dictionary containing:
            - 'original_data': The original input data.
            - 'classified_intent': The determined intent as a string.
            - 'confidence_score': A simulated confidence score (1.0 if matched, 0.0 if default).
            - 'matched_keywords': A list of unique keywords that contributed to the classification.
            - 'context': The original context dictionary, passed through.

        Raises:
            ValueError: If the input `data` is not a string.
        """
        if not isinstance(data, str):
            logger.error(f"IntentClassifierNode received non-string data: {type(data)}. Expected a string utterance.")
            raise ValueError(f"Input data must be a string. Received type: {type(data)}")

        utterance = data
        processed_utterance = utterance if self._case_sensitive else utterance.lower()
        
        classified_intent: str = self._default_intent
        confidence_score: float = 0.0
        matched_keywords: List[str] = []
        
        # Iterate through intents to find the first match based on keywords
        for intent, keywords in self._intent_keywords.items():
            current_intent_matched_kws = []
            for keyword in keywords:
                if keyword in processed_utterance:
                    current_intent_matched_kws.append(keyword)
            
            if current_intent_matched_kws:
                classified_intent = intent
                confidence_score = 1.0 # Simulate a perfect match
                matched_keywords.extend(current_intent_matched_kws)
                logger.debug(f"Classified '{utterance}' as '{intent}' based on keywords: {', '.join(current_intent_matched_kws)}")
                # For this simple model, we assume the first intent with any match is the winner
                break 

        if classified_intent == self._default_intent:
            logger.info(f"No specific intent keywords found for utterance: '{utterance}'. Assigned default intent: '{self._default_intent}'.")
        else:
            # Ensure unique keywords and maintain original order if desired, or just use a set.
            # For simplicity, we'll use a set to remove duplicates if multiple keywords for the *same* intent matched.
            # If multiple intents had keywords that matched (which wouldn't happen with the `break` above unless no break),
            # the `matched_keywords` would accumulate from the first winning intent.
            matched_keywords = list(set(matched_keywords)) 
            logger.info(f"Final classification for '{utterance}': '{classified_intent}' with matched keywords: {', '.join(matched_keywords)}")

        return {
            "original_data": utterance,
            "classified_intent": classified_intent,
            "confidence_score": confidence_score,
            "matched_keywords": matched_keywords,
            "context": context
        }
