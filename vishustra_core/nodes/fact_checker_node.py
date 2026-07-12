import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra processing node designed to simulate fact-checking of statements.

    This node takes an input dictionary expected to contain a 'statement' key
    and attempts to verify its factual accuracy against a simplistic,
    internal knowledge base. The output is the original data augmented with
    fact-checking results, including 'is_factual', 'confidence', and
    'evidence_summary'.

    In a production environment, this node would typically integrate with
    external fact-checking APIs, knowledge graphs, or advanced LLM reasoning
    capabilities.
    """

    # For demonstration purposes, a very simplistic hardcoded knowledge base.
    # This simulates how a real node might consult a source of truth.
    _KNOWN_FACTS = {
        "The Earth is round": {"is_factual": True, "confidence": 0.95, "evidence_summary": "Confirmed by satellite imagery and scientific consensus."},
        "Water boils at 100 degrees Celsius at sea level": {"is_factual": True, "confidence": 0.99, "evidence_summary": "Standard scientific measurement."},
        "The sun is a star": {"is_factual": True, "confidence": 0.98, "evidence_summary": "Astronomical classification based on nuclear fusion."},
    }

    _KNOWN_FALLACIES = {
        "The Earth is flat": {"is_factual": False, "confidence": 0.99, "evidence_summary": "Refuted by extensive scientific evidence and observation."},
        "Humans have only 5 senses": {"is_factual": False, "confidence": 0.85, "evidence_summary": "Humans possess more than 5 senses (e.g., proprioception, thermoception)."},
    }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to perform a simulated fact-check on a statement.

        The method expects `data` to be a dictionary containing a 'statement' key
        with the text to be fact-checked. It augments the input dictionary
        with fact-checking results.

        Args:
            data: A dictionary, typically containing a 'statement' key (str)
                  representing the text to verify.
            context: A dictionary for passing contextual information between nodes.
                     Not directly used in this simplified simulation but available
                     for extensibility (e.g., passing API keys or service clients).

        Returns:
            A dictionary identical to the input `data` but augmented with
            'is_factual' (bool | None), 'confidence' (float), and
            'evidence_summary' (str) keys.

        Raises:
            TypeError: If the input `data` is not a dictionary.
            ValueError: If the 'statement' key is missing or not a non-empty string
                        within the input `data` dictionary.
        """
        log_prefix = f"[{self.node_name}]"
        logger.debug(f"{log_prefix} Starting process for data type: {type(data)}")

        if not isinstance(data, dict):
            logger.error(f"{log_prefix} Invalid input data type. Expected a dictionary, received {type(data)}.")
            raise TypeError(f"{log_prefix} Input 'data' must be a dictionary, but got {type(data)}.")

        statement = data.get('statement')
        if not isinstance(statement, str) or not statement.strip():
            logger.error(f"{log_prefix} Missing or invalid 'statement' key in input data. Expected a non-empty string.")
            raise ValueError(f"{log_prefix} Input 'data' must contain a non-empty 'statement' string key.")

        processed_data = data.copy() # Create a mutable copy to augment results
        statement_normalized = statement.strip()
        statement_lower = statement_normalized.lower()

        # Initialize default fact-checking results for no match
        processed_data['is_factual'] = None
        processed_data['confidence'] = 0.0
        processed_data['evidence_summary'] = "No definitive fact-check result from this node's knowledge base."

        # Simulate checking against known facts
        for fact_phrase, result in self._KNOWN_FACTS.items():
            if fact_phrase.lower() in statement_lower:
                processed_data.update(result)
                logger.info(
                    f"{log_prefix} Statement '{statement_normalized[:100]}...' "
                    f"identified as FACTUAL with confidence {result['confidence']:.2f} based on '{fact_phrase}'."
                )
                return processed_data

        # Simulate checking against known fallacies
        for fallacy_phrase, result in self._KNOWN_FALLACIES.items():
            if fallacy_phrase.lower() in statement_lower:
                processed_data.update(result)
                logger.info(
                    f"{log_prefix} Statement '{statement_normalized[:100]}...' "
                    f"identified as FALLACIOUS with confidence {result['confidence']:.2f} based on '{fallacy_phrase}'."
                )
                return processed_data

        # If no specific match found
        logger.warning(
            f"{log_prefix} Statement '{statement_normalized[:100]}...' "
            "could not be definitively fact-checked against the current knowledge base. "
            "Returning as unverified."
        )
        return processed_data