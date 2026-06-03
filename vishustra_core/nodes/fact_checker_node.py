import logging
from typing import Any, Dict, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of statements.

    This node takes a statement (string or within a dictionary) and attempts
    to verify its truthfulness against a simulated internal knowledge base.
    In a production environment, this would integrate with sophisticated
    external fact-checking services, NLP models, or curated databases.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to perform a simulated fact-check.

        The `data` input is expected to be either:
        1. A string representing the statement to be fact-checked.
        2. A dictionary containing a 'statement' key whose value is the string to be fact-checked.

        The `context` dictionary can be used to pass configuration, API keys,
        or references to external services in a more advanced implementation.
        For this simulation, it is noted but not actively used.

        Args:
            data: The input statement (str) or a dict with a 'statement' key.
            context: A dictionary for additional runtime information or configuration.

        Returns:
            A dictionary containing the fact-checking result:
            - 'original_statement': The statement that was processed.
            - 'is_verified': A boolean (True/False) or None if verification is uncertain.
            - 'confidence': A float score (0.0 to 1.0) indicating confidence in the result.
            - 'reason': A brief explanation for the verification status.
            - 'sources': A list of simulated sources for the verification.

        Raises:
            ValueError: If the input data is not in an expected string or dictionary format.
            RuntimeError: For unexpected internal errors during the fact-checking simulation.
        """
        statement_to_check: str

        if isinstance(data, str):
            statement_to_check = data
        elif isinstance(data, dict) and 'statement' in data and isinstance(data['statement'], str):
            statement_to_check = data['statement']
        else:
            logger.error(f"FactCheckerNode received invalid data format: {type(data)}. Expected string or dict with 'statement' key.")
            raise ValueError(
                "FactCheckerNode: Input data must be a string or a dictionary "
                "containing a 'statement' string key."
            )

        # --- Simulated Internal Fact Database ---
        # This is a simplified lookup for demonstration purposes.
        # A real-world FactCheckerNode would integrate with external APIs,
        # knowledge graphs, or NLP models for comprehensive verification.
        known_facts: Dict[str, Dict[str, Any]] = {
            "the sky is blue": {
                "is_verified": True, "confidence": 0.95,
                "reason": "Common scientific observation based on Rayleigh scattering.",
                "sources": ["Atmospheric Physics Textbooks"]
            },
            "water boils at 100 degrees celsius": {
                "is_verified": True, "confidence": 0.98,
                "reason": "Fundamental physical property at standard atmospheric pressure.",
                "sources": ["Thermodynamics Principles"]
            },
            "the earth is flat": {
                "is_verified": False, "confidence": 0.99,
                "reason": "Overwhelming scientific and empirical evidence proves Earth is an oblate spheroid.",
                "sources": ["NASA", "Geodetic Surveys", "Astronomy"]
            },
            "2 + 2 = 5": {
                "is_verified": False, "confidence": 0.99,
                "reason": "Basic mathematical error; the sum is 4.",
                "sources": ["Elementary Mathematics"]
            },
            "birds are not real": {
                "is_verified": False, "confidence": 0.85,
                "reason": "This is a popular internet meme, not a verifiable fact. Birds are biological organisms.",
                "sources": ["Ornithology", "Biology"]
            },
        }

        # Normalize statement for lookup
        normalized_statement = statement_to_check.lower().strip()

        # Default result for statements not found in the simulated database
        result: Dict[str, Any] = {
            "original_statement": statement_to_check,
            "is_verified": None,  # Can be True, False, or None (for uncertain)
            "confidence": 0.5,
            "reason": "Could not conclusively verify or refute based on available (simulated) data.",
            "sources": [],
        }

        try:
            if normalized_statement in known_facts:
                fact_info = known_facts[normalized_statement]
                result.update(fact_info)
                logger.info(
                    f"Statement '{statement_to_check}' fact-checked: "
                    f"Verified={result['is_verified']}, Confidence={result['confidence']:.2f}"
                )
            else:
                # Basic keyword matching for more dynamic, albeit simple, responses
                if "climate change is a hoax" in normalized_statement:
                    result.update({
                        "is_verified": False, "confidence": 0.90,
                        "reason": "Scientific consensus overwhelmingly confirms human-caused climate change.",
                        "sources": ["IPCC Reports", "NASA"]
                    })
                elif "aliens built the pyramids" in normalized_statement:
                    result.update({
                        "is_verified": False, "confidence": 0.80,
                        "reason": "Archaeological and historical evidence attributes pyramid construction to ancient Egyptians.",
                        "sources": ["Egyptology", "Historical Records"]
                    })
                logger.info(
                    f"Statement '{statement_to_check}' fact-checked: Status={result['is_verified']}. "
                    "No direct match in simulated database, attempted keyword inference."
                )

        except Exception as e:
            logger.exception(f"An unexpected error occurred during fact-checking simulation for statement: '{statement_to_check}'")
            raise RuntimeError(f"FactCheckerNode internal processing error: {e}") from e

        return result
