import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A Vishustra node that simulates fact-checking textual statements.

    This node processes a string or a list of strings (statements) and
    provides a simulated verification status and confidence score for each.
    It's designed to illustrate the structure of a fact-checking component
    within the orchestration framework by applying a set of predefined rules
    and a simple internal "knowledge base".
    """

    def __init__(self):
        """
        Initializes the FactCheckerNode with a simulated knowledge base
        and linguistic pattern detectors.
        """
        logger.debug(f"Initializing {self.node_name}.")
        # In a production environment, this would involve loading ML models,
        # connecting to external fact-checking APIs, or initializing a knowledge graph client.
        self._knowledge_base = {
            "the sky is blue": "VERIFIED",
            "water boils at 100 degrees celsius": "VERIFIED",
            "cats are dogs": "UNVERIFIED",
            "birds can fly": "VERIFIED",
            "humans have three eyes": "UNVERIFIED",
            "the sun rises in the west": "UNVERIFIED",
            "all cars are red": "UNVERIFIED",
            "python is a programming language": "VERIFIED",
            "vishustra is an llm orchestration framework": "VERIFIED",
        }
        self._uncertain_keywords = ["some", "many", "often", "can be", "might", "potentially"]
        self._overstated_keywords = ["always", "never", "all", "every", "only", "must"]

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def _simulate_check(self, statement: str) -> Dict[str, Any]:
        """
        Simulates the fact-checking process for a single statement.
        This method applies simple rule-based logic to determine status and confidence.

        Args:
            statement (str): The individual text statement to check.

        Returns:
            Dict[str, Any]: A dictionary containing the original statement,
                            its simulated verification status, confidence score,
                            and a timestamp.
        """
        statement_lower = statement.lower().strip()
        status = "UNVERIFIED"
        confidence = round(random.uniform(0.1, 0.4), 2)  # Default low confidence for unknown

        # 1. Check against the simulated internal knowledge base
        if statement_lower in self._knowledge_base:
            status = self._knowledge_base[statement_lower]
            confidence = round(random.uniform(0.85, 0.99), 2) if status == "VERIFIED" else round(random.uniform(0.8, 0.95), 2)
            logger.debug(f"[{self.node_name}] KB match found for '{statement_lower}'. Status: {status}")

        # 2. Apply linguistic nuance detection (overstated claims)
        for keyword in self._overstated_keywords:
            if keyword in statement_lower:
                if status == "VERIFIED":
                    # A verified fact with overstated language implies nuance is required
                    status = "REQUIRES_NUANCE"
                    confidence = round(random.uniform(0.6, 0.8), 2)
                    logger.debug(f"[{self.node_name}] Overstated keyword '{keyword}' found, status adjusted to REQUIRES_NUANCE.")
                elif status == "UNVERIFIED":
                    # An unverified fact with overstated language is likely an overstated claim
                    status = "OVERSTATED_CLAIM"
                    confidence = round(random.uniform(0.7, 0.9), 2)
                    logger.debug(f"[{self.node_name}] Overstated keyword '{keyword}' found, status adjusted to OVERSTATED_CLAIM.")
                break # Apply the first matching keyword and exit loop

        # 3. Apply linguistic nuance detection (uncertain claims)
        # Only apply if not already marked as REQUIRES_NUANCE or OVERSTATED_CLAIM
        if status not in ["REQUIRES_NUANCE", "OVERSTATED_CLAIM"]:
            for keyword in self._uncertain_keywords:
                if keyword in statement_lower:
                    status = "PARTIALLY_VERIFIED"
                    confidence = round(random.uniform(0.4, 0.7), 2)
                    logger.debug(f"[{self.node_name}] Uncertain keyword '{keyword}' found, status adjusted to PARTIALLY_VERIFIED.")
                    break

        # 4. Handle very short statements or lack of information
        if len(statement_lower.split()) < 3 and status == "UNVERIFIED":
            status = "NOT_ENOUGH_INFO"
            confidence = round(random.uniform(0.1, 0.3), 2)
            logger.debug(f"[{self.node_name}] Statement too short, status adjusted to NOT_ENOUGH_INFO.")

        # 5. Introduce some general randomness for less clear-cut cases
        if status == "UNVERIFIED":
            if random.random() < 0.15: # 15% chance to be partially verified for generic unverified
                status = "PARTIALLY_VERIFIED"
                confidence = round(random.uniform(0.3, 0.5), 2)
            elif random.random() < 0.05: # 5% chance of a false positive verification
                status = "VERIFIED"
                confidence = round(random.uniform(0.5, 0.7), 2)

        return {
            "original_statement": statement,
            "verification_status": status,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }

    def process(self, data: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes the input data by performing simulated fact-checking on each statement.

        Args:
            data (Union[str, List[str]]): The text or a list of statements to fact-check.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the node's operation. While not directly
                                       used in this simulation, it's available for
                                       configuration like API keys, knowledge base URLs, etc.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a checked statement with its status,
                                  confidence, and original statement.

        Raises:
            ValueError: If the input data is not a string or a list of strings,
                        or if any item in an input list is not a string.
            Exception: For any unexpected errors during the simulated processing.
        """
        logger.info(f"[{self.node_name}] Starting fact-checking process for incoming data.")
        results: List[Dict[str, Any]] = []
        statements_to_check: List[str] = []

        # Acknowledge context, even if not used, to demonstrate awareness
        if context:
            logger.debug(f"[{self.node_name}] Context received: {list(context.keys())}. (Not directly used in this simulation).")

        try:
            if isinstance(data, str):
                statements_to_check.append(data)
                logger.debug(f"[{self.node_name}] Received a single statement for checking.")
            elif isinstance(data, list):
                if not all(isinstance(item, str) for item in data):
                    raise ValueError("All items in the input list must be strings.")
                statements_to_check.extend(data)
                logger.debug(f"[{self.node_name}] Received {len(data)} statements for checking.")
            else:
                raise ValueError("Input data must be a string or a list of strings for fact-checking.")

            if not statements_to_check:
                logger.warning(f"[{self.node_name}] No statements were provided for fact-checking. Returning empty results.")
                return []

            for statement in statements_to_check:
                logger.debug(f"[{self.node_name}] Attempting to simulate check for statement: '{statement}'")
                check_result = self._simulate_check(statement)
                results.append(check_result)
                logger.debug(f"[{self.node_name}] Fact-check result for '{statement}': Status={check_result['verification_status']}, Confidence={check_result['confidence']:.2f}")

        except ValueError as ve:
            logger.error(f"[{self.node_name}] Input validation error: {ve}", exc_info=True)
            raise # Re-raise to propagate the error up the orchestration chain
        except Exception as e:
            logger.critical(f"[{self.node_name}] An unexpected critical error occurred during processing: {e}", exc_info=True)
            raise # Re-raise to ensure framework handles unexpected failures

        logger.info(f"[{self.node_name}] Fact-checking process completed. {len(results)} statements processed.")
        return results