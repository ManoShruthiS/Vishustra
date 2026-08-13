import logging
from typing import Any, Dict, List

# Assuming BaseNode is available at the specified path within the Vishustra framework.
from vishustra_core.nodes.base_node import BaseNode

# Configure logging for this module
logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking claims against a predefined knowledge base.
    It takes input data, identifies potential claims (from 'text' or 'claims' fields),
    and augments the data with verification results indicating truthfulness based on
    its internal knowledge base.
    """

    def __init__(self):
        """
        Initializes the FactCheckerNode.
        A static, simulated knowledge base is used for demonstration. In a production
        environment, this would typically interface with external APIs, databases,
        or more sophisticated NLP models for fact retrieval and verification.
        """
        # A simple, static knowledge base for simulation purposes.
        # Key: claim string, Value: boolean (True for fact, False for misinformation).
        self._knowledge_base: Dict[str, bool] = {
            "The capital of France is Paris.": True,
            "Elephants can fly.": False,
            "Water boils at 100 degrees Celsius at sea level.": True,
            "The sun revolves around the Earth.": False,
            "Humans cannot breathe underwater.": True,
            "The Earth is flat.": False,
            "Birds are mammals.": False,
            "The Great Wall of China is visible from space with the naked eye.": False,
            "Spinach is high in iron.": True,
            "Mount Everest is the highest mountain in the world.": True,
            "The moon is made of cheese.": False,
        }
        logger.debug(f"{self.node_name} initialized with a simulated knowledge base.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, attempting to identify and verify claims.
        The input `data` is expected to be a dictionary, potentially containing:
        - A "text" key (str): The entire text is treated as a single claim to verify.
        - A "claims" key (List[str]): A list of explicit claims to verify.

        The method augments the input `data` with a "fact_check_results" key,
        which is a list of dictionaries, each containing:
        - "claim" (str): The original claim string.
        - "status" (bool | None): True if verified as true, False if refuted as false,
                                  None if verification is inconclusive (not in KB).
        - "reason" (str): A descriptive reason for the status.

        Args:
            data (Any): The input data to process, typically a dictionary.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the current processing flow. Not directly
                                       used for fact-checking logic in this simulation,
                                       but available for logging.

        Returns:
            Any: The original `data` dictionary, augmented with "fact_check_results".
                 Returns original `data` unchanged if it's not a dictionary or on critical errors.
        """
        logger.info(f"[{self.node_name}] Starting fact-checking process for incoming data.")

        # --- Input Validation and Pre-processing ---
        if not isinstance(data, dict):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected dict, got {type(data)}. "
                         "Returning original data without processing.")
            return data

        claims_to_check: List[str] = []

        if "claims" in data and isinstance(data["claims"], list):
            # Extract claims, ensuring they are strings.
            claims_to_check.extend([str(c) for c in data["claims"] if isinstance(c, str)])
            logger.debug(f"[{self.node_name}] Found {len(claims_to_check)} claims in 'claims' field.")
        elif "text" in data and isinstance(data["text"], str):
            # If a 'text' field exists, treat the entire text as a single claim.
            # In a more advanced system, an NLP step would extract discrete claims from the text.
            claims_to_check.append(data["text"])
            logger.debug(f"[{self.node_name}] Treating 'text' field as a single claim.")
        else:
            logger.warning(
                f"[{self.node_name}] Input data does not contain identifiable 'claims' (list of str) or 'text' (str) fields. "
                "No specific claims identified for fact-checking. Returning data with an empty results list."
            )
            # Ensure the output structure is consistent even if no claims were found.
            data["fact_check_results"] = []
            return data

        # If no claims were successfully extracted, log and return.
        if not claims_to_check:
            logger.warning(f"[{self.node_name}] No valid claims were extracted or provided to check. "
                           "Returning data with an empty results list.")
            data["fact_check_results"] = []
            return data

        fact_check_results: List[Dict[str, Any]] = []

        # --- Fact-Checking Logic ---
        for claim in claims_to_check:
            result: Dict[str, Any] = {"claim": claim}
            try:
                # Simulate checking the claim against the knowledge base
                status = self._knowledge_base.get(claim)

                if status is True:
                    result["status"] = True
                    result["reason"] = "Verified as true based on Vishustra's knowledge base."
                    logger.debug(f"[{self.node_name}] Claim '{claim}' verified as TRUE.")
                elif status is False:
                    result["status"] = False
                    result["reason"] = "Identified as false/misinformation based on Vishustra's knowledge base."
                    logger.debug(f"[{self.node_name}] Claim '{claim}' identified as FALSE.")
                else:
                    result["status"] = None
                    result["reason"] = "Could not be verified or refuted with current knowledge base (unknown claim)."
                    logger.warning(f"[{self.node_name}] Claim '{claim}' could not be verified/refuted. Not found in KB.")
            except Exception as e:
                # Catch any unexpected errors during the simulated check
                result["status"] = None
                result["reason"] = f"An error occurred during verification: {str(e)}"
                logger.error(f"[{self.node_name}] Error processing claim '{claim}': {e}", exc_info=True)
            finally:
                fact_check_results.append(result)
        
        # --- Output Augmentation ---
        data["fact_check_results"] = fact_check_results
        logger.info(f"[{self.node_name}] Completed fact-checking for {len(claims_to_check)} claims. "
                    f"Results added to 'fact_check_results' key.")

        # Log context if available, but do not modify it.
        if context:
            logger.debug(f"[{self.node_name}] Context keys provided: {list(context.keys())}")

        return data
