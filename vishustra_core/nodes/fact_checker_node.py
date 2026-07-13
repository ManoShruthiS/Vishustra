import logging
from typing import Any, Dict, List, Tuple

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking statements against a predefined
    or context-provided knowledge base. It aims to verify the veracity of textual claims.
    """

    # --- Internal simulated knowledge base for demonstration ---
    _KNOWN_FACTS: Dict[str, str] = {
        "The capital of France is Paris.": "VERIFIED",
        "Water boils at 100 degrees Celsius at sea level.": "VERIFIED",
        "The sun revolves around the Earth.": "UNVERIFIED", # This is a false statement
        "Vishustra is a highly modular LLM orchestration framework.": "VERIFIED",
        "Artificial intelligence is sentient.": "UNCERTAIN", # A nuanced example
        "The moon is made of cheese.": "UNVERIFIED",
        "Elephants can fly.": "UNVERIFIED",
        "Birds can fly.": "VERIFIED",
    }
    _KEYWORDS_TO_TRUTH: Dict[str, str] = {
        "Paris": "VERIFIED",
        "France capital": "VERIFIED",
        "water boils": "VERIFIED",
        "sun revolves around earth": "UNVERIFIED",
        "AI sentient": "UNCERTAIN",
        "Vishustra framework": "VERIFIED",
        "moon cheese": "UNVERIFIED",
        "elephants fly": "UNVERIFIED",
        "birds fly": "VERIFIED",
        "earth is flat": "UNVERIFIED",
    }
    # --- End of internal simulated knowledge base ---

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def _check_statement_against_kb(self, statement: str) -> Tuple[str, List[str]]:
        """
        Simulates checking a single statement against the node's internal knowledge base.
        Returns a tuple of (status, details_list).
        """
        statement_lower = statement.lower()
        details: List[str] = []

        # Check for exact matches first
        for fact, status in self._KNOWN_FACTS.items():
            if statement_lower == fact.lower():
                details.append(f"Exact match found in internal known facts. Status: {status}.")
                logger.debug(f"Statement '{statement}' exact matched to '{fact}', status: {status}")
                return status, details

        # Check for keyword matches
        for keyword, status in self._KEYWORDS_TO_TRUTH.items():
            if keyword.lower() in statement_lower:
                details.append(f"Keyword match found in internal knowledge base for '{keyword}'. Status: {status}.")
                logger.debug(f"Statement '{statement}' keyword matched to '{keyword}', status: {status}")
                return status, details
        
        details.append("No direct match or significant keyword found in the simulated knowledge base.")
        logger.debug(f"Statement '{statement}' did not find a direct or keyword match.")
        return "UNCERTAIN", details

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to perform fact-checking.

        This method supports various input formats and attempts to extract
        statements for verification. It prioritizes an external fact-checking
        service provided in the `context` if available, otherwise, it falls
        back to its internal simulated knowledge base.

        Expected `data` formats:
        - A `str` representing a single statement to check.
        - A `Dict[str, Any]` containing a 'statement' key with the text to check.
        - A `List[str]` where each string is a statement.
        - A `List[Dict[str, Any]]` where each dict contains a 'statement' key.

        Expected `context` keys (optional):
        - `fact_checking_service`: A callable (e.g., a function or method) that
          takes a `str` statement and returns a `Tuple[str, List[str]]`
          (status, details). If provided, this service will be used instead
          of the internal simulation.

        Returns:
            A dictionary if the input `data` was a single item (str or dict),
            or a list of dictionaries if the input `data` was a list. Each
            result dictionary contains:
            - 'original_text': The text that was checked.
            - 'status': One of 'VERIFIED', 'UNVERIFIED', 'UNCERTAIN', 'ERROR'.
            - 'details': A list of strings explaining the verification process or findings.
            - 'original_input_structure': The original piece of data that was processed
                                          (e.g., the string or the dict it came from).
        
        Raises:
            TypeError: If the input `data` is of an unsupported type.
        """
        logger.info(f"FactCheckerNode processing data of type: {type(data)}")
        results: List[Dict[str, Any]] = []

        statements_to_check: List[str] = []
        original_data_mapping: List[Any] = [] # To map results back to original structure

        # Normalize input data into a list of statements
        if isinstance(data, str):
            statements_to_check.append(data)
            original_data_mapping.append(data)
        elif isinstance(data, dict):
            if "statement" in data and isinstance(data["statement"], str):
                statements_to_check.append(data["statement"])
                original_data_mapping.append(data)
            else:
                logger.warning(
                    "Input dict does not contain a 'statement' key or its value is not a string. "
                    "Returning an error status for this item."
                )
                return {
                    "status": "ERROR",
                    "details": ["Invalid input format for dictionary data. Expected a 'statement' key with a string value."],
                    "original_input_structure": data
                }
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str):
                    statements_to_check.append(item)
                    original_data_mapping.append(item)
                elif isinstance(item, dict) and "statement" in item and isinstance(item["statement"], str):
                    statements_to_check.append(item["statement"])
                    original_data_mapping.append(item)
                else:
                    logger.warning(f"List item at index {i} of unexpected format: {type(item)}. Skipping processing for this item.")
                    results.append({
                        "original_data_item": item,
                        "status": "ERROR",
                        "details": ["Invalid format for list item. Expected string or dict with 'statement' key."],
                        "original_input_structure": item
                    })
        else:
            logger.error(f"Unsupported data type for FactCheckerNode: {type(data)}")
            raise TypeError(
                f"FactCheckerNode received unsupported data type: {type(data)}. "
                "Expected str, dict with 'statement', or list of these."
            )

        if not statements_to_check and not results: # If no valid statements extracted and no errors added to results
            logger.info("No valid statements found to fact-check in the input data.")
            # Return empty list for list input, or a specific status for single item input
            return [] if isinstance(data, list) else {"status": "UNCERTAIN", "details": ["No statements processed."], "original_input_structure": data}


        # Process each extracted statement
        for i, statement in enumerate(statements_to_check):
            current_original_input = original_data_mapping[i]
            try:
                # Prioritize external fact-checking service if provided in context
                if "fact_checking_service" in context and callable(context["fact_checking_service"]):
                    logger.debug("Using external fact-checking service from context for statement: '%s'", statement)
                    status, details = context["fact_checking_service"](statement)
                else:
                    logger.debug("Using internal simulated knowledge base for statement: '%s'", statement)
                    status, details = self._check_statement_against_kb(statement)

                results.append({
                    "original_text": statement,
                    "status": status,
                    "details": details,
                    "original_input_structure": current_original_input
                })
            except Exception as e:
                logger.exception(f"Error fact-checking statement '{statement}': {e}")
                results.append({
                    "original_text": statement,
                    "status": "ERROR",
                    "details": [f"An unexpected error occurred during processing: {str(e)}"],
                    "original_input_structure": current_original_input
                })

        # If the original input was a single item (str or dict), return a single result dict
        if isinstance(data, (str, dict)) and len(results) == 1:
            return results[0]
        
        return results