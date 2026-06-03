import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking statements against a predefined
    set of known facts. This node helps in verifying the truthfulness of textual data
    within the Vishustra pipeline.

    The node expects input `data` to be a string (representing a single statement)
    or a list of strings (for multiple statements). It queries a set of `known_facts`
    expected in the `context` dictionary and returns a verification status for each statement.
    """

    def __init__(self):
        """
        Initializes the FactCheckerNode.
        This simulated version does not require specific configuration at initialization,
        as the "known facts" for verification are intended to be provided dynamically
        via the `context` during the `process` call.
        """
        super().__init__()
        logger.debug("FactCheckerNode initialized, awaiting statements and known facts.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> List[Dict[str, Union[str, bool, None]]]:
        """
        Processes the input data by attempting to fact-check the provided statement(s).

        The input `data` can be a single string statement or a list of string statements.
        The `context` dictionary is critical and should optionally contain a 'known_facts' key.
        This 'known_facts' value is expected to be a dictionary where keys are statements
        (strings) and values are their boolean truth values (e.g., `{"The sky is blue": True}`).

        Args:
            data: The statement(s) to be fact-checked. Expected type is `str` or `List[str]`.
            context: A dictionary containing operational context, which *should* include
                     'known_facts' for effective verification.

        Returns:
            A list of dictionaries, where each dictionary represents a processed statement
            and its corresponding verification status. Each result dictionary contains:
            - "statement": The original statement string that was processed.
            - "is_fact": `True` if the statement is verified as true, `False` if verified
                         as false, or `None` if the statement could not be verified
                         (e.g., no matching known fact was found).
            - "reason": A brief string explaining the verification status.

        Raises:
            TypeError: If the input `data` is not a string or a list of strings.
            ValueError: If any item within a list of `data` is not a string.
        """
        processed_results: List[Dict[str, Union[str, bool, None]]] = []
        statements_to_check: List[str] = []

        # --- Input Data Validation and Normalization ---
        if isinstance(data, str):
            statements_to_check.append(data)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if not isinstance(item, str):
                    logger.error(
                        f"FactCheckerNode received non-string item at index {i} in data list. "
                        f"Type found: {type(item).__name__}, value: {item!r}."
                    )
                    raise ValueError(f"All items in 'data' list must be strings. Found type "
                                     f"{type(item).__name__} at index {i}.")
                statements_to_check.append(item)
        else:
            logger.error(
                f"FactCheckerNode received invalid input data type: {type(data).__name__}. "
                f"Expected str or List[str]."
            )
            raise TypeError(f"Input 'data' must be a string or a list of strings, "
                            f"but got {type(data).__name__}.")

        if not statements_to_check:
            logger.warning("FactCheckerNode received an empty list of statements to process. Returning empty results.")
            return []

        # --- Retrieve Known Facts from Context ---
        # In a production environment, this might involve calling an external microservice
        # or querying a dedicated knowledge base. Here, we simulate with a dictionary from context.
        known_facts: Dict[str, bool] = context.get("known_facts", {})
        if not known_facts:
            logger.warning(
                "FactCheckerNode operating without 'known_facts' in the context. "
                "All statements will be marked as 'UNVERIFIABLE'."
            )

        logger.info(f"FactCheckerNode starting processing for {len(statements_to_check)} statement(s).")

        # --- Simulate Fact-Checking Process ---
        for statement in statements_to_check:
            result: Dict[str, Union[str, bool, None]] = {
                "statement": statement,
                "is_fact": None,  # Default to None (unverifiable)
                "reason": "Unverifiable: No matching known fact found in context."
            }
            # Simple normalization for lookup: trim whitespace and convert to lower case
            normalized_statement_for_lookup = statement.strip().lower()

            found_match = False
            for known_stmt, truth_value in known_facts.items():
                if known_stmt.strip().lower() == normalized_statement_for_lookup:
                    result["is_fact"] = truth_value
                    result["reason"] = (
                        f"Verified as {'TRUE' if truth_value else 'FALSE'} "
                        f"against supplied known facts."
                    )
                    found_match = True
                    break
            
            # If no direct match in known_facts, the result remains 'UNVERIFIABLE' as per default.
            
            processed_results.append(result)
            log_msg_statement = statement[:100] + ("..." if len(statement) > 100 else "")
            logger.debug(
                f"Statement '{log_msg_statement}' processed: "
                f"Status: {result['is_fact']} | Reason: {result['reason']}"
            )

        logger.info(f"FactCheckerNode finished processing {len(processed_results)} statement(s).")
        return processed_results