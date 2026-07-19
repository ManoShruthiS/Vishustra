import logging
from typing import Any, Dict, Union

# Import the base node class from the framework's core
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node designed to simulate fact-checking of a given statement.

    This node takes an input statement (either as a string or embedded in a dictionary)
    and evaluates its veracity. For this implementation, the fact-checking mechanism
    is simulated using internal, hardcoded rules. In a production scenario, this node
    would integrate with external fact-checking APIs, knowledge graphs, or NLP models
    to perform real-time verification.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this node.
        """
        return "FactCheckerNode"

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to perform a simulated fact-check on a statement.

        The `data` input can be either:
        - A `str`: The direct statement to be fact-checked.
        - A `Dict[str, Any]`: Must contain a 'statement' key whose value is the string
          to be fact-checked. Additional keys in the dictionary will be ignored
          by the core fact-checking logic but might be preserved in the output.

        The `context` dictionary provides runtime context, which can include parameters
        like 'fact_check_sources' (a list of preferred sources) or 'confidence_threshold'.
        This simulation primarily uses context for illustrative logging and result enrichment.

        Args:
            data: The statement to be fact-checked or a dictionary containing it.
            context: A dictionary providing contextual information for processing.

        Returns:
            A `Dict[str, Any]` containing:
            - 'original_statement': The statement that was checked.
            - 'is_fact': `True` if verified as true, `False` if verified as false,
                         `None` if the veracity could not be determined.
            - 'reasoning': A string explaining the outcome of the fact-check.
            - 'checked_by': The name of this node.
            - 'context_info_used': An example of how context data might be reflected.

        Raises:
            TypeError: If the `data` provided is neither a string nor a dictionary.
            ValueError: If `data` is a dictionary but lacks a valid 'statement' key,
                        or if the statement itself is empty or consists only of whitespace.
        """
        statement: str
        original_input_data = data # Keep a reference to the original input for error messages/logging

        # Extract the statement from the input data
        if isinstance(data, str):
            statement = data
        elif isinstance(data, dict):
            if 'statement' not in data or not isinstance(data.get('statement'), str):
                logger.error(
                    f"FactCheckerNode received a dictionary without a valid 'statement' key "
                    f"or its value is not a string. Data: {original_input_data}"
                )
                raise ValueError(
                    "Input dictionary must contain a 'statement' key with a non-empty string value."
                )
            statement = data['statement']
        else:
            logger.error(
                f"Unsupported data type received by FactCheckerNode. Expected `str` or `Dict[str, Any]`, "
                f"but got `{type(data).__name__}`. Data: {original_input_data}"
            )
            raise TypeError(
                f"FactCheckerNode expects 'data' to be a string or a dictionary "
                f"with a 'statement' key, not {type(data).__name__}."
            )

        if not statement.strip():
            logger.error(f"Received an empty or whitespace-only statement for fact-checking. Data: {original_input_data}")
            raise ValueError("Statement for fact-checking cannot be empty.")

        logger.info(f"Initiating fact-check for statement: '{statement[:100]}{'...' if len(statement) > 100 else ''}'")

        # --- Simulated Fact-Checking Logic ---
        # This section simulates the core logic. In a real-world application,
        # this would involve calls to external services, complex NLP, or database lookups.
        is_fact: Union[bool, None] = None
        reasoning: str = "Simulated: Initial lookup yielded no definitive result."

        # Normalize the statement for easier comparison in this simulation
        normalized_statement = statement.lower().strip().replace('.', '').replace('!', '').replace('?', '')

        # A small internal knowledge base for demonstration
        known_facts_db = {
            "water boils at 100 degrees celsius at sea level": True,
            "the earth is flat": False,
            "paris is the capital of france": True,
            "the moon is made of cheese": False,
            "birds are reptiles": False,
            "the sun is a star": True,
            "human beings can fly without assistance": False
        }

        if normalized_statement in known_facts_db:
            is_fact = known_facts_db[normalized_statement]
            reasoning = "Simulated: Matched against a small internal knowledge base."
            logger.debug(f"Statement '{statement[:50]}...' matched in known facts. Result: {is_fact}")
        else:
            # Further simple heuristic for simulation
            if "flat earth" in normalized_statement:
                is_fact = False
                reasoning = "Simulated: Identified as a widely disproven claim."
                logger.debug(f"Statement '{statement[:50]}...' identified as a common misconception.")
            elif "capital" in normalized_statement and "france" in normalized_statement and "paris" in normalized_statement:
                is_fact = True
                reasoning = "Simulated: Recognized as common geographical knowledge."
                logger.debug(f"Statement '{statement[:50]}...' recognized as common knowledge via keywords.")
            elif "sun" in normalized_statement and "star" in normalized_statement:
                is_fact = True
                reasoning = "Simulated: Recognized as basic astronomical knowledge."
                logger.debug(f"Statement '{statement[:50]}...' recognized as basic knowledge via keywords.")
            else:
                is_fact = None # Undetermined
                reasoning = "Simulated: Internal knowledge base and heuristics could not definitively verify or refute."
                logger.info(f"Statement '{statement[:50]}...' could not be definitively verified by simulation.")

        # Construct the result dictionary
        result = {
            "original_statement": statement,
            "is_fact": is_fact,
            "reasoning": reasoning,
            "checked_by": self.node_name,
            # Example of passing context information through the result
            "context_info_used": context.get("fact_check_sources", ["N/A - Simulation"])
        }

        logger.info(
            f"Fact-check completed for statement '{statement[:100]}{'...' if len(statement) > 100 else ''}'. "
            f"Result: is_fact={is_fact}"
        )
        return result
