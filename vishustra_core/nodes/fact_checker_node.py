import logging
from typing import Any, Dict, Union, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking of textual statements.

    This node takes a statement (either directly as a string or within a dictionary
    under the 'statement' key) and attempts to verify its accuracy against a
    simulated internal knowledge base. It returns a structured result indicating
    accuracy, confidence, and any identified discrepancies.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def __init__(self):
        """
        Initializes the FactCheckerNode with a simulated knowledge base.
        In a real-world scenario, this would typically connect to an external
        fact-checking API or a robust internal data source.
        """
        self._knowledge_base: Dict[str, bool] = {
            "The capital of France is Paris.": True,
            "The Earth is flat.": False,
            "Water boils at 100 degrees Celsius at sea level.": True,
            "The moon is made of cheese.": False,
            "Python is a compiled language.": False,
            "The sun orbits the Earth.": False,
            "Mount Everest is the highest mountain in the world.": True,
        }
        logger.info(f"{self.node_name} initialized with simulated knowledge base.")

    def process(self, data: Union[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to perform fact-checking.

        Expects `data` to be either a string containing the statement or a dictionary
        with a 'statement' key.

        The `context` dictionary can optionally provide:
        - 'simulate_error': A boolean to trigger an artificial error for testing purposes.
        - 'confidence_threshold': A float (0.0-1.0) to filter results by confidence.

        Args:
            data: The input data, expected to be a string statement or a dictionary
                  containing a 'statement' key.
            context: A dictionary containing additional runtime information or configurations.

        Returns:
            A dictionary containing the fact-checking result with the following keys:
            - 'original_statement': The statement that was checked.
            - 'is_accurate': bool | None (True if accurate, False if inaccurate, None if uncertain/unverifiable).
            - 'confidence': float (0.0 to 1.0, representing the confidence in the 'is_accurate' determination).
            - 'evidence': List[str] (list of supporting evidence or reasons).
            - 'details': str (a descriptive message about the fact-checking outcome).
            - 'error': str | None (an error message if an issue occurred during processing).
        """
        original_statement: Union[str, None] = None
        result: Dict[str, Any] = {
            'original_statement': None,
            'is_accurate': None,
            'confidence': 0.0,
            'evidence': [],
            'details': "No fact-checking performed due to input or processing error.",
            'error': None
        }

        # Simulate an external service error if requested via context
        if context.get('simulate_error', False):
            error_msg = "Simulated external service error during fact-checking operation."
            logger.error(error_msg)
            result['error'] = error_msg
            result['details'] = "Failed to connect to the simulated fact-checking service."
            return result

        # Validate and extract the statement from input data
        if isinstance(data, str):
            original_statement = data
        elif isinstance(data, dict) and 'statement' in data and isinstance(data['statement'], str):
            original_statement = data['statement']
        else:
            error_msg = (f"Invalid input data format for {self.node_name}. "
                         f"Expected string or dict with 'statement' key, got: {type(data).__name__}.")
            logger.error(error_msg)
            result['error'] = error_msg
            result['details'] = "Input data did not match expected format for fact-checking."
            return result
        
        result['original_statement'] = original_statement

        try:
            # Simulate fact-checking against the internal knowledge base
            is_accurate = self._knowledge_base.get(original_statement)
            
            if is_accurate is not None:
                # Statement found in the knowledge base
                result['is_accurate'] = is_accurate
                result['confidence'] = 0.95  # High confidence for known facts
                result['evidence'] = [f"Found in internal knowledge base: '{original_statement}' is {('true' if is_accurate else 'false')}."]
                result['details'] = "Statement verified against internal knowledge base."
                logger.info(f"Statement '{original_statement}' checked: Accurate={is_accurate}")
            else:
                # Statement not found, simulate a lower confidence "unknown" state
                result['is_accurate'] = None  # Cannot definitively say true or false
                result['confidence'] = 0.3    # Low confidence as it's an educated guess
                result['evidence'] = []
                result['details'] = "Statement not found in internal knowledge base; requires further verification."
                logger.warning(f"Statement '{original_statement}' not found in knowledge base. Cannot definitively verify.")

            # Apply an optional confidence threshold from context if provided
            confidence_threshold = context.get('confidence_threshold', 0.5)
            if result['confidence'] < confidence_threshold:
                result['details'] += f" (Confidence {result['confidence']:.2f} below threshold {confidence_threshold:.2f})"
                # If confidence is too low, even if we had an initial accuracy, mark as uncertain
                if result['is_accurate'] is not None: 
                    result['is_accurate'] = None 
                logger.debug(f"Confidence {result['confidence']:.2f} for '{original_statement}' below threshold {confidence_threshold:.2f}.")

        except Exception as e:
            error_msg = f"An unexpected error occurred during fact-checking for statement '{original_statement}': {e}"
            logger.error(error_msg, exc_info=True)
            result['error'] = error_msg
            result['is_accurate'] = None
            result['confidence'] = 0.0
            result['details'] = "Fact-checking failed due to an internal error."

        return result