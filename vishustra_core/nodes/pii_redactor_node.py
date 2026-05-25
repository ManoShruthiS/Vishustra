import logging
import re
from typing import Any, Dict, List, Union, Set, Tuple

# Assuming BaseNode is located in vishustra_core.nodes.base_node relative to the project root
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that identifies and redacts common Personally
    Identifiable Information (PII) from textual content within various data
    structures.

    This node is designed to sanitize data by replacing detected PII patterns
    (e.g., email addresses, phone numbers, simplified credit card numbers,
    social security numbers, IP addresses, dates) with '[REDACTED]'.
    It supports recursive redaction within strings, dictionaries, lists, tuples,
    and sets.
    """

    # Pre-compiled regular expression patterns for common PII types.
    # The order of these patterns is considered for potential overlaps.
    _PII_PATTERNS = [
        # Email addresses: basic format
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        # US Phone numbers: common formats (e.g., (123) 456-7890, 123.456.7890, 123-456-7890)
        re.compile(r'(\+?\d{1,2}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'),
        # Simplified Credit Card Numbers: 13-16 digits, with optional spaces/hyphens
        # Note: True credit card validation requires Luhn algorithm, this is a pattern match.
        re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
        # US Social Security Numbers (SSN): XXX-XX-XXXX format
        re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        # IP Addresses: IPv4 format
        re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
        # Dates: simple DD/MM/YYYY or MM-DD-YYYY or YYYY-MM-DD
        re.compile(r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b'),
    ]

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this processing node.
        """
        return "PIIRedactorNode"

    def _redact_string(self, text: str) -> str:
        """
        Applies all defined PII redaction patterns to a given string.

        Args:
            text: The input string potentially containing PII.

        Returns:
            The string with all detected PII replaced by '[REDACTED]'.
        """
        redacted_text = text
        for pattern in self._PII_PATTERNS:
            try:
                redacted_text = pattern.sub('[REDACTED]', redacted_text)
            except re.error as e:
                logger.error(f"[{self.node_name}] Regex pattern error encountered: {e} for pattern '{pattern.pattern}'", exc_info=True)
            except Exception as e:
                logger.error(f"[{self.node_name}] Unexpected error during string redaction with pattern '{pattern.pattern}': {e}", exc_info=True)
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        This method traverses the input `data` structure recursively.
        It redacts string values based on predefined PII patterns.
        Dictionaries, lists, tuples, and sets are iterated, and their
        string elements are processed. Other data types (e.g., numbers,
        booleans) are returned as-is.

        Args:
            data: The input data to be processed. This can be a string,
                  dict, list, tuple, set, or other primitive types.
            context: A dictionary providing runtime context. This node
                     does not currently use the context for configuration
                     but it's available for future extensions (e.g.,
                     dynamic PII patterns, redaction placeholder).

        Returns:
            The processed data with identified PII redacted.

        Raises:
            RecursionError: If the input data contains deeply nested or
                            self-referential structures that exceed Python's
                            recursion limit.
            Exception: For any other unexpected errors during processing.
        """
        logger.debug(f"[{self.node_name}] Initiating PII redaction for input data of type: {type(data)}")

        if data is None:
            logger.debug(f"[{self.node_name}] Input data is None. Returning as-is.")
            return None

        def _recursive_redact_value(item: Any) -> Any:
            """
            Internal helper function to recursively redact PII within various
            data types.
            """
            if isinstance(item, str):
                return self._redact_string(item)
            elif isinstance(item, dict):
                # Create a new dictionary to avoid modifying the original during iteration
                redacted_dict = {k: _recursive_redact_value(v) for k, v in item.items()}
                return redacted_dict
            elif isinstance(item, list):
                # Create a new list with redacted elements
                return [_recursive_redact_value(element) for element in item]
            elif isinstance(item, tuple):
                # Convert tuple to list for redaction, then back to tuple
                return tuple(_recursive_redact_value(list(item)))
            elif isinstance(item, set):
                # Convert set to list for redaction, then back to set
                # Note: This might change the order of elements in the set due to list conversion
                return {_recursive_redact_value(element) for element in list(item)}
            else:
                # Return other data types (numbers, booleans, objects) as-is
                return item

        try:
            redacted_output = _recursive_redact_value(data)
            logger.debug(f"[{self.node_name}] PII redaction process completed successfully.")
            return redacted_output
        except RecursionError:
            logger.exception(f"[{self.node_name}] Recursion limit exceeded during PII redaction. "
                             "Check for extremely deeply nested or self-referential data structures.")
            raise # Re-raise to indicate a critical processing failure
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during PII redaction: {e}")
            # Re-raise the exception to propagate the error up the orchestration chain
            raise
