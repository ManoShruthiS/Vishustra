import logging
import re
from typing import Any, Dict, List, Union

# Assuming the BaseNode is located here as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node responsible for identifying and redacting Personally
    Identifiable Information (PII) from input data.

    This node uses regular expressions to find common PII patterns like email addresses,
    phone numbers, and other sensitive identifiers, replacing them with a configurable
    placeholder string. It can process strings, dictionaries, and lists recursively.
    """

    def __init__(self, patterns: Dict[str, str] = None, replacement_text: str = "[REDACTED]"):
        """
        Initializes the PII Redactor Node with custom patterns and replacement text.

        Args:
            patterns (Dict[str, str], optional): A dictionary where keys are descriptive
                                                 names for PII types (e.g., "email", "phone")
                                                 and values are raw regex strings to identify
                                                 those PII types. If None, a set of default
                                                 common PII patterns will be used.
            replacement_text (str, optional): The string used to replace identified PII.
                                              Defaults to "[REDACTED]".
        """
        self._replacement_text = replacement_text
        self._patterns = patterns if patterns is not None else self._default_patterns()
        self._compiled_patterns = {
            key: re.compile(pattern, re.IGNORECASE)
            for key, pattern in self._patterns.items()
        }
        logger.debug(f"[{self.node_name}] Initialized with patterns: {list(self._patterns.keys())}")

    def _default_patterns(self) -> Dict[str, str]:
        """
        Provides a set of default regular expression patterns for common PII types.
        """
        return {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_us": r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "ssn_like": r'\b\d{3}[- ]?\d{2}[- ]?\d{4}\b', # Simplified for US SSN-like patterns
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "url": r'(https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*)', # Basic URL pattern
        }

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies configured redaction patterns to a single string.

        Args:
            text (str): The input string to redact.

        Returns:
            str: The string with PII replaced by the configured replacement text.
        """
        redacted_text = text
        for pii_type, compiled_pattern in self._compiled_patterns.items():
            # Check if any match exists before substitution for logging clarity
            if compiled_pattern.search(redacted_text):
                redacted_text = compiled_pattern.sub(self._replacement_text, redacted_text)
                logger.debug(f"[{self.node_name}] Redacted '{pii_type}' in string data.")
        return redacted_text

    def _redact_data_recursive(self, data: Any) -> Any:
        """
        Recursively traverses and redacts PII within dictionaries, lists, and strings.

        Args:
            data (Any): The data structure (string, dict, or list) to process.

        Returns:
            Any: The processed data structure with PII redacted.
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            return {k: self._redact_data_recursive(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._redact_data_recursive(elem) for elem in data]
        else:
            # For non-string, non-dict, non-list types, we simply return them as-is.
            # These types are unlikely to contain PII in a text format directly
            # relevant to this node's regex-based redaction.
            logger.debug(
                f"[{self.node_name}] Skipping redaction for data of unsupported base type: "
                f"'{type(data).__name__}'. Returning as-is."
            )
            return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact Personally Identifiable Information (PII).

        The node iterates through its configured PII patterns (email, phone, SSN-like, etc.)
        and replaces any matched text with the `replacement_text` (default: '[REDACTED]').
        It is capable of processing data provided as strings, dictionaries, or lists,
        recursively applying redaction to nested structures. Primitive types (like int, bool)
        and other unsupported types will be returned as-is.

        Args:
            data (Any): The input data to be processed. Expected to be a string, dictionary,
                        or list containing strings, or any other type to be returned as-is.
            context (Dict[str, Any]): A dictionary containing runtime context information
                                     passed along the orchestration pipeline.
                                     (Not directly utilized by this node for redaction logic).

        Returns:
            Any: The processed data with identified PII redacted. The original data type
                 and structure are preserved where possible.

        Raises:
            Exception: Propagates any unexpected errors that occur during the redaction
                       process to ensure pipeline integrity.
        """
        logger.info(f"[{self.node_name}] Starting PII redaction process.")

        try:
            redacted_data = self._redact_data_recursive(data)
            logger.info(f"[{self.node_name}] PII redaction completed successfully.")
            return redacted_data
        except Exception as e:
            logger.error(
                f"[{self.node_name}] Critical error occurred during PII redaction: {e}",
                exc_info=True
            )
            # Re-raise to indicate a failure in the node's core operation,
            # allowing the orchestration framework to handle the pipeline failure.
            raise