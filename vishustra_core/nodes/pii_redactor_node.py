import logging
import re
from typing import Any, Dict, List, Union, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts Personally Identifiable Information (PII)
    from input data. It can process strings, dictionaries, and lists, recursively
    applying redaction to all string values.

    PII patterns are configurable during initialization, allowing for flexible
    definition of sensitive data types to be identified and replaced.
    """

    def __init__(self,
                 patterns: Optional[List[Dict[str, str]]] = None,
                 replacement_string: str = "[REDACTED]",
                 redaction_marker_format: str = "[REDACTED_{category}]"):
        """
        Initializes the PII Redactor Node with specified patterns and replacement rules.

        Args:
            patterns (Optional[List[Dict[str, str]]]): A list of dictionaries, where each
                dictionary defines a PII pattern. Each dictionary must contain:
                - "name" (str): A descriptive name for the PII category (e.g., "EMAIL").
                - "pattern" (str): The regular expression string to match the PII.
                If None, a set of default patterns will be used.
            replacement_string (str): The default string to replace detected PII with.
                Used if `redaction_marker_format` does not include `{category}`.
            redaction_marker_format (str): A format string for the replacement marker.
                If it contains `{category}`, it will be replaced with the PII pattern's
                "name" (e.g., "[REDACTED_EMAIL]"). Otherwise, `replacement_string` is used.
        """
        self._replacement_string = replacement_string
        self._redaction_marker_format = redaction_marker_format
        self._pii_patterns = self._compile_patterns(patterns)
        logger.info(f"[{self.node_name}] Initialized with {len(self._pii_patterns)} PII patterns.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII_Redactor"

    def _compile_patterns(self, patterns: Optional[List[Dict[str, str]]]) -> List[Dict[str, Union[str, re.Pattern]]]:
        """
        Compiles regex patterns provided during initialization for efficient matching.
        Sets up default patterns if none are provided.

        Args:
            patterns (Optional[List[Dict[str, str]]]): User-defined PII patterns.

        Returns:
            List[Dict[str, Union[str, re.Pattern]]]: A list of dictionaries with compiled patterns.
        """
        default_patterns = [
            {"name": "EMAIL", "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"},
            {"name": "PHONE_NUMBER", "pattern": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"},
            {"name": "SSN_LIKE", "pattern": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"}, # e.g., xxx-xx-xxxx
            {"name": "IP_ADDRESS", "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"},
            {"name": "CREDIT_CARD_LIKE", "pattern": r"\b(?:\d{4}[- ]){3}\d{4}\b"}
        ]
        
        compiled_patterns = []
        effective_patterns = patterns if patterns is not None else default_patterns

        for pii_item in effective_patterns:
            try:
                if not isinstance(pii_item, dict) or "name" not in pii_item or "pattern" not in pii_item:
                    logger.warning(
                        f"[{self.node_name}] Invalid PII pattern definition: {pii_item}. "
                        "Expected a dict with 'name' and 'pattern' keys. Skipping."
                    )
                    continue

                compiled_patterns.append({
                    "name": pii_item["name"],
                    "pattern": re.compile(pii_item["pattern"])
                })
            except re.error as e:
                logger.error(f"[{self.node_name}] Failed to compile regex pattern '{pii_item.get('pattern', 'N/A')}': {e}. Skipping this pattern.")
            except Exception as e:
                logger.exception(f"[{self.node_name}] An unexpected error occurred while processing pattern '{pii_item.get('pattern', 'N/A')}': {e}. Skipping this pattern.")
        return compiled_patterns

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured PII patterns to a given string and redacts matches.

        Args:
            text (str): The input string to redact.

        Returns:
            str: The string with detected PII replaced by redaction markers.
        """
        redacted_text = text
        for pii_item in self._pii_patterns:
            name = pii_item["name"]
            pattern = pii_item["pattern"]
            
            # Determine the redaction marker based on configuration
            marker = (
                self._redaction_marker_format.replace("{category}", name)
                if "{category}" in self._redaction_marker_format
                else self._replacement_string
            )
            
            # Replace all occurrences of the pattern
            # logger.debug(f"[{self.node_name}] Applying pattern '{name}' to text fragment: '{redacted_text[:100]}...'")
            redacted_text = pattern.sub(marker, redacted_text)
        return redacted_text

    def _traverse_and_redact(self, data: Any) -> Any:
        """
        Recursively traverses data structures (strings, dicts, lists) and applies
        PII redaction to all string values.

        Args:
            data (Any): The input data, which can be a string, dict, list, or other type.

        Returns:
            Any: The data with all string values (that contained PII) redacted.
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            return {k: self._traverse_and_redact(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._traverse_and_redact(item) for item in data]
        else:
            # For other types (int, bool, float, None, etc.), return as is
            return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        The `context` parameter is currently not used for dynamic PII pattern
        updates in this implementation, but it is available for future extensions
        to support runtime configuration overrides.

        Args:
            data (Any): The input data to be processed. This can be a string,
                        dictionary, or list.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     for the processing operation.

        Returns:
            Any: The processed data with PII redacted.

        Raises:
            Exception: If an unexpected error occurs during the redaction process.
        """
        logger.debug(f"[{self.node_name}] Starting PII redaction process for input data type: {type(data)}.")
        try:
            # The core logic is handled by the recursive _traverse_and_redact method.
            result = self._traverse_and_redact(data)
            logger.debug(f"[{self.node_name}] PII redaction complete.")
            return result
        except Exception as e:
            logger.exception(f"[{self.node_name}] An error occurred during PII redaction: {e}")
            # Re-raise the exception after logging for upstream orchestration to handle
            raise
