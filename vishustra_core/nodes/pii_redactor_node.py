from vishustra_core.nodes.base_node import BaseNode
from typing import Any, Dict, List, Union
import logging
import re

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra node that redacts personally identifiable information (PII)
    from input text data using configurable regex patterns.

    This node supports redacting PII from single strings or lists of strings.
    The specific PII patterns to be detected and the replacement string
    can be configured dynamically via the processing context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor"

    def _get_default_pii_patterns(self) -> List[Dict[str, str]]:
        """
        Provides a default set of regex patterns for common PII types.
        These patterns are intended to be a robust starting point but may
        require tuning based on specific data characteristics and privacy needs.
        """
        return [
            # Email addresses (common formats)
            {"name": "Email Address", "regex": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'},
            # US Phone numbers (e.g., (123) 456-7890, 123-456-7890, +1 123-456-7890)
            {"name": "Phone Number", "regex": r'\b(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b'},
            # US Social Security Numbers (e.g., XXX-XX-XXXX)
            {"name": "SSN", "regex": r'\b\d{3}-\d{2}-\d{4}\b'},
            # IPv4 Addresses
            {"name": "IP Address", "regex": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'},
            # Basic credit card numbers (13-16 digits, with optional spaces/hyphens) - note: very broad
            {"name": "Credit Card Number", "regex": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})\b(?:[ -]?\d{4}){3,}\b'},
            # Dates in YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY formats (generic, might redact non-PII dates)
            {"name": "Date", "regex": r'\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b'},
            # Generic UUIDs/GUIDs
            {"name": "UUID/GUID", "regex": r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'},
        ]

    def _redact_single_string(self, text: str, patterns: List[Dict[str, str]], replacement: str) -> str:
        """
        Applies a list of PII redaction patterns to a single input string.

        Args:
            text: The string to be redacted.
            patterns: A list of dictionaries, each containing 'name' and 'regex' keys
                      for a specific PII type.
            replacement: The string to substitute for detected PII.

        Returns:
            The string with all identified PII redacted.
        """
        redacted_text = text
        for pattern_info in patterns:
            pattern_name = pattern_info.get("name", "Unknown PII Type")
            regex_str = pattern_info.get("regex")
            
            if not regex_str:
                logger.warning(f"PII Redactor: Skipping pattern '{pattern_name}' due to missing regex string.")
                continue
            
            try:
                # Compile regex for efficiency if used multiple times, but `re.sub` handles it.
                # Using re.IGNORECASE for broader matching, common for some PII.
                redacted_text = re.sub(regex_str, replacement, redacted_text, flags=re.IGNORECASE)
                logger.debug(f"PII Redactor: Applied redaction for '{pattern_name}'.")
            except re.error as e:
                logger.error(f"PII Redactor: Invalid regex pattern '{regex_str}' for '{pattern_name}': {e}. Skipping this pattern.")
                # Continue processing with other valid patterns
                continue
        return redacted_text

    def process(self, data: Union[str, List[str]], context: Dict[str, Any]) -> Union[str, List[str]]:
        """
        Processes the input data (string or list of strings) to redact PII
        based on configured patterns in the context.

        Args:
            data: The input data, which can be a single string or a list of strings
                  to be processed for PII redaction.
            context: A dictionary containing runtime information and configuration.
                     Expected keys for configuration (within 'pii_redactor_config' dict):
                       - 'patterns': `List[Dict[str, str]]`, where each dict must have
                                     'name' and 'regex' keys. If not provided or invalid,
                                     a default set of common PII patterns is used.
                       - 'replacement_string': `str`, the string to replace detected PII with.
                                             Defaults to "[REDACTED]" if not provided or invalid.

        Returns:
            The data with PII redacted, preserving the original input type
            (a single string if input was str, or a list of strings if input was List[str]).

        Raises:
            ValueError: If the input 'data' is not a string or a list of strings.
        """
        if not isinstance(data, (str, list)):
            logger.error(f"PII Redactor: Invalid input data type. Expected str or List[str], got {type(data)}.")
            raise ValueError(f"PII Redactor: Input 'data' must be a string or a list of strings, got {type(data)}.")

        # Extract configuration from the context, providing sensible defaults
        config = context.get('pii_redactor_config', {})
        
        patterns_to_redact = config.get('patterns')
        if not isinstance(patterns_to_redact, list) or \
           not all(isinstance(p, dict) and 'name' in p and 'regex' in p for p in patterns_to_redact):
            logger.info("PII Redactor: 'patterns' in context is invalid or not provided. Using default PII patterns.")
            patterns_to_redact = self._get_default_pii_patterns()
        
        replacement_string = config.get('replacement_string', "[REDACTED]")
        if not isinstance(replacement_string, str):
            logger.warning(
                "PII Redactor: 'replacement_string' in context is not a string. "
                "Falling back to default '[REDACTED]'."
            )
            replacement_string = "[REDACTED]"

        logger.info(f"PII Redactor: Starting redaction with {len(patterns_to_redact)} patterns using '{replacement_string}'.")

        if isinstance(data, str):
            redacted_data = self._redact_single_string(data, patterns_to_redact, replacement_string)
        else:  # data is List[str]
            redacted_data = [
                self._redact_single_string(item, patterns_to_redact, replacement_string)
                for item in data
            ]
        
        logger.info("PII Redactor: Redaction process completed.")
        return redacted_data