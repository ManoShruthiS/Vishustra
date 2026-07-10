import re
import logging
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra node designed to redact Personally Identifiable Information (PII)
    from text data. This includes processing text within strings, dictionaries,
    and lists by recursively traversing the data structure.

    The node comes with a set of default regex patterns for common PII types
    (e.g., email addresses, US phone numbers, credit card numbers, US SSNs),
    and allows for custom patterns to be provided during initialization.
    """

    def __init__(self, custom_patterns: Dict[str, str] = None):
        """
        Initializes the PIIRedactorNode with default or custom PII patterns.

        Args:
            custom_patterns (Dict[str, str], optional): A dictionary where keys are
                descriptive names (e.g., "email") and values are regex patterns
                (as strings). If provided, these patterns will *replace* the
                node's default patterns. Placeholders for custom patterns
                will default to '[REDACTED]' unless explicitly matched by a
                default pattern's key.
        """
        self._default_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_us": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "credit_card": r'\b(?:4\d{3}|5[1-5]\d{2}|6011|3[47]\d{2})[- ]?(?:\d{4}[- ]?){3}\b', # More specific CC pattern
            "ssn_us": r'\b\d{3}[- ]\d{2}[- ]\d{4}\b',
        }
        self._default_placeholders = {
            "email": "[REDACTED_EMAIL]",
            "phone_us": "[REDACTED_PHONE]",
            "credit_card": "[REDACTED_CREDIT_CARD]",
            "ssn_us": "[REDACTED_SSN]",
        }

        # Determine the active set of patterns based on custom_patterns
        self.patterns = custom_patterns if custom_patterns is not None else self._default_patterns
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._active_placeholders: Dict[str, str] = {}

        for name, pattern_str in self.patterns.items():
            try:
                self._compiled_patterns[name] = re.compile(pattern_str)
                # Use default placeholder if the name matches, otherwise a generic one
                self._active_placeholders[name] = self._default_placeholders.get(name, "[REDACTED]")
            except re.error as e:
                logger.error(f"[{self.node_name}] Failed to compile regex pattern '{name}': '{pattern_str}'. Error: {e}. This pattern will be skipped.")
                # Continue to process other patterns even if one fails
        
        logger.debug(f"[{self.node_name}] Initialized with active patterns: {list(self._compiled_patterns.keys())}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured redaction patterns to a given string.

        Args:
            text (str): The input string to be redacted.

        Returns:
            str: The string with PII patterns replaced by their respective placeholders.
        """
        if not isinstance(text, str):
            logger.warning(
                f"[{self.node_name}] Expected string for redaction, but received {type(text)}. "
                "Returning original data type without redaction."
            )
            return text

        redacted_text = text
        total_redactions = 0
        for pattern_name, compiled_pattern in self._compiled_patterns.items():
            placeholder = self._active_placeholders.get(pattern_name, "[REDACTED]")
            
            # Use re.subn to get the count of substitutions made
            new_text, subs_made = compiled_pattern.subn(placeholder, redacted_text)
            
            if subs_made > 0:
                redacted_text = new_text
                total_redactions += subs_made
                logger.debug(f"[{self.node_name}] Redacted {subs_made} occurrences of '{pattern_name}'.")

        if total_redactions > 0:
            logger.info(f"[{self.node_name}] Completed string redaction. Total {total_redactions} PII items redacted.")
        else:
            logger.debug(f"[{self.node_name}] No PII items found for redaction in string.")
        
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        This method supports recursive redaction for strings, dictionaries,
        and lists. Other data types are returned unmodified.

        Args:
            data (Any): The input data structure (string, dict, list, or other).
            context (Dict[str, Any]): A dictionary containing contextual information
                                      for the processing pipeline. This node does not
                                      directly use the context, but it is passed
                                      along for consistency with BaseNode.

        Returns:
            Any: The data with PII redacted. The return type will match the input type.
        """
        logger.info(f"[{self.node_name}] Starting PII redaction process for data of type: {type(data).__name__}.")

        if isinstance(data, str):
            # Process individual strings
            return self._redact_string(data)
        
        elif isinstance(data, dict):
            # Recursively process values in dictionaries
            redacted_dict = {}
            for key, value in data.items():
                redacted_dict[key] = self.process(value, context) # Recursive call
            logger.debug(f"[{self.node_name}] Processed dictionary data.")
            return redacted_dict
        
        elif isinstance(data, list):
            # Recursively process items in lists
            redacted_list = [self.process(item, context) for item in data] # Recursive call
            logger.debug(f"[{self.node_name}] Processed list data.")
            return redacted_list
        
        else:
            # For any other data type, return it as is.
            # This makes the node robust against unexpected inputs.
            logger.debug(
                f"[{self.node_name}] Data type {type(data).__name__} is not a string, dict, or list. "
                "Returning data as is without PII redaction."
            )
            return data