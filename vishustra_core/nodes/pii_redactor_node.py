import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node designed to identify and redact Personally Identifiable Information (PII)
    from textual data. It uses regular expressions to find common PII patterns and replaces them
    with a configurable redaction string.
    """

    DEFAULT_PII_PATTERNS: Dict[str, str] = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone_us": r'\b(?:\d{3}[-.\s]?|\(\d{3}\)\s?)\d{3}[-.\s]?\d{4}\b',
        "ssn_us": r'\b\d{3}-\d{2}-\d{4}\b', # Basic U.S. SSN format
        "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        # Note: Name detection is highly complex and typically requires NLP.
        # Simple regex for names is prone to high false positives/negatives.
    }
    DEFAULT_REDACTION_STRING: str = "[REDACTED]"

    def __init__(self, patterns: Union[List[str], Dict[str, str], None] = None,
                 redaction_string: str = DEFAULT_REDACTION_STRING):
        """
        Initializes the PIIRedactorNode.

        Args:
            patterns: A list of regex strings or a dictionary mapping pattern names to regex strings
                      to use for PII detection. If None, default patterns will be used.
                      Example: ["\\bEmail: (.*?)\\b", "\\bPhone: (.*?)\\b"]
                      Example: {"custom_id": "\\bID:\\s*([0-9A-Za-z-]+)\\b"}
            redaction_string: The string to replace identified PII with.
        """
        self._redaction_string = redaction_string
        
        compiled_patterns = []
        if patterns is None:
            for name, pattern_str in self.DEFAULT_PII_PATTERNS.items():
                try:
                    compiled_patterns.append(re.compile(pattern_str))
                except re.error as e:
                    logger.error(f"Failed to compile default PII pattern '{name}': {e}")
        elif isinstance(patterns, list):
            for pattern_str in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern_str))
                except re.error as e:
                    logger.error(f"Failed to compile custom list PII pattern '{pattern_str}': {e}")
        elif isinstance(patterns, dict):
            for name, pattern_str in patterns.items():
                try:
                    compiled_patterns.append(re.compile(pattern_str))
                except re.error as e:
                    logger.error(f"Failed to compile custom dict PII pattern '{name}': {e}")
        else:
            logger.warning(
                "Invalid 'patterns' type provided. Expected list, dict, or None. "
                "Using default PII patterns."
            )
            for name, pattern_str in self.DEFAULT_PII_PATTERNS.items():
                try:
                    compiled_patterns.append(re.compile(pattern_str))
                except re.error as e:
                    logger.error(f"Failed to compile default PII pattern '{name}': {e}")

        self._compiled_patterns = compiled_patterns
        logger.info(f"{self.node_name} initialized with {len(self._compiled_patterns)} PII patterns.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PIIRedactorNode"

    def _redact_text(self, text: str) -> str:
        """Applies all compiled PII patterns to redact the given text."""
        redacted_text = text
        for pattern in self._compiled_patterns:
            matches_found = 0
            # Use a lambda function for replacement to ensure the exact matched string is replaced
            redacted_text, count = pattern.subn(lambda m: self._redaction_string, redacted_text)
            matches_found += count
            if matches_found > 0:
                logger.debug(f"Redacted {matches_found} occurrences using pattern: {pattern.pattern}")
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        If the input `data` is a string, it will be directly redacted.
        If `data` is a dictionary, string values within the top level of the dictionary
        will be redacted. Other data types (lists, nested dicts, numbers) will be passed through
        unchanged.

        Args:
            data: The input data to process. Can be a string or a dictionary.
            context: A dictionary containing contextual information for processing.
                     (Not directly used by this node, but required by BaseNode interface).

        Returns:
            The data with PII redacted. The return type matches the input type where applicable.

        Raises:
            TypeError: If the input data is not a string or a dictionary.
        """
        if isinstance(data, str):
            logger.debug(f"{self.node_name} processing string data.")
            return self._redact_text(data)
        elif isinstance(data, dict):
            logger.debug(f"{self.node_name} processing dictionary data.")
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    processed_data[key] = self._redact_text(value)
                else:
                    # For non-string values, pass them through unchanged for now.
                    # Complex data structures would require recursive processing which is
                    # outside the scope of a basic single node implementation for simplicity.
                    processed_data[key] = value
            return processed_data
        else:
            logger.warning(
                f"{self.node_name} received unsupported data type: {type(data).__name__}. "
                "Data will be returned unchanged."
            )
            return data

# Example of basic usage (for local testing, not part of Vishustra framework execution)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Set to DEBUG for more detailed logging
    
    # Test with default patterns
    redactor_default = PIIRedactorNode()
    
    test_string_1 = "My email is test@example.com and my phone is 123-456-7890. SSN: 999-88-7777."
    redacted_string_1 = redactor_default.process(test_string_1, {})
    logger.info(f"Original 1: {test_string_1}")
    logger.info(f"Redacted 1: {redacted_string_1}")
    # Expected: My email is [REDACTED] and my phone is [REDACTED]. SSN: [REDACTED].

    test_dict_1 = {
        "user_message": "Please contact me at john.doe@mail.org or call (555) 123-4567.",
        "log_id": "abc-123-xyz",
        "timestamp": "2023-10-27T10:00:00Z"
    }
    redacted_dict_1 = redactor_default.process(test_dict_1, {})
    logger.info(f"Original 2: {test_dict_1}")
    logger.info(f"Redacted 2: {redacted_dict_1}")
    # Expected: 'user_message' value redacted.

    # Test with custom patterns and redaction string
    custom_patterns = {
        "user_id": r'\bUSER_[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\b',
        "product_code": r'\bPCODE-[A-Z]{3}-\d{4}\b'
    }
    redactor_custom = PIIRedactorNode(patterns=custom_patterns, redaction_string="[CENSORED]")

    test_string_2 = "User ID USER_a1b2c3d4-e5f6-7890-1234-567890abcdef accessed PCODE-XYZ-9876. Not PII."
    redacted_string_2 = redactor_custom.process(test_string_2, {})
    logger.info(f"Original 3: {test_string_2}")
    logger.info(f"Redacted 3: {redacted_string_2}")
    # Expected: 'USER_...' and 'PCODE-...' redacted with [CENSORED].

    # Test unsupported data type
    unsupported_data = ["item1", "item2"]
    result_unsupported = redactor_default.process(unsupported_data, {})
    logger.info(f"Original 4 (unsupported): {unsupported_data}")
    logger.info(f"Processed 4 (unsupported): {result_unsupported}")
    # Expected: Warning logged, data returned unchanged.

    test_with_no_pii = "This is a safe message with no sensitive information."
    redacted_no_pii = redactor_default.process(test_with_no_pii, {})
    logger.info(f"Original 5: {test_with_no_pii}")
    logger.info(f"Redacted 5: {redacted_no_pii}")
    # Expected: Original message returned, no changes.
