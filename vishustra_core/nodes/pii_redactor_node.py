import re
import logging
from typing import Any, Dict, List, Pattern, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts personally identifiable information (PII)
    from text data using configurable regular expressions.

    This node is designed to sanitize data streams by replacing matched PII patterns
    with specified mask strings. It supports processing single strings or lists of strings.
    """

    # Default common PII patterns and their replacement masks.
    # Users can override or extend these via the constructor.
    # Note: Regex patterns for PII can be complex and context-dependent.
    # These defaults cover common cases but may require tuning for specific data sets.
    _DEFAULT_PII_PATTERNS: List[Dict[str, str]] = [
        {"regex": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "mask": "[EMAIL_REDACTED]"},
        {"regex": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "mask": "[PHONE_REDACTED]"},
        {"regex": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "mask": "[SSN_REDACTED]"},
        # Common credit card number patterns (Visa, Mastercard, Amex, Discover).
        # This regex is highly specific and doesn't validate checksums.
        {"regex": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3(?:4|7)[0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})\b", "mask": "[CREDIT_CARD_REDACTED]"},
    ]

    def __init__(self, pii_patterns: Union[List[Dict[str, str]], None] = None):
        """
        Initializes the PIIRedactorNode with a set of PII patterns.

        Args:
            pii_patterns: An optional list of dictionaries. Each dictionary must
                          contain 'regex' (the regex string) and 'mask' (the
                          replacement string). If None, a default set of common PII
                          patterns will be used.
                          Example: [{'regex': r'\\b\\d{5}\\b', 'mask': '[ZIP_REDACTED]'}]
        """
        self._compiled_patterns: List[Dict[str, Union[Pattern[str], str]]] = []
        patterns_to_use = pii_patterns if pii_patterns is not None else self._DEFAULT_PII_PATTERNS

        for idx, pattern_config in enumerate(patterns_to_use):
            if not isinstance(pattern_config, dict) or 'regex' not in pattern_config or 'mask' not in pattern_config:
                logger.warning(
                    f"PIIRedactorNode: Invalid pattern configuration at index {idx}. "
                    "Each pattern must be a dictionary with 'regex' (string) and 'mask' (string) keys. Skipping."
                )
                continue
            try:
                compiled_regex = re.compile(pattern_config['regex'], re.IGNORECASE)
                self._compiled_patterns.append({
                    "compiled_regex": compiled_regex,
                    "mask": pattern_config['mask']
                })
                logger.debug(f"PIIRedactorNode: Successfully compiled regex pattern: '{pattern_config['regex']}'")
            except re.error as e:
                logger.error(f"PIIRedactorNode: Failed to compile regex pattern '{pattern_config['regex']}': {e}")
            except Exception as e:
                logger.error(f"PIIRedactorNode: An unexpected error occurred during pattern compilation for '{pattern_config.get('regex', 'N/A')}': {e}")

        if not self._compiled_patterns:
            logger.warning("PIIRedactorNode: No valid PII patterns were successfully loaded. This node will not perform any redaction.")

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "PIIRedactorNode"

    def _redact_single_string(self, text: str) -> str:
        """Helper to apply all compiled PII patterns to a single string."""
        redacted_text = text
        for pattern_config in self._compiled_patterns:
            compiled_regex = pattern_config["compiled_regex"]
            mask = pattern_config["mask"]
            redacted_text = compiled_regex.sub(mask, redacted_text)
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII based on configured patterns.

        The `data` input can be a single string or a list of strings.
        - If `data` is a `str`, it redacts PII directly from the string.
        - If `data` is a `list`, it iterates through the list, redacting PII
          from each `str` element. Non-string elements in the list are preserved.
        - For any other data type, the input data is returned as-is, with a warning.

        Args:
            data: The input data, expected to be a string or a list of strings.
            context: A dictionary containing contextual information for processing.
                     This node currently does not utilize the context dictionary.

        Returns:
            The data with PII redacted. The return type mirrors the input type:
            `str` if input was `str`, `list[Any]` if input was `list`.
            For unsupported input types, the original data is returned.
        """
        if not self._compiled_patterns:
            logger.warning(f"{self.node_name}: No valid PII patterns configured. Returning data without redaction.")
            return data

        try:
            if isinstance(data, str):
                logger.debug(f"{self.node_name}: Redacting PII from a single string.")
                return self._redact_single_string(data)
            elif isinstance(data, list):
                logger.debug(f"{self.node_name}: Redacting PII from a list of items.")
                redacted_list = []
                for idx, item in enumerate(data):
                    if isinstance(item, str):
                        redacted_list.append(self._redact_single_string(item))
                    else:
                        redacted_list.append(item) # Preserve non-string items in the list
                        logger.warning(
                            f"{self.node_name}: Received a non-string item at index {idx} in a list "
                            f"(type: {type(item).__name__}). It will be included in the output unredacted."
                        )
                return redacted_list
            else:
                logger.warning(
                    f"{self.node_name}: Unsupported data type received (type: {type(data).__name__}). "
                    "Expected str or list[str]. Returning data unredacted."
                )
                return data
        except Exception as e:
            logger.error(
                f"{self.node_name}: An unexpected error occurred during PII redaction. "
                f"Returning original data. Error: {e}",
                exc_info=True
            )
            # In case of an unhandled error, it's safer to return the original data
            # rather than crashing the pipeline or returning malformed data.
            return data