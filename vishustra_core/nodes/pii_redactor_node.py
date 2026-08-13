import re
import logging
from typing import Any, Dict, List, Pattern

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node designed to redact Personally Identifiable Information (PII)
    from text data. It supports processing strings, dictionaries, and lists containing strings,
    and uses regular expressions for pattern matching.

    Redaction rules are configurable upon initialization, allowing flexibility for
    various PII types and compliance requirements.
    """

    def __init__(self,
                 redaction_rules: Dict[str, str] = None,
                 redaction_placeholder: str = "[REDACTED]",
                 log_redacted_matches: bool = False):
        """
        Initializes the PII Redactor node with custom redaction rules, a placeholder,
        and logging behavior.

        Args:
            redaction_rules (Dict[str, str], optional): A dictionary where keys are
                descriptive names (e.g., "email", "phone") and values are
                regular expression strings. If None, a set of default rules is used.
            redaction_placeholder (str, optional): The string used to replace
                redacted PII. Defaults to "[REDACTED]".
            log_redacted_matches (bool, optional): If True, logs the actual PII
                matches before redaction at DEBUG level. Use with caution due to
                potential PII leakage in logs. Defaults to False.
        """
        self._redaction_placeholder = redaction_placeholder
        self._log_redacted_matches = log_redacted_matches

        # Default comprehensive PII regex patterns
        default_rules = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone_number": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{4}\b',
            "ssn": r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "credit_card": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})\b', # Common CC patterns
            "url": r'\b(?:https?://|www\.)[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\S*)\b',
            "date_of_birth": r'\b(?:(?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12][0-9]|3[01])[-/.](?:19|20)?\d{2})|(?:(?:0?[1-9]|[12][0-9]|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)?\d{2})\b' # MM/DD/YYYY or DD/MM/YYYY
        }
        self._redaction_rules: Dict[str, Pattern] = {}
        rules_to_compile = redaction_rules if redaction_rules is not None else default_rules

        for name, pattern_str in rules_to_compile.items():
            try:
                self._redaction_rules[name] = re.compile(pattern_str, re.IGNORECASE)
            except re.error as e:
                logger.error(
                    f"[{self.node_name}] Failed to compile regex rule '{name}': '{pattern_str}'. Error: {e}. This rule will be skipped."
                )
            except TypeError as e:
                logger.error(
                    f"[{self.node_name}] Invalid type for regex pattern '{name}': '{pattern_str}'. Error: {e}. This rule will be skipped."
                )

        if not self._redaction_rules:
            logger.warning(f"[{self.node_name}] No valid redaction rules were loaded. Node will not redact any PII.")
        else:
            logger.info(f"[{self.node_name}] Initialized with {len(self._redaction_rules)} redaction rules: {list(self._redaction_rules.keys())}.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PIIRedactor"

    def _redact_string(self, text: str) -> str:
        """
        Applies all configured regex rules to redact PII from a given string.
        """
        redacted_text = text
        for rule_name, pattern in self._redaction_rules.items():
            try:
                # Find all matches before substituting to log them if enabled
                if self._log_redacted_matches and logger.isEnabledFor(logging.DEBUG):
                    matches = pattern.findall(redacted_text)
                    if matches:
                        logger.debug(f"[{self.node_name}] Redacting '{rule_name}' matches: {matches[:5]}... from text snippet: '{redacted_text[:100]}...'")
                redacted_text = pattern.sub(self._redaction_placeholder, redacted_text)
            except Exception as e:
                logger.warning(
                    f"[{self.node_name}] Error applying redaction rule '{rule_name}': {e}. "
                    "Skipping this rule for the current text segment."
                )
        return redacted_text

    def _traverse_and_redact(self, data: Any) -> Any:
        """
        Recursively traverses data structures (dictionaries, lists) and redacts
        string values. Other data types are returned as-is.
        """
        if isinstance(data, str):
            return self._redact_string(data)
        elif isinstance(data, dict):
            return {k: self._traverse_and_redact(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._traverse_and_redact(item) for item in data]
        else:
            # For other types (int, float, bool, None), return as is
            return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII based on configured rules.

        This method expects `data` to be a string, a dictionary, or a list.
        It recursively redacts string values found within these structures.

        Args:
            data (Any): The input data to be processed. Expected types are
                        `str`, `dict`, or `list` of `str`/`dict`.
            context (Dict[str, Any]): A dictionary containing context information
                                     for the processing chain. Not directly used
                                     for PII rules in this implementation, but
                                     available for future extensions (e.g., dynamic rule loading).

        Returns:
            Any: The processed data with PII redacted. The structure of the
                 data is preserved.

        Raises:
            TypeError: If the input `data` is of an unsupported type that cannot
                       be traversed for redaction.
            Exception: For unexpected errors during the redaction process.
        """
        logger.info(f"[{self.node_name}] Starting PII redaction process.")
        logger.debug(f"[{self.node_name}] Input data type: {type(data)}. Context keys: {list(context.keys()) if context else 'None'}")

        if not self._redaction_rules:
            logger.warning(f"[{self.node_name}] No redaction rules are active. Returning data as-is.")
            return data

        if not isinstance(data, (str, dict, list)):
            logger.error(
                f"[{self.node_name}] Unsupported data type for PII redaction: {type(data)}. "
                "Expected str, dict, or list. Data will not be processed."
            )
            raise TypeError(
                f"PIIRedactorNode received unsupported data type: {type(data)}. "
                "Expected str, dict, or list for redaction."
            )

        try:
            processed_data = self._traverse_and_redact(data)
            logger.info(f"[{self.node_name}] PII redaction completed successfully.")
            return processed_data
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during PII redaction.")
            # Depending on the desired error handling strategy, could re-raise,
            # return original data, or return a partial result. Re-raising
            # for robustness in a processing chain.
            raise