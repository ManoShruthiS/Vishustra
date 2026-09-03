import logging
import re
from typing import Any, Dict, List, Pattern, Union

# Assuming BaseNode is available at this path within Vishustra
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node designed to redact Personal Identifiable Information (PII)
    from text data using configurable regular expressions.

    This node can redact common PII types such as emails, phone numbers, and social
    security numbers by default. It supports custom regex patterns and a customizable
    redaction mask. When processing dictionary inputs, it can be configured to target
    specific keys.
    """

    DEFAULT_REDACTION_PATTERNS: Dict[str, Union[str, Pattern[str]]] = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"(\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b|\b\(\d{3}\)\s*\d{3}[-.\s]??\d{4}\b|\b\d{10}\b)",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }
    DEFAULT_REDACTION_MASK: str = "[REDACTED_PII]"

    def __init__(
        self,
        patterns: Dict[str, Union[str, Pattern[str]]] = None,
        mask: str = None,
        keys_to_redact: List[str] = None
    ):
        """
        Initializes the PIIRedactorNode with optional custom patterns and mask.

        Args:
            patterns (Dict[str, Union[str, Pattern[str]]], optional):
                A dictionary where keys are descriptive names (e.g., "email") and values
                are regex patterns (string or compiled `re.Pattern` objects) for PII types.
                If provided, these patterns will extend or override the default set.
            mask (str, optional): The string used to replace redacted PII.
                Defaults to "[REDACTED_PII]".
            keys_to_redact (List[str], optional):
                If the input `data` is a dictionary, only the values associated with these
                keys will be processed for PII. If `None`, all string values within the
                dictionary will be processed.
        """
        self._compiled_patterns: Dict[str, Pattern[str]] = {}
        # Merge default patterns with user-provided patterns, prioritizing user patterns
        all_patterns = self.DEFAULT_REDACTION_PATTERNS.copy()
        if patterns:
            all_patterns.update(patterns)

        for name, pattern in all_patterns.items():
            try:
                if isinstance(pattern, str):
                    self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                elif isinstance(pattern, Pattern):
                    self._compiled_patterns[name] = pattern
                else:
                    logger.warning(
                        "PIIRedactorNode: Pattern for type '%s' is not a string or compiled regex. Skipping.",
                        name
                    )
            except re.error as e:
                logger.error(
                    "PIIRedactorNode: Failed to compile regex pattern '%s' for type '%s': %s",
                    pattern, name, e
                )

        self._mask = mask if mask is not None else self.DEFAULT_REDACTION_MASK
        self._keys_to_redact = keys_to_redact
        logger.info(
            "PIIRedactorNode initialized with %d redaction patterns and mask '%s'. "
            "Targeted keys for dict processing: %s",
            len(self._compiled_patterns), self._mask,
            "all string values" if self._keys_to_redact is None else self._keys_to_redact
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor"

    def _redact_string(self, text: str) -> str:
        """Applies all configured PII patterns to a given string."""
        redacted_text = text
        for pii_type, pattern in self._compiled_patterns.items():
            # Use a temporary variable to check if any substitution occurred for logging
            temp_redacted_text = pattern.sub(self._mask, redacted_text)
            if temp_redacted_text != redacted_text:
                logger.debug("PII Redactor: Redacted PII of type '%s' in string.", pii_type)
                redacted_text = temp_redacted_text
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        If `data` is a string, it applies all configured redaction patterns directly.
        If `data` is a dictionary, it iterates through its string values and redacts them.
        The `keys_to_redact` configuration from `__init__` determines which dictionary
        keys are processed. Other data types are returned unchanged, with a warning logged.

        Args:
            data (Any): The input data to be processed for PII. Expected to be a string or a dictionary.
            context (Dict[str, Any]): A dictionary containing contextual information for the
                                       current processing flow. (Currently not used for
                                       dynamic configuration; configuration is via `__init__`).

        Returns:
            Any: The data with PII redacted, or the original data if its type is not supported
                 for redaction.
        """
        if context:
            logger.debug("PII Redactor: Context received in process method, but node configuration "
                         "is primarily handled during initialization.")

        if isinstance(data, str):
            logger.debug("PII Redactor: Processing string data for PII redaction.")
            return self._redact_string(data)
        elif isinstance(data, dict):
            logger.debug("PII Redactor: Processing dictionary data for PII redaction.")
            processed_data = data.copy() # Operate on a copy to avoid modifying the original input
            for key, value in processed_data.items():
                if isinstance(value, str):
                    if self._keys_to_redact is None or key in self._keys_to_redact:
                        processed_data[key] = self._redact_string(value)
                    else:
                        logger.debug(
                            "PII Redactor: Skipping redaction for key '%s' as it's not in the configured "
                            "list of keys to redact.", key
                        )
                elif isinstance(value, (dict, list)):
                    logger.debug(
                        "PII Redactor: Skipping nested dict/list for key '%s'. Recursive redaction "
                        "is not implemented in this version.", key
                    )
            return processed_data
        else:
            logger.warning(
                "PII Redactor: Input data type '%s' is not supported for PII redaction. "
                "Returning original data without processing.", type(data).__name__
            )
            return data