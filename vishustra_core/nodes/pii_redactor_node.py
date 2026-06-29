import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node designed to identify and redact Personally Identifiable Information (PII)
    from incoming data. It supports redaction within strings, and recursively within
    dictionaries and lists containing string values.

    The node comes with default patterns for common PII like emails and phone numbers,
    which can be extended or overridden at initialization or via the processing context.
    """

    DEFAULT_REDACTION_TOKEN: str = "[REDACTED]"
    DEFAULT_PII_PATTERNS: Dict[str, str] = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        # Additional common PII patterns can be added here, e.g.:
        # "credit_card": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})\b',
        # "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
        # "social_security_number": r'\b\d{3}-\d{2}-\d{4}\b' # US specific
    }

    def __init__(self, patterns: Dict[str, str] = None, redaction_token: str = None):
        """
        Initializes the PII Redactor Node with custom patterns and a redaction token.

        Args:
            patterns (Dict[str, str], optional): A dictionary of regular expression patterns
                                                 for PII detection. Keys are descriptive names,
                                                 values are regex strings. If provided, these
                                                 patterns are merged with (and can override)
                                                 the default patterns.
            redaction_token (str, optional): The string to replace identified PII with.
                                            Defaults to '[REDACTED]'.
        """
        self._patterns = {**self.DEFAULT_PII_PATTERNS, **(patterns or {})}
        self._redaction_token = redaction_token if redaction_token is not None else self.DEFAULT_REDACTION_TOKEN
        
        logger.debug(f"PIIRedactorNode initialized with patterns: {list(self._patterns.keys())} "
                     f"and redaction token: '{self._redaction_token}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII Redactor"

    def _redact_string(self, text: str, patterns: Dict[str, str], redaction_token: str) -> str:
        """
        Applies all defined PII patterns to a string and redacts matches.
        """
        redacted_text = text
        for name, pattern_str in patterns.items():
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                # Only log if a redaction actually happened
                if pattern.search(redacted_text):
                    logger.debug(f"Applying PII pattern '{name}' to string data.")
                    redacted_text = pattern.sub(redaction_token, redacted_text)
            except re.error as e:
                logger.error(f"Invalid regex pattern for '{name}': '{pattern_str}'. Error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during redaction for pattern '{name}': {e}", exc_info=True)
        return redacted_text

    def _redact_recursive(self, data: Any, patterns: Dict[str, str], redaction_token: str) -> Any:
        """
        Recursively redacts PII within dictionaries, lists, and strings.
        """
        if isinstance(data, str):
            return self._redact_string(data, patterns, redaction_token)
        elif isinstance(data, dict):
            return {k: self._redact_recursive(v, patterns, redaction_token) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._redact_recursive(item, patterns, redaction_token) for item in data]
        else:
            return data

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to identify and redact PII.

        The `context` dictionary can be used to dynamically adjust redaction behavior
        for a specific invocation:
        - `context['patterns']` (Dict[str, str]): A dictionary of regex patterns that
          will be merged with the node's configured patterns for this run.
        - `context['redaction_token']` (str): Overrides the default or configured
          redaction token for this run.

        Args:
            data (Any): The input data to be processed for PII. This can be a string,
                        a dictionary, a list, or other primitive types.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing, potentially including
                                       runtime overrides for patterns and token.

        Returns:
            Any: The processed data with identified PII redacted. The structure of the
                 data remains the same.
        """
        current_patterns = {**self._patterns, **context.get("patterns", {})}
        current_redaction_token = context.get("redaction_token", self._redaction_token)

        logger.info(f"PIIRedactorNode starting process for data type: {type(data).__name__}. "
                    f"Using {len(current_patterns)} patterns and token '{current_redaction_token}'.")

        if not current_patterns:
            logger.warning("No PII patterns configured or provided. Redaction will not occur.")
            return data

        try:
            redacted_data = self._redact_recursive(data, current_patterns, current_redaction_token)
            logger.info("PIIRedactorNode successfully completed PII redaction.")
            return redacted_data
        except Exception as e:
            logger.exception(f"An unhandled error occurred during PII redaction in process method: {e}")
            # Depending on the system's error handling policy, we might:
            # 1. Re-raise the exception to halt processing (current behavior).
            # 2. Return the original unredacted data (might expose PII).
            # 3. Return a specific error object or indicator.
            # For a critical data security component, re-raising is often preferred
            # to prevent potential PII leakage if redaction fails silently.
            raise