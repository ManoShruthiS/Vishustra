import logging
import re
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A Vishustra processing node that redacts common Personally Identifiable Information (PII)
    from text data using regular expressions.

    This node is designed to identify and replace sensitive data patterns, ensuring
    privacy compliance for information flowing through the orchestration framework.

    Supported PII types for redaction (configurable via `redact_patterns` in context):
    - Email addresses: Standard email formats.
    - US Phone numbers: Various common US phone number formats.
    - US Social Security Numbers (SSN): Standard XXX-XX-XXXX format.
    - Credit Card numbers: Basic 16-digit patterns (with or without hyphens/spaces).
    - IP Addresses: Standard IPv4 formats.
    """

    # Class-level definition of PII patterns. This design centralizes pattern management
    # and allows for efficient compilation and reuse across instances.
    _PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone_us": r'\b(?:\+?1[\s-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b',
        "ssn_us": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b(?:\d{4}[- ]){3}\d{4}|\d{16}\b',
        "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII Redactor"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII based on configured patterns.

        The method expects text data (`str`) and will return the redacted text.
        If the input `data` is not a string, a warning will be logged, and the
        original data will be returned unmodified, ensuring non-textual data
        is not erroneously processed.

        Configuration options passed via the `context` dictionary:
        - `redaction_token` (str, optional): The string used to replace identified PII.
          Defaults to "[REDACTED]".
        - `redact_patterns` (list[str], optional): A list of keys from the `_PII_PATTERNS`
          dictionary specifying which types of PII should be targeted for redaction.
          If this key is omitted or an empty list is provided, all defined PII patterns
          will be applied. Example: `['email', 'phone_us']`.
        """
        if not isinstance(data, str):
            logger.warning(
                f"[{self.node_name}] Received non-string data (type: '{type(data).__name__}'). "
                "PII redaction skipped. Returning data as is."
            )
            return data

        redacted_text = data
        redaction_token = context.get('redaction_token', "[REDACTED]")
        
        # Determine which specific PII patterns to apply based on context.
        # If 'redact_patterns' is not specified or empty, all known patterns are used.
        patterns_to_use_keys = context.get('redact_patterns', list(self._PII_PATTERNS.keys()))
        
        applied_redactions_log = []

        for pii_type_key in patterns_to_use_keys:
            pattern_str = self._PII_PATTERNS.get(pii_type_key)
            if pattern_str:
                try:
                    # Compile regex for efficiency if not already compiled (re.compile caches)
                    pattern = re.compile(pattern_str)
                    
                    # Track if any changes occur for logging purposes
                    initial_length = len(redacted_text)
                    redacted_text = pattern.sub(redaction_token, redacted_text)
                    
                    if len(redacted_text) != initial_length or re.search(pattern, data):
                        # Log only if actual redaction happened or pattern was found in original
                        applied_redactions_log.append(pii_type_key)
                        logger.debug(
                            f"[{self.node_name}] Redacted occurrences of '{pii_type_key}'. "
                            f"Using token: '{redaction_token}'."
                        )
                except re.error as e:
                    logger.error(
                        f"[{self.node_name}] Failed to apply regex pattern for '{pii_type_key}': {e}. "
                        "Skipping this pattern."
                    )
            else:
                logger.warning(
                    f"[{self.node_name}] Requested PII type '{pii_type_key}' not defined "
                    "in available patterns. Skipping redaction for this type."
                )
        
        if applied_redactions_log:
            logger.info(
                f"[{self.node_name}] Successfully redacted PII types: "
                f"{', '.join(sorted(set(applied_redactions_log)))}. "
                f"Original text length: {len(data)}, Redacted text length: {len(redacted_text)}."
            )
        else:
            logger.info(f"[{self.node_name}] No specified PII types found or redacted in the data.")

        return redacted_text