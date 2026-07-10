
import logging
import re
from typing import Any, Dict, List, Union

# Assuming BaseNode resides in vishustra_core.nodes.base_node as per project structure
from vishustra_core.nodes.base_node import BaseNode 

logger = logging.getLogger(__name__)

class PII_RedactorNode(BaseNode):
    """
    A Vishustra processing node designed to identify and redact Personally Identifiable Information (PII)
    from input data. This node supports redaction within strings, and recursively within
    dictionary values and list elements.

    It utilizes configurable regex patterns to detect PII and replaces it with a specified
    placeholder string.
    """

    def __init__(self, default_replacement_string: str = "[REDACTED]") -> None:
        """
        Initializes the PII Redactor Node with a default replacement string and a set of
        common PII regex patterns. These defaults can be overridden or extended via the
        'context' dictionary during processing.

        Args:
            default_replacement_string: The string to use as a placeholder for redacted PII.
                                        Defaults to "[REDACTED]".
        
        Raises:
            TypeError: If `default_replacement_string` is not a string.
        """
        if not isinstance(default_replacement_string, str):
            logger.error(f"Initialization error: default_replacement_string must be a string, got {type(default_replacement_string)}")
            raise TypeError("Default replacement string must be a string.")
            
        self._default_replacement: str = default_replacement_string
        
        # Default PII patterns. These are illustrative and can be expanded or refined.
        # In a production system, these might be loaded from a configuration service.
        self._default_patterns: Dict[str, str] = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone_number_us": r"\b(?:\+1[-\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "credit_card_simple": r"\b(?:\d[ -]*?){13,16}\b", # Basic pattern, requires enhancement for robustness
            # Add more patterns like Social Security Numbers, IP addresses, names (more complex), etc.
        }
        logger.debug("PII_RedactorNode initialized with default patterns and replacement string.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "PII_Redactor"

    def _redact_string(self, text: str, patterns: Dict[str, str], replacement: str) -> str:
        """
        Applies a set of regex patterns to a single string to redact identified PII.

        Args:
            text: The string content to process for PII.
            patterns: A dictionary where keys are PII types (e.g., "email") and values
                      are their corresponding regex pattern strings.
            replacement: The string to substitute for each instance of identified PII.
            
        Returns:
            The string with all detected PII replaced by the `replacement` string.
        """
        redacted_text = text
        for pii_type, pattern_str in patterns.items():
            try:
                # Ensure the pattern is a string before compiling/using
                if not isinstance(pattern_str, str):
                    logger.warning(
                        f"Pattern for PII type '{pii_type}' is not a string ({type(pattern_str)}). Skipping."
                    )
                    continue
                redacted_text = re.sub(pattern_str, replacement, redacted_text, flags=re.IGNORECASE)
                logger.debug(f"Applied redaction pattern for '{pii_type}'.")
            except re.error as e:
                logger.error(
                    f"Failed to apply regex pattern for PII type '{pii_type}' ('{pattern_str}'). "
                    f"Invalid regex: {e}"
                )
                # Continue processing with other patterns even if one fails
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred while applying pattern '{pii_type}' ('{pattern_str}'): {e}"
                )
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input `data` to identify and redact Personally Identifiable Information (PII).
        
        This method supports arbitrary nesting within dictionaries and lists, recursively applying
        redaction to any string values found within these structures. Data types that are not
        strings, dictionaries, or lists (e.g., integers, booleans, None) are passed through
        unchanged.

        The 'context' dictionary can be used to dynamically configure the redaction behavior:
        - `context["redaction_patterns"]`: A dictionary (`Dict[str, str]`) mapping PII type names
          to regex pattern strings. These patterns will be merged with, and take precedence over,
          the node's default patterns.
        - `context["replacement_string"]`: A string to use as the placeholder for redacted PII.
          This overrides the node's `default_replacement_string`.
        
        Args:
            data: The input data, which can be a string, a dictionary, a list, or any other type.
                  Nested dictionaries and lists are supported for recursive redaction.
            context: A dictionary containing additional processing parameters for this specific
                     invocation, such as custom redaction patterns or a replacement string.
        
        Returns:
            The `data` with PII redacted, maintaining the original structure.
            
        Raises:
            Exception: Propagates any unexpected errors encountered during the redaction process
                       to signal a processing failure.
        """
        
        replacement_string = context.get("replacement_string", self._default_replacement)
        
        # Consolidate active PII patterns: start with defaults, then apply context overrides
        active_patterns = self._default_patterns.copy()
        if "redaction_patterns" in context:
            if isinstance(context["redaction_patterns"], dict):
                active_patterns.update(context["redaction_patterns"])
                logger.debug(f"Custom redaction patterns applied from context. Total active patterns: {len(active_patterns)}")
            else:
                logger.warning(
                    f"Context 'redaction_patterns' must be a dictionary, but received {type(context['redaction_patterns'])}. "
                    "Only default patterns will be used."
                )

        def _recursive_redact(item: Any) -> Any:
            """
            Internal helper function to recursively traverse data structures and redact PII.
            """
            if isinstance(item, str):
                return self._redact_string(item, active_patterns, replacement_string)
            elif isinstance(item, dict):
                # Recursively process values in a dictionary
                return {k: _recursive_redact(v) for k, v in item.items()}
            elif isinstance(item, list):
                # Recursively process elements in a list
                return [_recursive_redact(elem) for elem in item]
            else:
                # For any other data types (int, float, bool, None, etc.), return as is
                return item

        logger.info(f"Starting PII redaction for input data of type: {type(data).__name__}")
        try:
            redacted_data = _recursive_redact(data)
            logger.info("PII redaction completed successfully.")
            return redacted_data
        except Exception as e:
            logger.exception(f"An unexpected error occurred during PII redaction: {e}")
            # Re-raise the exception to signal a critical failure in this node's processing step.
            raise

