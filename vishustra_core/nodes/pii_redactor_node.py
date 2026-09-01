import logging
import re
from typing import Any, Dict, Union, List
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class PIIRedactorNode(BaseNode):
    """
    A processing node that redacts Personally Identifiable Information (PII)
    from input data based on predefined patterns or patterns provided in the context.

    This node supports redaction from strings, dictionaries, and lists,
    handling nested structures recursively.
    """

    _DEFAULT_PII_PATTERNS: Dict[str, str] = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone_number_us": r"\b(?:\+?1[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?:/\d{1,2})?\b",
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b", # Simple, prone to false positives without checksum
        "ssn_us": r"\b\d{3}-\d{2}-\d{4}\b", # Specific format
    }
    _DEFAULT_REDACTION_PLACEHOLDER: str = "[REDACTED_PII]"

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "PII Redactor"

    def _redact_string(self, text: str, patterns: Dict[str, str], placeholder_base: str) -> str:
        """
        Applies PII redaction patterns to a single string.
        """
        redacted_text = text
        for pii_type, pattern_str in patterns.items():
            try:
                # Compile pattern only once if performance is critical,
                # but for simplicity and dynamic patterns, compiling here is fine.
                pattern = re.compile(pattern_str, re.IGNORECASE)
                redacted_text = pattern.sub(f"{placeholder_base}_{pii_type.upper()}]", redacted_text)
            except re.error as e:
                logger.error(
                    f"[{self.node_name}] Invalid regex pattern for '{pii_type}': '{pattern_str}'. Error: {e}"
                )
                continue
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] Unexpected error during redaction for '{pii_type}': {e}"
                )
                continue
        return redacted_text

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to redact PII.

        The `context` dictionary can override:
        - `redaction_patterns` (Dict[str, str]): A dictionary mapping PII type names
          to their regex patterns. Defaults to `_DEFAULT_PII_PATTERNS`.
        - `redaction_placeholder` (str): The base string used for redaction
          (e.g., "[REDACTED_PII"). The PII type will be appended (e.g., "[REDACTED_PII_EMAIL]").
          Defaults to `_DEFAULT_REDACTION_PLACEHOLDER`.

        Args:
            data: The input data, which can be a string, dict, list, or other types.
            context: A dictionary containing runtime context and configuration.

        Returns:
            The data with identified PII redacted.
        """
        redaction_patterns = context.get("redaction_patterns", self._DEFAULT_PII_PATTERNS)
        redaction_placeholder_base = context.get("redaction_placeholder", self._DEFAULT_REDACTION_PLACEHOLDER)

        logger.debug(
            f"[{self.node_name}] Starting PII redaction with patterns: {list(redaction_patterns.keys())}"
        )

        try:
            return self._recursive_redact(data, redaction_patterns, redaction_placeholder_base)
        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected error occurred during PII redaction.")
            # Depending on policy, might re-raise or return original data
            raise

    def _recursive_redact(self, item: Any, patterns: Dict[str, str], placeholder_base: str) -> Any:
        """
        Recursively redacts PII from nested data structures.
        """
        if isinstance(item, str):
            redacted_item = self._redact_string(item, patterns, placeholder_base)
            if redacted_item != item:
                logger.debug(f"[{self.node_name}] Redacted string value.")
            return redacted_item
        elif isinstance(item, dict):
            redacted_dict = {}
            for key, value in item.items():
                redacted_dict[key] = self._recursive_redact(value, patterns, placeholder_base)
            return redacted_dict
        elif isinstance(item, list):
            return [self._recursive_redact(elem, patterns, placeholder_base) for elem in item]
        else:
            logger.debug(
                f"[{self.node_name}] Skipping redaction for unsupported data type: {type(item)}. "
                "Returning original item."
            )
            return item

# Example Usage (for testing purposes, not part of the node's core logic)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Set to DEBUG for more verbose logs

    redactor = PIIRedactorNode()

    test_data_str = "Please contact me at john.doe@example.com or call 555-123-4567. My IP is 192.168.1.1."
    test_data_dict = {
        "user_info": {
            "name": "Jane Smith",
            "email": "jane.smith@domain.net",
            "phone": "+1 (222) 333-4444",
            "address": "123 Main St, Anytown", # Not covered by default patterns
            "card_number": "4111 2222 3333 4444"
        },
        "log_entry": "Failed login from 10.0.0.5. SSN: 123-45-6789.",
        "misc_data": ["some text", "another@test.org", "No PII here."],
        "numeric_id": 12345,
    }
    test_data_list = [
        "Reach out to info@company.com.",
        {"transaction_id": "abc123", "client_ip": "203.0.113.45"},
        "Call +1-800-RED-ACTED for support.",
    ]
    
    # Test with default patterns
    print("--- Testing with Default Patterns ---")
    redacted_str = redactor.process(test_data_str, {})
    print(f"Original String:\n{test_data_str}")
    print(f"Redacted String:\n{redacted_str}\n")

    redacted_dict = redactor.process(test_data_dict, {})
    print(f"Original Dict:\n{test_data_dict}")
    print(f"Redacted Dict:\n{redacted_dict}\n")

    redacted_list = redactor.process(test_data_list, {})
    print(f"Original List:\n{test_data_list}")
    print(f"Redacted List:\n{redacted_list}\n")

    # Test with custom patterns and placeholder
    print("--- Testing with Custom Patterns and Placeholder ---")
    custom_patterns = {
        "my_phone": r"\b800-\w{3}-\w{6}\b", # Specific format
        "custom_email": r"\b[\w.-]+@custom\.com\b",
    }
    custom_context = {
        "redaction_patterns": custom_patterns,
        "redaction_placeholder": "<HIDDEN>",
    }
    
    test_custom_data = "My email is user@custom.com and call 800-RED-ACTED. Not john.doe@example.com"
    redacted_custom = redactor.process(test_custom_data, custom_context)
    print(f"Original Custom Data:\n{test_custom_data}")
    print(f"Redacted Custom Data:\n{redacted_custom}\n")

    # Test with empty data
    print("--- Testing with Empty Data ---")
    empty_result = redactor.process("", {})
    print(f"Empty String Result: '{empty_result}'")
    empty_dict_result = redactor.process({}, {})
    print(f"Empty Dict Result: {empty_dict_result}")
    empty_list_result = redactor.process([], {})
    print(f"Empty List Result: {empty_list_result}\n")
    
    # Test with unredactable data type
    print("--- Testing with Unredactable Data Type ---")
    unredactable_data = 12345
    unredactable_result = redactor.process(unredactable_data, {})
    print(f"Unredactable Data Result: {unredactable_result}\n")

    # Test with an invalid regex pattern in context (should log error and skip)
    print("--- Testing with Invalid Regex Pattern in Context ---")
    invalid_regex_context = {
        "redaction_patterns": {"bad_pattern": r"["}, # Invalid regex
        "redaction_placeholder": "[INVALID]"
    }
    test_invalid_regex_data = "This string has no PII for the bad pattern."
    redacted_invalid_regex = redactor.process(test_invalid_regex_data, invalid_regex_context)
    print(f"Original Invalid Regex Data:\n{test_invalid_regex_data}")
    print(f"Redacted Invalid Regex Data:\n{redacted_invalid_regex}\n")
    print("(Check logs above for regex error messages)")