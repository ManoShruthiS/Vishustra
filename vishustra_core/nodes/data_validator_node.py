
import logging
import re
from typing import Any, Dict, Type, Union, Callable

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A processing node designed to validate input data against a set of predefined rules.

    This node ensures that incoming data conforms to expected types, structures,
    and constraints before it proceeds further in the orchestration pipeline.
    If validation fails, a `ValueError` or `TypeError` is raised, halting
    further processing for that data packet and signaling an issue upstream.

    Validation rules are defined during the node's initialization, allowing for
    flexible and declarative data schema enforcement.
    """

    def __init__(self, validation_rules: Dict[str, Dict[str, Any]]):
        """
        Initializes the DataValidatorNode with a set of validation rules.

        Args:
            validation_rules: A dictionary where keys are field names to be validated,
                              and values are dictionaries specifying validation criteria
                              for that field.

                              Example `validation_rules` structure:
                              ```python
                              {
                                  "user_id": {"type": str, "min_length": 5, "max_length": 50, "required": True},
                                  "age": {"type": int, "min_value": 0, "max_value": 120, "required": False},
                                  "email": {"type": str, "regex": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
                                  "preferences": {"type": dict, "required": False},
                                  "tags": {"type": list, "min_length": 1, "custom_validator": lambda t: all(isinstance(x, str) for x in t) }
                              }
                              ```
                              Supported rule keys within each field's dictionary:
                              - `'type'`: `Type`, e.g., `str`, `int`, `float`, `bool`, `list`, `dict`.
                              - `'required'`: `bool`, if the field must be present (default: `True`).
                              - `'min_length'`: `int`, minimum length for `str`/`list`/`dict`.
                              - `'max_length'`: `int`, maximum length for `str`/`list`/`dict`.
                              - `'min_value'`: `Union[int, float]`, minimum value for numbers.
                              - `'max_value'`: `Union[int, float]`, maximum value for numbers.
                              - `'regex'`: `str`, regular expression pattern for strings.
                              - `'custom_validator'`: `Callable[[Any], bool]`, a custom function that
                                                      takes the field's value and returns `True` for valid,
                                                      `False` for invalid. If an exception is raised by the
                                                      custom validator, it's caught and re-raised as a `ValueError`.
        """
        if not isinstance(validation_rules, dict):
            raise TypeError("Validation rules must be a dictionary.")
        self.validation_rules = validation_rules
        logger.debug(f"[{self.node_name}] Initialized with {len(validation_rules)} validation rules.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def _validate_field(self, field_name: str, value: Any, rules: Dict[str, Any]) -> None:
        """
        Helper method to validate a single field against its specific rules.

        Args:
            field_name: The name of the field being validated.
            value: The actual value of the field from the input data.
            rules: The dictionary of validation rules for this specific field.

        Raises:
            TypeError: If the value's type does not match the expected type.
            ValueError: If any other validation rule (length, range, regex, custom) fails.
        """
        expected_type = rules.get('type')
        required = rules.get('required', True)

        if value is None and not required:
            # If field is optional and its value is None (or missing and resulted in None), it's valid.
            return

        if value is None and required:
            raise ValueError(f"Field '{field_name}' is required but its value is None.")

        if expected_type and not isinstance(value, expected_type):
            raise TypeError(f"Field '{field_name}': Expected type {expected_type.__name__}, got {type(value).__name__}.")

        # Length validation for strings, lists, dicts
        if isinstance(value, (str, list, dict)):
            min_length = rules.get('min_length')
            max_length = rules.get('max_length')
            current_length = len(value)
            if min_length is not None and current_length < min_length:
                raise ValueError(f"Field '{field_name}': Length {current_length} is less than minimum {min_length}.")
            if max_length is not None and current_length > max_length:
                raise ValueError(f"Field '{field_name}': Length {current_length} is greater than maximum {max_length}.")

        # Value range validation for numbers
        if isinstance(value, (int, float)):
            min_value = rules.get('min_value')
            max_value = rules.get('max_value')
            if min_value is not None and value < min_value:
                raise ValueError(f"Field '{field_name}': Value {value} is less than minimum {min_value}.")
            if max_value is not None and value > max_value:
                raise ValueError(f"Field '{field_name}': Value {value} is greater than maximum {max_value}.")

        # Regex validation for strings
        if isinstance(value, str):
            regex_pattern = rules.get('regex')
            if regex_pattern:
                if not re.fullmatch(regex_pattern, value):
                    raise ValueError(f"Field '{field_name}': Value '{value}' does not match regex pattern '{regex_pattern}'.")

        # Custom validator function
        custom_validator = rules.get('custom_validator')
        if callable(custom_validator):
            try:
                if not custom_validator(value):
                    raise ValueError(f"Field '{field_name}': Custom validator returned False for value '{value}'.")
            except Exception as e:
                # Catch any exception raised by the custom validator and wrap it.
                raise ValueError(f"Field '{field_name}': Custom validator raised an exception: {e}") from e

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured rules.

        Args:
            data: The input data (expected to be a dictionary) to be validated.
            context: A dictionary containing contextual information for the node.
                     (Not directly used by this validator node's logic).

        Returns:
            The validated data (unmodified if validation passes).

        Raises:
            TypeError: If the input data is not a dictionary, or a field's type is incorrect.
            ValueError: If any validation rule fails (e.g., missing required field,
                        invalid length, out-of-range value, regex mismatch, custom validator failure).
        """
        logger.info(f"[{self.node_name}] Starting data validation for incoming data.")

        if not isinstance(data, dict):
            error_msg = f"[{self.node_name}] Input data must be a dictionary for validation, received {type(data).__name__}."
            logger.error(error_msg)
            raise TypeError(error_msg)

        for field_name, rules in self.validation_rules.items():
            value = data.get(field_name) # Retrieve value, None if field is missing
            is_required = rules.get('required', True)

            try:
                if field_name not in data and is_required:
                    # Explicitly check for missing required fields before passing to _validate_field
                    error_msg = f"[{self.node_name}] Validation failed: Required field '{field_name}' is missing from data."
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                # If the field is present, or if it's not required (and thus 'value' might be None),
                # proceed with detailed validation. _validate_field handles the 'None' case for optional fields.
                self._validate_field(field_name, value, rules)
            except (ValueError, TypeError) as e:
                # Re-raise with node context for clarity in orchestrator logs
                error_msg = f"[{self.node_name}] Validation error for field '{field_name}': {e}"
                logger.error(error_msg)
                # Preserve the original exception type and traceback
                raise type(e)(error_msg) from e

        logger.info(f"[{self.node_name}] Data validation successful. Data passed through.")
        return data

