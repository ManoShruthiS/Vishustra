import logging
import re
from typing import Any, Dict, Type

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception raised when data validation fails within the DataValidatorNode."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra node for validating incoming data against a predefined schema.

    This node ensures that data conforms to expected types, presence, and ranges
    before being passed to subsequent processing stages. It supports various
    validation rules including type checking, required fields, length constraints
    for strings/lists, value ranges for numbers, and item type validation for lists.
    """

    _TYPE_MAP: Dict[str, Type] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "any": Any # Special type for when any type is acceptable
    }

    def __init__(self, schema: Dict[str, Any], on_validation_failure: str = 'raise'):
        """
        Initializes the DataValidatorNode with a validation schema.

        Args:
            schema (Dict[str, Any]): A dictionary defining the validation rules.
                                     Example:
                                     {
                                         "user_id": {"type": "int", "required": True, "min_value": 1},
                                         "username": {"type": "str", "required": True, "min_length": 3, "max_length": 50},
                                         "email": {"type": "str", "required": False, "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"},
                                         "tags": {"type": "list", "item_type": "str", "max_length": 5}
                                     }
            on_validation_failure (str): Strategy to handle validation failures.
                                         'raise': Raise a ValidationError immediately. (Default)
                                         'log_and_pass': Log the error and pass the original data.
                                         'log_and_none': Log the error and return None (effectively dropping the data).
        """
        if not isinstance(schema, dict):
            raise TypeError("Schema must be a dictionary.")
        if not schema:
            logger.warning(
                f"[{self.__class__.__name__}] Node initialized with an empty schema. "
                "Data will pass through without validation."
            )
        self._schema = schema

        allowed_failures = {'raise', 'log_and_pass', 'log_and_none'}
        if on_validation_failure not in allowed_failures:
            raise ValueError(
                f"Invalid 'on_validation_failure' strategy: '{on_validation_failure}'. "
                f"Must be one of {', '.join(allowed_failures)}"
            )
        self._on_validation_failure = on_validation_failure
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured schema.

        Args:
            data (Any): The data to be validated. Expected to be a dictionary
                        if the schema is non-empty and expects dict validation.
            context (Dict[str, Any]): The processing context.

        Returns:
            Any: The original data if validation passes, or `None` if
                 `on_validation_failure` is 'log_and_none' and validation fails.

        Raises:
            ValidationError: If `on_validation_failure` is 'raise' and validation fails.
            TypeError: If the input data is not a dictionary when a schema is defined
                       and `on_validation_failure` is 'raise'.
        """
        if not self._schema:
            self._logger.debug("No schema configured. Data passed through without validation.")
            return data

        if not isinstance(data, dict):
            msg = (
                f"Expected dictionary input for validation based on schema, "
                f"but received type '{type(data).__name__}'."
            )
            self._logger.error(msg)
            if self._on_validation_failure == 'raise':
                raise TypeError(msg)
            elif self._on_validation_failure == 'log_and_none':
                return None
            else: # 'log_and_pass'
                return data

        validation_errors = []
        for field_name, rules in self._schema.items():
            value = data.get(field_name)
            is_present = field_name in data

            # Rule: required
            if rules.get("required") and not is_present:
                validation_errors.append(f"Field '{field_name}' is required but missing.")
                continue # Skip further checks for this field if it's missing and required

            if is_present: # Only apply other rules if the field is present
                # Rule: type
                expected_type_str = rules.get("type")
                if expected_type_str:
                    expected_type = self._TYPE_MAP.get(expected_type_str)
                    if not expected_type:
                        self._logger.warning(
                            f"Unknown type '{expected_type_str}' specified for field '{field_name}' "
                            f"in schema. Skipping type validation for this field."
                        )
                    elif expected_type is not Any and not isinstance(value, expected_type):
                        validation_errors.append(
                            f"Field '{field_name}' has type '{type(value).__name__}', "
                            f"expected '{expected_type_str}'."
                        )

                # Rule: min_length / max_length (for strings and lists)
                if isinstance(value, (str, list)):
                    min_len = rules.get("min_length")
                    if min_len is not None and len(value) < min_len:
                        validation_errors.append(
                            f"Field '{field_name}' length ({len(value)}) is less than "
                            f"minimum required length ({min_len})."
                        )
                    max_len = rules.get("max_length")
                    if max_len is not None and len(value) > max_len:
                        validation_errors.append(
                            f"Field '{field_name}' length ({len(value)}) is greater than "
                            f"maximum allowed length ({max_len})."
                        )

                # Rule: min_value / max_value (for numbers)
                if isinstance(value, (int, float)):
                    min_val = rules.get("min_value")
                    if min_val is not None and value < min_val:
                        validation_errors.append(
                            f"Field '{field_name}' value ({value}) is less than "
                            f"minimum allowed value ({min_val})."
                        )
                    max_val = rules.get("max_value")
                    if max_val is not None and value > max_val:
                        validation_errors.append(
                            f"Field '{field_name}' value ({value}) is greater than "
                            f"maximum allowed value ({max_val})."
                        )
                
                # Rule: pattern (for strings)
                if isinstance(value, str):
                    pattern = rules.get("pattern")
                    if pattern is not None:
                        try:
                            if not re.fullmatch(pattern, value):
                                validation_errors.append(
                                    f"Field '{field_name}' value ('{value}') does not match "
                                    f"required pattern '{pattern}'."
                                )
                        except re.error as e:
                            self._logger.error(
                                f"Invalid regex pattern '{pattern}' for field '{field_name}': {e}"
                            )
                            # Do not add to validation_errors as it's a schema issue, not data issue
                            
                # Rule: item_type for lists
                if isinstance(value, list) and "item_type" in rules:
                    expected_item_type_str = rules["item_type"]
                    expected_item_type = self._TYPE_MAP.get(expected_item_type_str)
                    if not expected_item_type:
                        self._logger.warning(
                            f"Unknown item_type '{expected_item_type_str}' specified for list field '{field_name}' "
                            f"in schema. Skipping item type validation for this field."
                        )
                    else:
                        for idx, item in enumerate(value):
                            if expected_item_type is not Any and not isinstance(item, expected_item_type):
                                validation_errors.append(
                                    f"List field '{field_name}' at index {idx} has item type "
                                    f"'{type(item).__name__}', expected '{expected_item_type_str}'."
                                )

        if validation_errors:
            full_error_msg = (
                f"Data validation failed for node '{self.node_name}' (Data ID: {context.get('request_id', 'N/A')}). "
                f"Errors: {'; '.join(validation_errors)}"
            )
            self._logger.error(full_error_msg)
            if self._on_validation_failure == 'raise':
                raise ValidationError(full_error_msg)
            elif self._on_validation_failure == 'log_and_none':
                return None
            else: # 'log_and_pass'
                return data
        else:
            self._logger.debug(
                f"Data successfully validated by '{self.node_name}' "
                f"(Data ID: {context.get('request_id', 'N/A')})."
            )
            return data