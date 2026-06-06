import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a predefined schema.

    This node ensures that data conforms to specified types, formats, and constraints
    before further processing, preventing issues downstream in the orchestration flow.
    """

    def __init__(self, validation_schema: Dict[str, Dict[str, Any]]):
        """
        Initializes the DataValidatorNode with a validation schema.

        The schema defines rules for expected fields within the input data.

        Example schema structure:
        ```python
        {
            "user_id": {"type": "int", "min": 1, "required": True},
            "username": {"type": "str", "min_length": 3, "max_length": 50, "required": True},
            "email": {"type": "str", "format": "email", "required": False},
            "tags": {"type": "list", "of_type": "str", "max_items": 5, "required": False}
        }
        ```

        :param validation_schema: A dictionary defining validation rules for expected data fields.
                                  Each key represents a field name, and its value is a dictionary
                                  of rules (e.g., "type", "required", "min", "max", "min_length", "format").
        :raises TypeError: If `validation_schema` is not a dictionary.
        """
        if not isinstance(validation_schema, dict):
            raise TypeError("`validation_schema` must be a dictionary.")
        self.validation_schema = validation_schema
        logger.debug(f"DataValidatorNode initialized with schema: {self.validation_schema}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidatorNode"

    def _validate_field(self, field_name: str, value: Any, rules: Dict[str, Any]) -> None:
        """
        Helper method to validate a single field against its defined rules.

        :param field_name: The name of the field being validated.
        :param value: The actual value of the field from the input data.
        :param rules: A dictionary of validation rules for this specific field.
        :raises TypeError: If the field's type does not match the expected type.
        :raises ValueError: If the field fails any other validation rule (e.g., required, length, range, format).
        """
        is_required = rules.get("required", False)

        if value is None:
            if is_required:
                raise ValueError(f"Field '{field_name}' is required but received `None`.")
            else:
                return # Not required and None, so considered valid.

        expected_type_str = rules.get("type")
        if expected_type_str:
            type_map = {
                "str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict
            }
            expected_type = type_map.get(expected_type_str)
            if not expected_type:
                logger.warning(
                    f"Unsupported type '{expected_type_str}' specified in schema for field '{field_name}'. "
                    "Skipping type validation for this field."
                )
            elif not isinstance(value, expected_type):
                raise TypeError(
                    f"Field '{field_name}' expected type '{expected_type_str}', "
                    f"but got '{type(value).__name__}' with value: {value}"
                )

        if expected_type_str == "str":
            min_length = rules.get("min_length")
            if min_length is not None and len(value) < min_length:
                raise ValueError(
                    f"Field '{field_name}' requires a minimum length of {min_length}, "
                    f"but got length {len(value)}: '{value}'"
                )
            max_length = rules.get("max_length")
            if max_length is not None and len(value) > max_length:
                raise ValueError(
                    f"Field '{field_name}' exceeds maximum length of {max_length}, "
                    f"got length {len(value)}: '{value}'"
                )
            format_rule = rules.get("format")
            if format_rule == "email":
                if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", value):
                    raise ValueError(f"Field '{field_name}' must be a valid email format, but got: '{value}'")
            # Additional string format validations (e.g., "url", "uuid", "date-time") can be added here.

        elif expected_type_str == "int" or expected_type_str == "float":
            min_val = rules.get("min")
            if min_val is not None and value < min_val:
                raise ValueError(
                    f"Field '{field_name}' must be at least {min_val}, "
                    f"but got {value}"
                )
            max_val = rules.get("max")
            if max_val is not None and value > max_val:
                raise ValueError(
                    f"Field '{field_name}' must be at most {max_val}, "
                    f"but got {value}"
                )

        elif expected_type_str == "list":
            of_type_str = rules.get("of_type")
            if of_type_str:
                type_map = {
                    "str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict
                }
                item_expected_type = type_map.get(of_type_str)
                if not item_expected_type:
                    logger.warning(
                        f"Unsupported 'of_type' '{of_type_str}' specified for list field '{field_name}'. "
                        "Skipping item type validation for this list."
                    )
                else:
                    for idx, item in enumerate(value):
                        if not isinstance(item, item_expected_type):
                            raise TypeError(
                                f"Field '{field_name}' item at index {idx} expected type "
                                f"'{of_type_str}', but got '{type(item).__name__}' with value: {item}"
                            )
            max_items = rules.get("max_items")
            if max_items is not None and len(value) > max_items:
                raise ValueError(
                    f"Field '{field_name}' list exceeds maximum items of {max_items}, "
                    f"got {len(value)} items"
                )
            min_items = rules.get("min_items")
            if min_items is not None and len(value) < min_items:
                raise ValueError(
                    f"Field '{field_name}' list requires minimum items of {min_items}, "
                    f"but got {len(value)} items"
                )

        # Extend with validations for 'dict' with nested schemas, 'enum', custom regex, etc.

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured `validation_schema`.

        If validation fails for any field, a `ValueError` or `TypeError` is raised,
        containing details of the encountered errors.
        If validation succeeds, the original `data` is returned unchanged.

        :param data: The data to be validated. Expected to be a dictionary if a validation schema
                     with field-specific rules is provided.
        :param context: A dictionary containing contextual information for the node.
                        This node currently does not utilize the context.
        :return: The validated data (same as input if valid).
        :raises TypeError: If the input data is not a dictionary when the `validation_schema`
                           is non-empty, or if a field's type does not match its schema definition.
        :raises ValueError: If a field fails any other validation rule (e.g., required, length, range, format).
        """
        logger.info(f"Node '{self.node_name}' starting data validation.")
        validation_errors: List[str] = []

        if not isinstance(data, dict):
            # If a schema is defined (i.e., not empty) but input data is not a dictionary, it's an error.
            if self.validation_schema:
                error_msg = (
                    f"Input data expected to be a dictionary for validation based on the provided schema, "
                    f"but received '{type(data).__name__}'."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)
            else:
                # If no schema is provided for dict-level validation or the schema is empty,
                # and the data is not a dictionary, we pass it through.
                logger.warning(
                    f"Node '{self.node_name}' has an empty validation schema or received non-dictionary data. "
                    "Passing data without specific field validation."
                )
                return data

        for field_name, rules in self.validation_schema.items():
            value = data.get(field_name)
            try:
                self._validate_field(field_name, value, rules)
            except (TypeError, ValueError) as e:
                validation_errors.append(f"Validation failed for field '{field_name}': {e}")
                logger.debug(f"Validation failure for '{field_name}': {e}")

        # Note: By default, this validator ignores extra fields in `data` that are not present
        # in `validation_schema`. A 'strict' mode could be implemented to raise errors for
        # unexpected fields if required.

        if validation_errors:
            full_error_msg = (
                f"Node '{self.node_name}' encountered {len(validation_errors)} data validation error(s): " +
                "; ".join(validation_errors)
            )
            logger.error(full_error_msg)
            raise ValueError(full_error_msg)
        
        logger.info(f"Node '{self.node_name}' successfully validated data.")
        return data