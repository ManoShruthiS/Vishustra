import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a set of
    predefined rules specified in the orchestration context.

    This node is crucial for ensuring data quality and integrity at various
    stages of an LLM pipeline, preventing malformed or invalid data from
    propagating downstream.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "Data Validator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data based on a list of rules provided in the context.

        Validation rules should be provided in `context` under the key
        'validation_rules'. This key expects a list of dictionaries, where
        each dictionary defines a single validation rule for a specific field
        within the input `data`.

        Example structure for a single rule:
        ```json
        {
            "field": "user_id",
            "type": "int",
            "required": true,
            "min_value": 1,
            "max_value": 100000
        }
        ```

        Supported rule properties:
        - `field` (str): The key in the data (e.g., 'user_id', 'email') to validate.
          Required for each rule.
        - `type` (str): The expected data type ('str', 'int', 'float', 'bool', 'list', 'dict').
          Required for each rule.
        - `required` (bool, optional): If `True`, the `field` must exist in `data`.
          Defaults to `False`.
        - `min_length` (int, optional): Minimum length for 'str' or 'list' types.
        - `max_length` (int, optional): Maximum length for 'str' or 'list' types.
        - `min_value` (Union[int, float], optional): Minimum value for 'int' or 'float' types.
        - `max_value` (Union[int, float], optional): Maximum value for 'int' or 'float' types.
        - `format` (str, optional): Special format validation for 'str' types.
          Currently supports:
            - 'email': Validates if the string is a basic email format.
        - `of_type` (str, optional): For 'list' types, specifies the expected type
          of each element within the list (e.g., 'str', 'int').

        Args:
            data (Any): The input data to be validated. It is expected to be
                        a dictionary if field-specific validation rules are applied.
            context (Dict[str, Any]): The operational context, containing validation
                                      rules under the 'validation_rules' key.

        Returns:
            Any: The original input data if all validation rules pass successfully.

        Raises:
            ValueError: If validation rules are malformed, or if the input data
                        fails to meet any of the specified validation criteria.
        """
        if not isinstance(context, dict):
            logger.error("Context for DataValidatorNode must be a dictionary.")
            raise ValueError("Context must be a dictionary to provide validation rules.")

        validation_rules: List[Dict[str, Any]] = context.get('validation_rules', [])

        if not validation_rules:
            logger.warning("No 'validation_rules' found in context. Data passed without validation.")
            return data

        if not isinstance(data, dict):
            # If rules are provided, they expect dictionary-like data for field access.
            logger.error(f"Validation rules expect dictionary-like data, but received {type(data).__name__}.")
            raise ValueError(
                "Input data must be a dictionary to apply field-specific validation rules. "
                "No 'validation_rules' were found for non-dictionary data types."
            )

        logger.info(f"Initiating data validation for {len(validation_rules)} rules.")
        for i, rule in enumerate(validation_rules):
            try:
                self._apply_rule(data, rule)
            except ValueError as e:
                logger.error(
                    f"Validation failed for rule {i+1} (field: '{rule.get('field', 'N/A')}'): {e}"
                )
                raise  # Re-raise the validation error to halt processing

        logger.info("Data successfully validated against all specified rules.")
        return data

    def _apply_rule(self, data: Dict[str, Any], rule: Dict[str, Any]) -> None:
        """Applies a single validation rule to a specific field in the data."""
        field = rule.get('field')
        expected_type_str = rule.get('type')
        required = rule.get('required', False)

        if not isinstance(field, str) or not field:
            raise ValueError(f"Malformed validation rule: 'field' must be a non-empty string. Rule: {rule}")
        if not isinstance(expected_type_str, str) or not expected_type_str:
            raise ValueError(f"Malformed validation rule for field '{field}': 'type' must be a non-empty string. Rule: {rule}")

        # Check if field exists and is required
        if field not in data:
            if required:
                raise ValueError(f"Required field '{field}' is missing from data.")
            else:
                # If not required and missing, no further validation needed for this field.
                return

        value = data[field]
        actual_type = type(value)

        # Map string type names to actual Python types
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict,
        }
        expected_type = type_mapping.get(expected_type_str)

        if expected_type is None:
            logger.warning(
                f"Unknown type '{expected_type_str}' specified for field '{field}'. "
                "Skipping strict type check for this rule."
            )
        elif not isinstance(value, expected_type):
            raise ValueError(
                f"Field '{field}' has incorrect type. Expected '{expected_type_str}', "
                f"but received '{actual_type.__name__}' (value: {value!r})."
            )

        # Apply specific constraints based on the expected type
        if expected_type in [str, list]:
            if 'min_length' in rule and not isinstance(rule['min_length'], int):
                raise ValueError(f"Malformed rule for '{field}': 'min_length' must be an integer.")
            if 'max_length' in rule and not isinstance(rule['max_length'], int):
                raise ValueError(f"Malformed rule for '{field}': 'max_length' must be an integer.")

            if 'min_length' in rule and len(value) < rule['min_length']:
                raise ValueError(
                    f"Field '{field}' must have a minimum length of {rule['min_length']}, "
                    f"but has length {len(value)}."
                )
            if 'max_length' in rule and len(value) > rule['max_length']:
                raise ValueError(
                    f"Field '{field}' must have a maximum length of {rule['max_length']}, "
                    f"but has length {len(value)}."
                )

        if expected_type in [int, float]:
            if 'min_value' in rule and not isinstance(rule['min_value'], (int, float)):
                raise ValueError(f"Malformed rule for '{field}': 'min_value' must be a number.")
            if 'max_value' in rule and not isinstance(rule['max_value'], (int, float)):
                raise ValueError(f"Malformed rule for '{field}': 'max_value' must be a number.")

            if 'min_value' in rule and value < rule['min_value']:
                raise ValueError(
                    f"Field '{field}' must have a minimum value of {rule['min_value']}, "
                    f"but is {value}."
                )
            if 'max_value' in rule and value > rule['max_value']:
                raise ValueError(
                    f"Field '{field}' must have a maximum value of {rule['max_value']}, "
                    f"but is {value}."
                )

        if expected_type == str and 'format' in rule:
            format_type = rule['format']
            if not isinstance(format_type, str):
                raise ValueError(f"Malformed rule for '{field}': 'format' must be a string.")

            if format_type == 'email':
                # A robust email regex for common cases.
                # This regex is a balance between strictness and practical acceptance.
                email_regex = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
                if not email_regex.fullmatch(value):
                    raise ValueError(f"Field '{field}' must be a valid email address, but is '{value}'.")
            else:
                logger.warning(
                    f"Unsupported format '{format_type}' specified for string field '{field}'. "
                    "Skipping format validation."
                )

        # Validate elements within a list if 'of_type' is specified
        if expected_type == list and 'of_type' in rule:
            item_expected_type_str = rule['of_type']
            if not isinstance(item_expected_type_str, str) or not item_expected_type_str:
                raise ValueError(f"Malformed rule for list field '{field}': 'of_type' must be a non-empty string.")

            item_expected_type = type_mapping.get(item_expected_type_str)
            if item_expected_type is None:
                logger.warning(
                    f"Unknown item type '{item_expected_type_str}' for list field '{field}'. "
                    "Skipping individual item type checks."
                )
            else:
                for idx, item in enumerate(value):
                    if not isinstance(item, item_expected_type):
                        raise ValueError(
                            f"List item at index {idx} in field '{field}' has incorrect type. "
                            f"Expected '{item_expected_type_str}', but got '{type(item).__name__}' (value: {item!r})."
                        )