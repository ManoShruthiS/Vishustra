import logging
from typing import Any, Dict, Type, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception raised when data validation fails against the defined schema."""
    pass

class DataValidator(BaseNode):
    """
    A Vishustra node that validates input data against a defined schema.

    This node expects the `context` dictionary to contain a 'validation_schema'
    key. The schema defines rules for validating the input `data`.

    Example schema structure for `context['validation_schema']`:
    ```python
    {
        'user_id': {'type': int, 'required': True, 'min_value': 1000},
        'username': {'type': str, 'required': True, 'min_length': 3, 'max_length': 20},
        'email': {'type': str, 'required': True, 'regex': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'},
        'age': {'type': (int, type(None)), 'required': False, 'min_value': 0, 'max_value': 120},
        'roles': {'type': list, 'required': True, 'min_length': 1},
        'profile_data': {'type': dict, 'required': False, 'nested_schema': {
            'bio': {'type': str, 'max_length': 500},
            'website': {'type': str, 'required': False}
        }}
    }
    ```

    The validator currently supports:
    - Required field check (`required`: `bool`).
    - Type checking (`type`: `Type` or `Tuple[Type, ...]`).
    - Minimum length for strings/lists/dicts (`min_length`: `int`).
    - Maximum length for strings/lists/dicts (`max_length`: `int`).
    - Minimum value for numbers (int, float) (`min_value`: `Union[int, float]`).
    - Maximum value for numbers (int, float) (`max_value`: `Union[int, float]`).
    - Basic nested schema validation (experimental, via `nested_schema`).
    - Future enhancements could include regex, enum checks, and custom validators.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input `data` against the schema provided in the `context`.

        Args:
            data: The data to be validated. Expected to be a dictionary if
                  a 'validation_schema' is provided that defines field-level rules.
            context: A dictionary potentially containing 'validation_schema',
                     which is a dictionary defining validation rules.

        Returns:
            The original data if validation passes.

        Raises:
            ValidationError: If the data fails any validation rule defined
                             in the schema.
            TypeError: If the 'validation_schema' in context is malformed
                       or if the `data` type is incompatible with schema validation.
        """
        validation_schema: Dict[str, Dict[str, Any]] = context.get('validation_schema', {})

        if not validation_schema:
            logger.warning("No 'validation_schema' found in context. Data will be passed through without validation.")
            return data

        if not isinstance(validation_schema, dict):
            logger.error(
                f"Invalid 'validation_schema' type in context: Expected dict, got {type(validation_schema).__name__}"
            )
            raise TypeError("The 'validation_schema' in context must be a dictionary.")

        if not isinstance(data, dict):
            logger.error(
                f"DataValidator expects 'data' to be a dictionary when a schema is provided. Got {type(data).__name__}."
            )
            raise TypeError("DataValidator can only validate dictionary data with a schema.")

        logger.debug(f"Starting validation for data with schema for fields: {list(validation_schema.keys())}")

        try:
            self._validate_dict_data(data, validation_schema, "root")
            logger.debug(f"Data validation successful against schema. Data passed through.")
            return data
        except ValidationError as e:
            logger.error(f"Data validation failed: {e}")
            raise
        except TypeError as e:
            logger.error(f"Schema configuration error: {e}")
            raise

    def _validate_dict_data(self, data: Dict[str, Any], schema: Dict[str, Dict[str, Any]], path: str):
        """
        Recursively validates dictionary data against a given schema.
        """
        for field_name, rules in schema.items():
            current_path = f"{path}.{field_name}"

            if not isinstance(rules, dict):
                raise TypeError(f"Malformed schema at '{current_path}': rules must be a dictionary. Got {type(rules).__name__}.")

            is_required = rules.get('required', False)
            expected_type: Union[Type, tuple, None] = rules.get('type')

            if field_name not in data:
                if is_required:
                    raise ValidationError(f"Required field '{current_path}' is missing from data.")
                else:
                    logger.debug(f"Optional field '{current_path}' is missing, skipping validation for it.")
                    continue # Skip further validation for missing optional fields

            field_value = data[field_name]

            # Type checking
            if expected_type is not None:
                if not isinstance(field_value, expected_type):
                    expected_type_names = expected_type.__name__ if isinstance(expected_type, type) else ', '.join(t.__name__ for t in expected_type)
                    raise ValidationError(
                        f"Field '{current_path}' has incorrect type. Expected {expected_type_names}, "
                        f"got {type(field_value).__name__} with value '{field_value}'."
                    )

            # Length/Value checks
            if isinstance(field_value, (str, list, dict)):
                if 'min_length' in rules:
                    min_len = rules['min_length']
                    if not isinstance(min_len, int) or min_len < 0:
                        raise TypeError(f"Invalid 'min_length' value for field '{current_path}'. Must be a non-negative integer.")
                    if len(field_value) < min_len:
                        raise ValidationError(
                            f"Field '{current_path}' length ({len(field_value)}) is less than minimum allowed ({min_len})."
                        )
                if 'max_length' in rules:
                    max_len = rules['max_length']
                    if not isinstance(max_len, int) or max_len < 0:
                        raise TypeError(f"Invalid 'max_length' value for field '{current_path}'. Must be a non-negative integer.")
                    if len(field_value) > max_len:
                        raise ValidationError(
                            f"Field '{current_path}' length ({len(field_value)}) is greater than maximum allowed ({max_len})."
                        )
            elif isinstance(field_value, (int, float)):
                if 'min_value' in rules:
                    min_val = rules['min_value']
                    if not isinstance(min_val, (int, float)):
                        raise TypeError(f"Invalid 'min_value' type for field '{current_path}'. Must be an int or float.")
                    if field_value < min_val:
                        raise ValidationError(
                            f"Field '{current_path}' value ({field_value}) is less than minimum allowed ({min_val})."
                        )
                if 'max_value' in rules:
                    max_val = rules['max_value']
                    if not isinstance(max_val, (int, float)):
                        raise TypeError(f"Invalid 'max_value' type for field '{current_path}'. Must be an int or float.")
                    if field_value > max_val:
                        raise ValidationError(
                            f"Field '{current_path}' value ({field_value}) is greater than maximum allowed ({max_val})."
                        )
            
            # Nested schema validation
            if isinstance(field_value, dict) and 'nested_schema' in rules:
                nested_schema = rules['nested_schema']
                if not isinstance(nested_schema, dict):
                    raise TypeError(f"Invalid 'nested_schema' for field '{current_path}'. Must be a dictionary.")
                self._validate_dict_data(field_value, nested_schema, current_path)
            # Add more specific validation types here if needed (e.g., regex, enum)
            
            logger.debug(f"Field '{current_path}' passed validation checks.")