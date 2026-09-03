import logging
from typing import Any, Dict, Type, Union, List, Tuple

# Assuming BaseNode is located at vishustra_core/nodes/base_node.py
# The base class definition provided in the prompt implies this structure.
from vishustra_core.nodes.base_node import BaseNode

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a predefined schema.

    This node is designed to ensure that data conforms to expected types, presence of keys,
    and optional value constraints (e.g., min/max for numbers, length for strings/lists).

    The validation schema is passed during initialization and should be a dictionary
    where keys correspond to expected data fields, and values are dictionaries
    specifying validation rules for that field.

    Example Schema:
    {
        "request_id": {"type": str, "min_length": 1},
        "payload": {
            "type": dict,
            "schema": { # Nested schema for complex objects
                "temperature": {"type": (int, float), "min_value": -273.15, "max_value": 5000},
                "unit": {"type": str, "allowed_values": ["celsius", "fahrenheit", "kelvin"]},
                "optional_field": {"type": bool, "required": False}
            }
        },
        "timestamp": {"type": int, "min_value": 0, "required": True},
        "tags": {"type": list, "item_type": str, "max_length": 10, "required": False},
        "status": {"type": str, "allowed_values": ["pending", "processed", "failed"]}
    }

    Supported rule types within a field's rule dictionary:
    - 'type': Expected Python type or tuple of types (e.g., str, int, (int, float)).
    - 'required': bool (default True). If False, the key can be missing without error.
    - 'min_length': int (for str, list, dict). Minimum length/size.
    - 'max_length': int (for str, list, dict). Maximum length/size.
    - 'min_value': int/float (for int, float). Minimum value.
    - 'max_value': int/float (for int, float). Maximum value.
    - 'allowed_values': list (for any type). Value must be one of the specified list.
    - 'item_type': Type or Tuple[Type] (for list). All items in the list must be of this type.
    - 'schema': dict (for dict). A nested schema for validating dictionary values.
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initializes the DataValidatorNode with a validation schema.

        Args:
            schema: A dictionary defining the validation rules for the input data.
                    See class docstring for schema structure examples.
        """
        self._schema = schema
        self._logger = logging.getLogger(self.node_name)
        self._logger.debug(f"[{self.node_name}] Initialized with schema: {self._schema}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidatorNode"

    def _validate_field(self, field_name: str, value: Any, rules: Dict[str, Any], path: str = "") -> List[str]:
        """
        Recursively validates a single field against its defined rules.
        """
        errors = []
        current_path = f"{path}.{field_name}" if path else field_name
        
        # 'required' status is handled by the caller (_validate_data) before calling _validate_field for missing fields.
        # So, if we reach here, the field is present, or it's an optional field that is present.

        if 'type' in rules:
            expected_type: Union[Type, Tuple[Type, ...]] = rules['type']
            if not isinstance(value, expected_type):
                errors.append(
                    f"Path '{current_path}': Type mismatch. Expected {expected_type.__name__ if isinstance(expected_type, type) else expected_type}, "
                    f"got {type(value).__name__}."
                )
                # If the type is fundamentally wrong, other type-dependent checks might fail
                # or produce misleading errors, so we return early for this field.
                return errors 

        if 'min_length' in rules and isinstance(value, (str, list, dict)):
            if len(value) < rules['min_length']:
                errors.append(
                    f"Path '{current_path}': Length {len(value)} is less than "
                    f"minimum allowed {rules['min_length']}."
                )

        if 'max_length' in rules and isinstance(value, (str, list, dict)):
            if len(value) > rules['max_length']:
                errors.append(
                    f"Path '{current_path}': Length {len(value)} is greater than "
                    f"maximum allowed {rules['max_length']}."
                )

        if 'min_value' in rules and isinstance(value, (int, float)):
            if value < rules['min_value']:
                errors.append(
                    f"Path '{current_path}': Value {value} is less than "
                    f"minimum allowed {rules['min_value']}."
                )

        if 'max_value' in rules and isinstance(value, (int, float)):
            if value > rules['max_value']:
                errors.append(
                    f"Path '{current_path}': Value {value} is greater than "
                    f"maximum allowed {rules['max_value']}."
                )

        if 'allowed_values' in rules:
            if value not in rules['allowed_values']:
                errors.append(
                    f"Path '{current_path}': Value '{value}' is not among "
                    f"allowed values: {rules['allowed_values']}."
                )

        if 'item_type' in rules and isinstance(value, list):
            expected_item_type: Union[Type, Tuple[Type, ...]] = rules['item_type']
            for i, item in enumerate(value):
                if not isinstance(item, expected_item_type):
                    errors.append(
                        f"Path '{current_path}[{i}]': Item type mismatch. "
                        f"Expected {expected_item_type.__name__ if isinstance(expected_item_type, type) else expected_item_type}, "
                        f"got {type(item).__name__}."
                    )
        
        if 'schema' in rules and isinstance(value, dict):
            errors.extend(self._validate_data(value, rules['schema'], path=current_path))
            
        return errors


    def _validate_data(self, data: Dict[str, Any], schema: Dict[str, Any], path: str = "") -> List[str]:
        """
        Performs the main validation logic recursively by iterating through the schema.
        """
        validation_errors: List[str] = []

        for key, rules in schema.items():
            field_path = f"{path}.{key}" if path else key
            is_required = rules.get('required', True)

            if key not in data:
                if is_required:
                    validation_errors.append(f"Path '{field_path}': Required key is missing.")
                else:
                    self._logger.debug(f"[{self.node_name}] Optional key '{field_path}' missing, skipping validation.")
                continue # Skip further validation for missing optional fields

            field_value = data[key]
            errors = self._validate_field(key, field_value, rules, path)
            validation_errors.extend(errors)
            
        return validation_errors


    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, validating it against the configured schema.

        Args:
            data: The input data to be validated. Expected to be a dictionary
                  that matches the schema structure defined during initialization.
            context: A dictionary containing contextual information for the node.
                     Not directly used for validation logic in this implementation,
                     but available for future extensions (e.g., dynamic schema loading).

        Returns:
            The original input data if validation is successful.

        Raises:
            TypeError: If the input data is not a dictionary when a dictionary schema
                       is applied.
            ValueError: If the data fails any validation rule defined in the schema.
        """
        self._logger.info(f"[{self.node_name}] Starting data validation process.")

        if not isinstance(data, dict):
            error_msg = f"[{self.node_name}] Input data for schema validation must be a dictionary. Got: {type(data).__name__}"
            self._logger.error(error_msg)
            raise TypeError(error_msg)

        validation_errors = self._validate_data(data, self._schema)

        if validation_errors:
            full_error_msg = (
                f"[{self.node_name}] Data validation failed with "
                f"{len(validation_errors)} error(s). Details:\n"
                + "\n".join([f"- {err}" for err in validation_errors])
            )
            self._logger.error(full_error_msg)
            # Re-raise with a concise message, full details are in logs
            raise ValueError(f"Data validation failed. First error: {validation_errors[0]}")
        
        self._logger.info(f"[{self.node_name}] Data validation successful. Returning original data.")
        return data

