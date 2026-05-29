import logging
import re
from typing import Any, Dict, Type

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidator(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against a
    predefined schema.

    This node ensures that data conforms to expected types, presence requirements,
    and specific constraints (e.g., length, range, enumeration, regex patterns).
    It is crucial for maintaining data integrity and preventing downstream
    processing errors.
    """

    # Maps string representations of types in the schema to actual Python types.
    _TYPE_MAP: Dict[str, Type[Any]] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    def __init__(self, validation_schema: Dict[str, Dict[str, Any]]):
        """
        Initializes the DataValidator node with a specific validation schema.

        The schema defines rules for each expected field in the input data.

        Args:
            validation_schema: A dictionary defining validation rules for data fields.
                               Each key represents a field name, and its value is
                               another dictionary of rules.
                               Example Schema Structure:
                               {
                                   "user_id": {"type": "int", "required": True, "min_value": 1},
                                   "username": {"type": "str", "required": True, "min_length": 3, "max_length": 50},
                                   "email": {"type": "str", "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", "required": False},
                                   "status": {"type": "str", "enum": ["active", "inactive", "pending"], "default": "pending"},
                                   "tags": {"type": "list", "item_type": "str", "required": False}
                               }
                               Supported rules:
                               - 'type': Expected Python type (e.g., "str", "int", "dict").
                               - 'required': boolean, if the field must be present.
                               - 'min_length', 'max_length': for strings and lists.
                               - 'min_value', 'max_value': for integers and floats.
                               - 'enum': list of allowed values.
                               - 'pattern': regex pattern for string fields.
        
        Raises:
            TypeError: If the provided validation_schema is not a dictionary.
        """
        if not isinstance(validation_schema, dict):
            raise TypeError("validation_schema must be a dictionary.")
        self.validation_schema = validation_schema
        logger.debug(f"{self.node_name} initialized with schema: {validation_schema}")

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured schema.

        If validation fails, a ValueError is raised with a comprehensive message
        detailing all encountered errors. If validation passes, the original
        (potentially defaulted, though not implemented in this version) data
        is returned.

        Args:
            data: The input data to be validated. Typically expected to be a
                  dictionary for schema-based validation.
            context: A dictionary containing contextual information relevant
                     to the orchestration, not directly used by this validator
                     but passed through.

        Returns:
            The original input data, if it passes all validation rules.

        Raises:
            TypeError: If the input 'data' is not a dictionary, which is
                       required for schema-based validation.
            ValueError: If 'data' fails to meet any of the specified constraints
                        in the validation schema.
        """
        if not isinstance(data, dict):
            logger.error(f"{self.node_name} received non-dictionary data for schema validation. Type: {type(data).__name__}")
            raise TypeError(f"Input data for '{self.node_name}' must be a dictionary for schema validation, but got '{type(data).__name__}'.")

        validation_errors = []

        for field_name, rules in self.validation_schema.items():
            field_value = data.get(field_name)
            is_present = field_name in data
            
            # 1. Required Field Check
            required = rules.get("required", False)
            if required and not is_present:
                validation_errors.append(f"Field '{field_name}': Is required but missing.")
                continue  # Cannot proceed with further validation for a missing required field.

            if not is_present: # Not required and not present, skip further validation for this field.
                continue

            # 2. Type Validation
            expected_type_name = rules.get("type")
            if expected_type_name:
                expected_type = self._TYPE_MAP.get(expected_type_name)
                if expected_type is None:
                    logger.warning(f"Schema for '{field_name}' specifies unknown type '{expected_type_name}'. Skipping type validation for this field.")
                elif not isinstance(field_value, expected_type):
                    validation_errors.append(f"Field '{field_name}': Expected type '{expected_type_name}', but got '{type(field_value).__name__}'.")

            # 3. Length Constraints (for strings and lists)
            if isinstance(field_value, (str, list)):
                min_length = rules.get("min_length")
                max_length = rules.get("max_length")

                if min_length is not None and len(field_value) < min_length:
                    validation_errors.append(f"Field '{field_name}': Length ({len(field_value)}) is less than minimum required ({min_length}).")
                if max_length is not None and len(field_value) > max_length:
                    validation_errors.append(f"Field '{field_name}': Length ({len(field_value)}) exceeds maximum allowed ({max_length}).")

            # 4. Value Range Constraints (for numbers)
            if isinstance(field_value, (int, float)):
                min_value = rules.get("min_value")
                max_value = rules.get("max_value")

                if min_value is not None and field_value < min_value:
                    validation_errors.append(f"Field '{field_name}': Value ({field_value}) is less than minimum allowed ({min_value}).")
                if max_value is not None and field_value > max_value:
                    validation_errors.append(f"Field '{field_name}': Value ({field_value}) exceeds maximum allowed ({max_value}).")

            # 5. Enumeration (allowed values)
            enum_values = rules.get("enum")
            if enum_values is not None:
                if not isinstance(enum_values, list):
                    logger.warning(f"Schema for '{field_name}' has invalid 'enum' rule (expected list, got {type(enum_values).__name__}). Skipping enum validation.")
                elif field_value not in enum_values:
                    validation_errors.append(f"Field '{field_name}': Value ('{field_value}') is not one of the allowed values: {enum_values}.")

            # 6. Regex Pattern Matching (for strings)
            pattern = rules.get("pattern")
            if pattern is not None and isinstance(field_value, str):
                if not re.fullmatch(pattern, field_value): # Use fullmatch to match the entire string
                    validation_errors.append(f"Field '{field_name}': Value ('{field_value}') does not match required pattern '{pattern}'.")

        if validation_errors:
            error_message = f"Data validation failed for '{self.node_name}'. Encountered {len(validation_errors)} error(s):\n"
            for error in validation_errors:
                error_message += f"- {error}\n"
            logger.error(error_message.strip())
            raise ValueError(error_message.strip())
        
        logger.info(f"Data successfully validated by '{self.node_name}'.")
        return data