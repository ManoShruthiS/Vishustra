import logging
from typing import Any, Dict, List, Union
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class NodeConfigurationError(ValueError):
    """Exception raised for errors in node configuration (e.g., missing required context)."""
    pass

class DataValidationError(ValueError):
    """Exception raised when the input data fails validation against defined rules."""
    pass

class DataValidatorNode(BaseNode):
    """
    A processing node that validates input data against a specified schema or set of rules.
    
    The validation rules are expected to be provided in the `context` dictionary
    under the key 'validation_schema'.
    
    Example `validation_schema` structure:
    {
        "field_name_1": {"type": "str", "required": True, "min_length": 5},
        "field_name_2": {"type": "int", "required": False, "min_value": 0, "max_value": 100},
        "field_name_3": {"type": "list", "items_type": "str", "required": True, "min_items": 1},
        "field_name_4": {"type": "bool", "required": True},
        "field_name_5": {"type": "dict", "required": False, "schema": {"sub_field": {"type": "str", "required": True}}},
    }
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidatorNode"

    def _validate_field(self, field_name: str, field_value: Any, rules: Dict[str, Any]) -> None:
        """
        Validates a single field against its defined rules.
        """
        is_required = rules.get("required", False)

        if field_value is None:
            if is_required:
                raise DataValidationError(f"Field '{field_name}' is required but received None.")
            else:
                return # Not required and not present, so no further validation needed.

        expected_type_str = rules.get("type")
        if not expected_type_str:
            logger.debug(f"No type specified for field '{field_name}', skipping type check.")
        else:
            expected_type_map = {
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "any": Any
            }
            expected_type = expected_type_map.get(expected_type_str)
            if expected_type is None:
                logger.warning(f"Unknown type '{expected_type_str}' specified for field '{field_name}'. Skipping type validation.")
            elif not isinstance(field_value, expected_type):
                raise DataValidationError(
                    f"Field '{field_name}' expected type '{expected_type_str}', "
                    f"but received '{type(field_value).__name__}' with value '{field_value}'."
                )

        if expected_type_str == "str":
            min_length = rules.get("min_length")
            if min_length is not None and len(field_value) < min_length:
                raise DataValidationError(
                    f"Field '{field_name}' (string) has length {len(field_value)}, "
                    f"which is less than the required minimum of {min_length}."
                )
            max_length = rules.get("max_length")
            if max_length is not None and len(field_value) > max_length:
                raise DataValidationError(
                    f"Field '{field_name}' (string) has length {len(field_value)}, "
                    f"which exceeds the maximum allowed length of {max_length}."
                )

        elif expected_type_str in ["int", "float"]:
            min_value = rules.get("min_value")
            if min_value is not None and field_value < min_value:
                raise DataValidationError(
                    f"Field '{field_name}' has value {field_value}, "
                    f"which is less than the required minimum of {min_value}."
                )
            max_value = rules.get("max_value")
            if max_value is not None and field_value > max_value:
                raise DataValidationError(
                    f"Field '{field_name}' has value {field_value}, "
                    f"which exceeds the maximum allowed value of {max_value}."
                )

        elif expected_type_str == "list":
            min_items = rules.get("min_items")
            if min_items is not None and len(field_value) < min_items:
                raise DataValidationError(
                    f"Field '{field_name}' (list) has {len(field_value)} items, "
                    f"which is less than the required minimum of {min_items}."
                )
            max_items = rules.get("max_items")
            if max_items is not None and len(field_value) > max_items:
                raise DataValidationError(
                    f"Field '{field_name}' (list) has {len(field_value)} items, "
                    f"which exceeds the maximum allowed items of {max_items}."
                )
            items_type_str = rules.get("items_type")
            if items_type_str and field_value: # Only validate items if list is not empty
                items_expected_type = expected_type_map.get(items_type_str)
                if items_expected_type:
                    for i, item in enumerate(field_value):
                        if not isinstance(item, items_expected_type):
                            raise DataValidationError(
                                f"Field '{field_name}' (list) item at index {i} expected type "
                                f"'{items_type_str}', but received '{type(item).__name__}'."
                            )
                else:
                    logger.warning(f"Unknown item type '{items_type_str}' for list field '{field_name}'. Skipping item type validation.")

        elif expected_type_str == "dict":
            sub_schema = rules.get("schema")
            if sub_schema is not None:
                # Recursively validate sub-dictionary
                try:
                    self._validate_data_against_schema(field_value, sub_schema)
                except DataValidationError as e:
                    raise DataValidationError(f"Validation failed for sub-dictionary in field '{field_name}': {e}") from e


    def _validate_data_against_schema(self, data: Any, schema: Dict[str, Any]) -> None:
        """
        Validates the given data against the provided schema.
        Assumes data is a dictionary if schema is provided.
        """
        if not isinstance(data, dict):
            raise DataValidationError(f"Expected data to be a dictionary for schema validation, but got {type(data).__name__}.")

        for field_name, rules in schema.items():
            field_value = data.get(field_name)

            if rules.get("required", False) and field_value is None:
                raise DataValidationError(f"Required field '{field_name}' is missing or None.")
            
            # If field is not required and not present, skip further validation for this field
            if field_value is None and not rules.get("required", False):
                logger.debug(f"Optional field '{field_name}' is missing; skipping validation for this field.")
                continue

            self._validate_field(field_name, field_value, rules)

        logger.debug("All fields validated successfully against the schema.")


    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data based on the 'validation_schema' provided in the context.

        Args:
            data: The input data to be validated. Expected to be a dictionary for schema validation.
            context: A dictionary containing runtime information, including
                     'validation_schema' which defines the rules for validation.

        Returns:
            The original data if validation passes.

        Raises:
            NodeConfigurationError: If 'validation_schema' is missing or malformed in context.
            DataValidationError: If the input data fails any of the defined validation rules.
        """
        logger.info(f"[{self.node_name}] Starting data validation.")

        validation_schema = context.get("validation_schema")

        if not validation_schema:
            raise NodeConfigurationError(
                f"[{self.node_name}] 'validation_schema' not found in context. "
                "The DataValidatorNode requires a schema to operate."
            )
        if not isinstance(validation_schema, dict):
            raise NodeConfigurationError(
                f"[{self.node_name}] 'validation_schema' in context must be a dictionary, "
                f"but received '{type(validation_schema).__name__}'."
            )

        try:
            self._validate_data_against_schema(data, validation_schema)
            logger.info(f"[{self.node_name}] Data successfully passed all validation checks.")
            return data
        except DataValidationError as e:
            logger.error(f"[{self.node_name}] Data validation failed: {e}")
            raise
        except Exception as e:
            logger.critical(f"[{self.node_name}] An unexpected error occurred during validation: {e}", exc_info=True)
            raise DataValidationError(f"An unexpected error occurred during validation: {e}") from e
