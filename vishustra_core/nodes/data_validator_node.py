import logging
import re
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationError(ValueError):
    """Custom exception raised when data validation fails."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra node that validates input data against a set of predefined rules.
    
    Validation rules are expected to be provided in the `context` dictionary
    under the key 'validation_rules'. These rules can specify required fields,
    data types, string length constraints, numeric ranges, and regex patterns.
    
    Expected `context['validation_rules']` structure:
    If `data` is a dictionary:
    {
        'field_name_1': {'type': 'str', 'required': True, 'min_length': 1, 'max_length': 255, 'regex': r'^[a-zA-Z0-9_]+$'},
        'field_name_2': {'type': 'int', 'required': False, 'min_value': 0, 'max_value': 100},
        'field_name_3': {'type': 'list', 'item_type': 'str', 'required': True, 'min_items': 1},
        'field_name_4': {'type': 'bool', 'required': False},
        ...
    }
    If `data` is not a dictionary (e.g., a single value):
    {
        'type': 'str', 'required': True, 'min_length': 1, 'max_length': 255
    }
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data based on rules provided in the context.

        Args:
            data: The input data to be validated. Can be any type, but
                  validation rules typically assume a dictionary structure
                  for complex data.
            context: A dictionary containing operational context, including
                     'validation_rules' for this node.

        Returns:
            The validated data. If validation passes, the original data is
            returned.

        Raises:
            DataValidationError: If the data fails any of the specified
                                 validation rules.
            ValueError: If 'validation_rules' are missing or malformed in context.
        """
        logger.info("Starting data validation process.")
        
        validation_rules: Union[Dict[str, Any], None] = context.get("validation_rules")

        if not validation_rules:
            logger.error("Validation rules not found in context for DataValidatorNode.")
            raise ValueError("DataValidatorNode requires 'validation_rules' in context.")
        
        try:
            if isinstance(data, dict):
                self._validate_dict_data(data, validation_rules)
            else:
                # If data is not a dict, rules should apply to the data itself
                # Expect rules like {'type': 'str', 'min_length': 1} directly in validation_rules
                self._validate_single_value_data(data, validation_rules)
            
            logger.info("Data validated successfully.")
            return data
        except DataValidationError as e:
            logger.warning(f"Data validation failed: {e}")
            raise
        except Exception as e:
            logger.exception(f"An unexpected error occurred during data validation: {e}")
            raise DataValidationError(f"Unexpected error during validation: {e}")

    def _validate_dict_data(self, data: Dict[str, Any], rules: Dict[str, Any]) -> None:
        """Internal method to validate dictionary-structured data."""
        logger.debug(f"Validating dictionary data against rules: {rules}")
        
        for field_name, field_rules in rules.items():
            field_value = data.get(field_name)
            
            is_required = field_rules.get('required', False)
            if is_required and field_value is None:
                raise DataValidationError(f"Field '{field_name}' is required but missing.")
            
            if field_value is not None:
                self._apply_field_rules(field_name, field_value, field_rules)

        # Check for unexpected fields if strict validation is needed
        # (currently not implemented, but can be added via a 'strict' rule)
        # for key in data.keys():
        #     if key not in rules:
        #         raise DataValidationError(f"Unexpected field '{key}' found in data.")

    def _validate_single_value_data(self, data: Any, rules: Dict[str, Any]) -> None:
        """Internal method to validate non-dictionary data."""
        logger.debug(f"Validating single value data '{data}' against rules: {rules}")
        
        is_required = rules.get('required', False)
        if is_required and data is None:
            raise DataValidationError("Data is required but is None.")
        
        if data is not None:
            self._apply_field_rules("root_data", data, rules)


    def _apply_field_rules(self, field_name: str, value: Any, rules: Dict[str, Any]) -> None:
        """Applies individual validation rules to a field's value."""
        expected_type = rules.get('type')
        if expected_type:
            type_map = {
                'str': str, 'int': int, 'float': float, 'bool': bool, 'list': list, 'dict': dict,
                'any': Any # For when type doesn't strictly matter but other rules do
            }
            if expected_type not in type_map:
                logger.warning(f"Unsupported type '{expected_type}' specified for field '{field_name}'. Skipping type validation.")
            elif not isinstance(value, type_map[expected_type]):
                raise DataValidationError(f"Field '{field_name}' must be of type '{expected_type}', but got '{type(value).__name__}'.")

        if isinstance(value, str):
            min_length = rules.get('min_length')
            max_length = rules.get('max_length')
            if min_length is not None and len(value) < min_length:
                raise DataValidationError(f"Field '{field_name}' must have a minimum length of {min_length}.")
            if max_length is not None and len(value) > max_length:
                raise DataValidationError(f"Field '{field_name}' must have a maximum length of {max_length}.")
            
            regex_pattern = rules.get('regex')
            if regex_pattern and not re.fullmatch(regex_pattern, value):
                raise DataValidationError(f"Field '{field_name}' does not match the required pattern '{regex_pattern}'.")

        elif isinstance(value, (int, float)):
            min_value = rules.get('min_value')
            max_value = rules.get('max_value')
            if min_value is not None and value < min_value:
                raise DataValidationError(f"Field '{field_name}' must have a minimum value of {min_value}.")
            if max_value is not None and value > max_value:
                raise DataValidationError(f"Field '{field_name}' must have a maximum value of {max_value}.")
        
        elif isinstance(value, list):
            min_items = rules.get('min_items')
            max_items = rules.get('max_items')
            item_type = rules.get('item_type')

            if min_items is not None and len(value) < min_items:
                raise DataValidationError(f"Field '{field_name}' must contain at least {min_items} items.")
            if max_items is not None and len(value) > max_items:
                raise DataValidationError(f"Field '{field_name}' must contain at most {max_items} items.")

            if item_type:
                item_type_map = {
                    'str': str, 'int': int, 'float': float, 'bool': bool, 'dict': dict, 'any': Any
                }
                if item_type not in item_type_map:
                    logger.warning(f"Unsupported item_type '{item_type}' specified for list field '{field_name}'. Skipping item type validation.")
                else:
                    for i, item in enumerate(value):
                        if not isinstance(item, item_type_map[item_type]):
                            raise DataValidationError(f"Item at index {i} in '{field_name}' must be of type '{item_type}', but got '{type(item).__name__}'.")
        
        # Add more type-specific validations here as needed (e.g., dict specific rules)
        logger.debug(f"Field '{field_name}' validated against rules: {rules}")
