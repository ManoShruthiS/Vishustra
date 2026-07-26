import logging
from typing import Any, Dict, Type

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a set of
    predefined rules.

    The node expects validation rules in its constructor, which are then
    applied to the 'data' passed to the process method. If data fails
    validation, a ValueError is raised.

    Supported rule types:
    - 'required': bool (Checks if a field must be present)
    - 'type': str (e.g., 'str', 'int', 'float', 'list', 'dict', 'bool', 'any')
    - 'min_length': int (For strings, lists, dicts)
    - 'max_length': int (For strings, lists, dicts)
    - 'min_value': int/float (For numbers)
    - 'max_value': int/float (For numbers)
    """

    # A class-level mapping for converting string type names to actual Python types
    _TYPE_MAP: Dict[str, Type[Any]] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "any": object, # 'object' is the base of all classes, effectively 'Any' for isinstance
    }

    def __init__(self, validation_rules: Dict[str, Dict[str, Any]]):
        """
        Initializes the DataValidatorNode with specific validation rules.

        Args:
            validation_rules: A dictionary where keys are field names and values
                              are dictionaries of rules for that field.
                              Example:
                              {
                                  "name": {"type": "str", "required": True, "min_length": 1},
                                  "age": {"type": "int", "min_value": 0, "max_value": 120, "required": True},
                                  "tags": {"type": "list", "max_length": 5},
                                  "config": {"type": "dict"}
                              }
        """
        if not isinstance(validation_rules, dict):
            raise TypeError("validation_rules must be a dictionary.")
        
        self.validation_rules = validation_rules
        logger.debug(f"[{self.node_name}] Initialized with rules: {self.validation_rules}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, validating it against the configured rules.

        Args:
            data: The input data to be validated. Expected to be a dictionary
                  if validation_rules are defined.
            context: A dictionary containing contextual information for the process.

        Returns:
            The original data if validation is successful.

        Raises:
            TypeError: If the input 'data' is not a dictionary when rules are present,
                       or if a specified type in rules is invalid.
            ValueError: If the 'data' fails any of the validation rules.
        """
        logger.info(f"[{self.node_name}] Starting data validation for incoming data.")
        logger.debug(f"[{self.node_name}] Context: {context}")

        # If no validation rules are configured, skip validation entirely
        if not self.validation_rules:
            logger.info(f"[{self.node_name}] No validation rules configured. Skipping validation.")
            return data

        # If validation rules are present, expect 'data' to be a dictionary for field-based validation
        if not isinstance(data, dict):
            error_msg = (
                f"[{self.node_name}] Data must be a dictionary when validation_rules are "
                f"configured. Got type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        for field_name, rules in self.validation_rules.items():
            # Ensure rules for a field are a dictionary
            if not isinstance(rules, dict):
                error_msg = (
                    f"[{self.node_name}] Configuration error: Rules for field "
                    f"'{field_name}' must be a dictionary."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)

            # 1. Required field check
            is_required = rules.get("required", False)
            if is_required and field_name not in data:
                error_msg = (
                    f"[{self.node_name}] Validation failed: Required field "
                    f"'{field_name}' is missing from data."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # If the field is not present and not required, skip further checks for this field
            if field_name not in data:
                continue

            field_value = data[field_name]

            # 2. Type check
            expected_type_str = rules.get("type")
            if expected_type_str:
                expected_type = self._TYPE_MAP.get(expected_type_str)
                if expected_type is None:
                    error_msg = (
                        f"[{self.node_name}] Configuration error: Unknown type "
                        f"'{expected_type_str}' specified for field '{field_name}'."
                    )
                    logger.error(error_msg)
                    raise TypeError(error_msg)
                
                if not isinstance(field_value, expected_type):
                    error_msg = (
                        f"[{self.node_name}] Validation failed for field '{field_name}': "
                        f"Expected type '{expected_type_str}', but got '{type(field_value).__name__}'."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # 3. Length checks (applicable to str, list, dict)
            if isinstance(field_value, (str, list, dict)):
                current_length = len(field_value)
                min_length = rules.get("min_length")
                max_length = rules.get("max_length")

                if min_length is not None and not isinstance(min_length, int):
                    logger.warning(f"[{self.node_name}] Configuration warning: 'min_length' for '{field_name}' is not an integer. Skipping.")
                elif min_length is not None and current_length < min_length:
                    error_msg = (
                        f"[{self.node_name}] Validation failed for field '{field_name}': "
                        f"Length {current_length} is less than required minimum {min_length}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                if max_length is not None and not isinstance(max_length, int):
                    logger.warning(f"[{self.node_name}] Configuration warning: 'max_length' for '{field_name}' is not an integer. Skipping.")
                elif max_length is not None and current_length > max_length:
                    error_msg = (
                        f"[{self.node_name}] Validation failed for field '{field_name}': "
                        f"Length {current_length} is greater than allowed maximum {max_length}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # 4. Value range checks (applicable to int, float)
            if isinstance(field_value, (int, float)):
                min_value = rules.get("min_value")
                max_value = rules.get("max_value")

                if min_value is not None and not isinstance(min_value, (int, float)):
                    logger.warning(f"[{self.node_name}] Configuration warning: 'min_value' for '{field_name}' is not a number. Skipping.")
                elif min_value is not None and field_value < min_value:
                    error_msg = (
                        f"[{self.node_name}] Validation failed for field '{field_name}': "
                        f"Value {field_value} is less than required minimum {min_value}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                if max_value is not None and not isinstance(max_value, (int, float)):
                    logger.warning(f"[{self.node_name}] Configuration warning: 'max_value' for '{field_name}' is not a number. Skipping.")
                elif max_value is not None and field_value > max_value:
                    error_msg = (
                        f"[{self.node_name}] Validation failed for field '{field_name}': "
                        f"Value {field_value} is greater than allowed maximum {max_value}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Additional validation rules can be extended here (e.g., regex pattern, enum checks)

        logger.info(f"[{self.node_name}] Data successfully validated.")
        return data