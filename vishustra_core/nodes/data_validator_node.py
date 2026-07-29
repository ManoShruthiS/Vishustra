import logging
from typing import Any, Dict, Type, Union

# Assuming vishustra_core is a package and nodes is a subpackage
# and base_node is a module within nodes.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidator(BaseNode):
    """
    A Vishustra processing node that validates input data against a set of predefined rules.

    This node is crucial for ensuring data integrity and consistency throughout
    the orchestration pipeline. It can check for required fields and data types,
    and manage nullable values.

    Validation rules are defined as a dictionary where keys represent the data fields
    to validate. Each field's rule can specify:
    - 'required': bool (default: False) - If True, the field must be present in the data.
    - 'type': Type (default: Any) - The expected Python type of the field's value.
                                   Must be a concrete type (e.g., str, int, dict).
    - 'allow_none': bool (default: False) - If True, allows a None value for the field.
                                            This overrides type checking if the value is None.

    Example validation_rules:
    {
        "user_id": {"type": str, "required": True},
        "email": {"type": str, "required": True, "allow_none": False},
        "age": {"type": int, "required": False, "allow_none": True},
        "preferences": {"type": dict, "required": True, "allow_none": True}
    }
    """

    def __init__(self, validation_rules: Dict[str, Dict[str, Any]]):
        """
        Initializes the DataValidator node with specific validation rules.

        Args:
            validation_rules: A dictionary defining the validation schema.
                              Keys are field names, values are dictionaries
                              specifying 'required' (bool), 'type' (Type),
                              and 'allow_none' (bool).

        Raises:
            TypeError: If `validation_rules` is not a dictionary or if a type in
                       rules is not a valid Python type object.
            ValueError: If the structure of a rule is invalid.
        """
        if not isinstance(validation_rules, dict):
            raise TypeError("Validation rules must be a dictionary.")
        
        # Validate the structure and types within validation_rules
        for field, rules in validation_rules.items():
            if not isinstance(rules, dict):
                raise ValueError(f"Rules for field '{field}' must be a dictionary.")
            
            # Validate 'type' entry
            if 'type' in rules and not (isinstance(rules['type'], type) or rules['type'] is Any):
                raise TypeError(
                    f"The 'type' specified for field '{field}' must be a Python type "
                    f"(e.g., str, int, dict) or typing.Any. Got {type(rules['type']).__name__}."
                )
            
            # Validate 'required' entry
            if 'required' in rules and not isinstance(rules['required'], bool):
                raise TypeError(f"The 'required' flag for field '{field}' must be a boolean.")
            
            # Validate 'allow_none' entry
            if 'allow_none' in rules and not isinstance(rules['allow_none'], bool):
                raise TypeError(f"The 'allow_none' flag for field '{field}' must be a boolean.")

        self._validation_rules = validation_rules
        logger.debug(f"DataValidator node initialized with rules: {validation_rules}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured rules.

        Args:
            data: The input data to be validated. Expected to be a dictionary.
            context: A dictionary containing context-specific information for the node.

        Returns:
            The original data if validation passes.

        Raises:
            TypeError: If the input 'data' is not a dictionary or if a field's value
                       does not match its specified type.
            ValueError: If a required field is missing or if a field's value is None
                        when `allow_none` is False.
        """
        logger.info(f"[{self.node_name}] Starting data validation.")

        if not isinstance(data, dict):
            logger.error(
                f"[{self.node_name}] Input data is not a dictionary. "
                "DataValidator is designed for dictionary validation. "
                f"Received type: {type(data).__name__}"
            )
            raise TypeError(
                f"Input data must be a dictionary for DataValidator, "
                f"but received {type(data).__name__}."
            )

        for field_name, rules in self._validation_rules.items():
            is_required = rules.get('required', False)
            expected_type = rules.get('type', Any)
            allow_none = rules.get('allow_none', False)

            if field_name not in data:
                if is_required:
                    logger.warning(
                        f"[{self.node_name}] Required field '{field_name}' is missing from data."
                    )
                    raise ValueError(f"Required field '{field_name}' is missing.")
                else:
                    # If not required and not present, skip further validation for this field
                    continue

            # Field is present, now validate its value
            field_value = data[field_name]

            if field_value is None:
                if not allow_none:
                    # If field is present but None, and None is not explicitly allowed.
                    logger.warning(
                        f"[{self.node_name}] Field '{field_name}' received None, but 'allow_none' is False. "
                        f"Expected type: {expected_type.__name__ if expected_type is not Any else 'Any'}."
                    )
                    raise ValueError(
                        f"Field '{field_name}' received None, but 'allow_none' is False "
                        f"(expected type: {expected_type.__name__ if expected_type is not Any else 'Any'})."
                    )
                else:
                    # None is allowed for this field, so it passes validation for this specific rule.
                    logger.debug(
                        f"[{self.node_name}] Field '{field_name}' is None, which is allowed by rules."
                    )
                    continue # Skip type check for None if allowed

            # If field_value is not None, proceed with type check
            if expected_type is not Any and not isinstance(field_value, expected_type):
                logger.warning(
                    f"[{self.node_name}] Field '{field_name}' has incorrect type. "
                    f"Expected '{expected_type.__name__}', got '{type(field_value).__name__}'."
                )
                raise TypeError(
                    f"Field '{field_name}' type mismatch: "
                    f"expected '{expected_type.__name__}', got '{type(field_value).__name__}'."
                )
            
            logger.debug(f"[{self.node_name}] Field '{field_name}' validated successfully.")

        logger.info(f"[{self.node_name}] All data validation checks passed.")
        return data
