import logging
from typing import Any, Dict, List, Type, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class SchemaValidatorNode(BaseNode):
    """
    A data processing node responsible for validating the structure and types
    of incoming data against a predefined schema.

    This node ensures that required fields are present and that field values
    conform to their expected Python types, preventing downstream processing
    errors caused by malformed or incomplete data.

    Validation rules are typically provided via the 'context' dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "Schema Validator Node"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against a set of schema rules defined in the context.

        The validation rules can be specified in the `context` dictionary
        under the key 'validation_config'. This dictionary should contain:
        - 'required_fields': List[str] - A list of field names that must be present.
        - 'field_types': Dict[str, Type] - A dictionary mapping field names to
          their expected Python types (e.g., {'name': str, 'age': int}).
          Type hints like `Union[str, int]` are supported for field types.

        If data fails validation, a ValueError is raised with details about the failures.
        Otherwise, the original, validated data is returned.

        Args:
            data: The input data to be validated. Expected to be a dictionary.
            context: A dictionary containing contextual information, including
                     validation rules under the 'validation_config' key.

        Returns:
            The original data if it passes all validation checks.

        Raises:
            TypeError: If the input 'data' is not a dictionary.
            ValueError: If 'data' fails any schema validation rule (missing fields,
                        incorrect types, etc.).
        """
        logger.debug(f"[{self.node_name}] Starting data validation for data of type: {type(data)}")

        if not isinstance(data, dict):
            logger.error(f"[{self.node_name}] Input data is not a dictionary. Type found: {type(data)}")
            raise TypeError(
                f"[{self.node_name}] Invalid input data type. Expected a dictionary, received {type(data).__name__}."
            )

        validation_config: Dict[str, Any] = context.get('validation_config', {})
        required_fields: List[str] = validation_config.get('required_fields', [])
        field_types: Dict[str, Type] = validation_config.get('field_types', {})

        validation_errors: List[str] = []

        # Validate required fields
        for field in required_fields:
            if field not in data:
                validation_errors.append(f"Missing required field: '{field}'")
                logger.debug(f"[{self.node_name}] Validation Error: Missing required field '{field}'.")

        # Validate field types
        for field, expected_type in field_types.items():
            if field in data:
                actual_value = data[field]
                
                # Check for Union types correctly as isinstance doesn't handle them directly
                if hasattr(expected_type, '__origin__') and expected_type.__origin__ is Union:
                    union_types = expected_type.__args__  # e.g., (str, int) for Union[str, int]
                    if not any(isinstance(actual_value, t) for t in union_types if isinstance(t, type)): # filter out NoneType if Union[T, None]
                        validation_errors.append(
                            f"Field '{field}' has incorrect type. Expected one of {union_types}, got {type(actual_value).__name__}."
                        )
                        logger.debug(f"[{self.node_name}] Validation Error: Field '{field}' has incorrect type. Expected one of {union_types}, got {type(actual_value).__name__}.")
                elif not isinstance(actual_value, expected_type):
                    validation_errors.append(
                        f"Field '{field}' has incorrect type. Expected {expected_type.__name__}, got {type(actual_value).__name__}."
                    )
                    logger.debug(f"[{self.node_name}] Validation Error: Field '{field}' has incorrect type. Expected {expected_type.__name__}, got {type(actual_value).__name__}.")
            elif field in required_fields:
                # This case means a required field also has a type constraint but was missing.
                # The 'missing required field' error already covers this.
                pass 
            else:
                # Field with a type constraint is not present and not required, which is fine.
                pass

        if validation_errors:
            error_msg = f"[{self.node_name}] Data validation failed with {len(validation_errors)} errors: {'; '.join(validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"[{self.node_name}] Data successfully validated. All schema checks passed.")
        return data