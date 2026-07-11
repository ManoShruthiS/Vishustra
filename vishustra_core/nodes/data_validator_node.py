import logging
from typing import Any, Dict, Type, Union

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception raised when data fails validation against a defined schema."""
    pass

class DataValidatorNode(BaseNode):
    """
    A processing node that validates input data against a predefined schema.

    This node ensures that incoming data conforms to expected types and presence
    of required fields, preventing malformed data from propagating further
    in the orchestration pipeline.

    The validation schema is a dictionary where keys are field names and values
    are dictionaries defining validation rules for that field.
    Supported rules:
    - 'type': The expected Python type (e.g., str, int, bool, float).
    - 'required': A boolean indicating if the field must be present (default: False).
    """

    def __init__(self, validation_schema: Dict[str, Dict[str, Union[Type, bool]]]):
        """
        Initializes the DataValidatorNode with a specific validation schema.

        Args:
            validation_schema: A dictionary defining the validation rules.
                               Example:
                               {
                                   "user_id": {"type": int, "required": True},
                                   "username": {"type": str, "required": True},
                                   "email": {"type": str, "required": False},
                                   "age": {"type": int, "required": False},
                               }
        """
        if not isinstance(validation_schema, dict):
            raise TypeError("validation_schema must be a dictionary.")
        self._validation_schema = validation_schema
        logger.debug(f"DataValidatorNode initialized with schema: {self._validation_schema}")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured schema.

        Args:
            data: The data payload to be validated. Expected to be a dictionary.
            context: A dictionary containing contextual information for the node.

        Returns:
            The original data payload if validation is successful.

        Raises:
            ValidationError: If the data does not conform to the validation schema.
            TypeError: If the input 'data' is not a dictionary.
        """
        node_id = context.get('node_id', self.node_name)
        logger.info(f"[{node_id}] Starting data validation process.")

        if not isinstance(data, dict):
            error_msg = f"[{node_id}] Input data for validation must be a dictionary, but received {type(data).__name__}."
            logger.error(error_msg)
            raise TypeError(error_msg)

        validation_errors = []

        for field_name, rules in self._validation_schema.items():
            is_required = rules.get("required", False)
            expected_type = rules.get("type")

            if is_required and field_name not in data:
                validation_errors.append(f"Required field '{field_name}' is missing.")
                continue # Skip type check if field is missing

            if field_name in data:
                field_value = data[field_name]
                if expected_type and not isinstance(field_value, expected_type):
                    validation_errors.append(
                        f"Field '{field_name}' has type {type(field_value).__name__}, "
                        f"but expected {expected_type.__name__}."
                    )

        if validation_errors:
            full_error_msg = f"[{node_id}] Data validation failed:\n" + "\n".join(validation_errors)
            logger.error(full_error_msg)
            raise ValidationError(full_error_msg)
        else:
            logger.info(f"[{node_id}] Data validation successful. Returning data.")
            return data