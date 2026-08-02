import logging
from typing import Any, Dict, List, Callable, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception raised when data fails validation."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra node responsible for validating input data against a set of predefined rules.

    This node takes a list of validation rules during initialization. Each rule specifies
    a `field_path` (dot-separated string for nested access), a `validator` function
    (a callable that returns True for valid data, False otherwise), and an `error_message`
    to be used if validation fails.

    If any validation rule fails, a `ValidationError` is raised, stopping the processing
    flow and providing detailed error messages.

    Example validation rule structure:
    [
        {
            "field_path": "request.user_id",
            "validator": lambda x: isinstance(x, str) and len(x) > 0,
            "error_message": "User ID must be a non-empty string."
        },
        {
            "field_path": "payload.age",
            "validator": lambda x: isinstance(x, int) and 0 < x < 120,
            "error_message": "Age must be an integer between 1 and 119."
        },
        {
            "field_path": "payload.email",
            "validator": lambda x: "@" in x if isinstance(x, str) else False,
            "error_message": "Invalid email format."
        }
    ]
    """

    def __init__(self, validation_rules: List[Dict[str, Any]]):
        """
        Initializes the DataValidatorNode with a list of validation rules.

        Args:
            validation_rules: A list of dictionaries, where each dictionary
                              represents a single validation rule.
                              Each rule dict must contain:
                              - "field_path" (str): A dot-separated string indicating
                                                    the path to the value within the data.
                              - "validator" (Callable[[Any], bool]): A function that takes
                                                                     the field's value and
                                                                     returns True if valid, False otherwise.
                              - "error_message" (str): The message to include if validation fails.
        """
        if not isinstance(validation_rules, list):
            raise TypeError("validation_rules must be a list of dictionaries.")
        for i, rule in enumerate(validation_rules):
            if not isinstance(rule, dict):
                raise TypeError(f"Validation rule at index {i} must be a dictionary.")
            if not all(k in rule for k in ["field_path", "validator", "error_message"]):
                raise ValueError(
                    f"Validation rule at index {i} is missing one of "
                    "'field_path', 'validator', or 'error_message'."
                )
            if not isinstance(rule["field_path"], str) or not rule["field_path"]:
                raise ValueError(f"Validation rule at index {i}: 'field_path' must be a non-empty string.")
            if not callable(rule["validator"]):
                raise TypeError(f"Validation rule at index {i}: 'validator' must be a callable function.")
            if not isinstance(rule["error_message"], str) or not rule["error_message"]:
                raise ValueError(f"Validation rule at index {i}: 'error_message' must be a non-empty string.")

        self.validation_rules = validation_rules
        logger.debug(f"DataValidatorNode initialized with {len(self.validation_rules)} rules.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def _get_nested_value(self, data: Any, path: str) -> Any:
        """
        Safely retrieves a nested value from a dictionary using a dot-separated path.
        Returns None if any part of the path is not found.
        """
        parts = path.split('.')
        current = data
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None  # Path diverged from dict, cannot proceed
            if current is None and part != parts[-1]:
                # An intermediate part of the path was None, implies not found
                return None
        return current

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against configured rules.

        Args:
            data: The input data to be validated. Expected to be a dictionary
                  if validation rules are configured.
            context: A dictionary holding context information for the current processing flow.
                     Not directly used for validation logic in this node, but passed through.

        Returns:
            The original data if all validations pass.

        Raises:
            TypeError: If `data` is not a dictionary when validation rules are present.
            ValidationError: If any validation rule fails.
        """
        if not self.validation_rules:
            logger.debug("No validation rules configured for DataValidatorNode. Returning data as is.")
            return data

        if not isinstance(data, dict):
            error_msg = (
                f"DataValidatorNode expects dictionary input when validation rules are present, "
                f"but received type: {type(data).__name__}. Data: {data!r}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        validation_errors: List[str] = []

        for rule in self.validation_rules:
            field_path = rule["field_path"]
            validator = rule["validator"]
            error_message = rule["error_message"]

            try:
                field_value = self._get_nested_value(data, field_path)

                if not validator(field_value):
                    validation_errors.append(
                        f"Validation failed for field '{field_path}': {error_message} "
                        f"(Value: {field_value!r})"
                    )
                    logger.warning(
                        f"Validation failed for '{field_path}'. Rule message: {error_message}. "
                        f"Current value: {field_value!r}"
                    )
            except Exception as e:
                # Catch exceptions during validator execution itself
                validation_errors.append(
                    f"An error occurred while validating field '{field_path}': {type(e).__name__} - {e}. "
                    f"(Rule message: {error_message})"
                )
                logger.exception(
                    f"Error executing validator for field '{field_path}'. Rule message: {error_message}."
                )

        if validation_errors:
            full_error_msg = "Data validation failed:\n" + "\n".join(
                [f"- {err}" for err in validation_errors]
            )
            logger.error(full_error_msg)
            raise ValidationError(full_error_msg)

        logger.debug("Data passed all validations successfully.")
        return data