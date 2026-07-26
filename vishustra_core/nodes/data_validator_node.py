import logging
from typing import Any, Dict, Callable

# Vishustra framework specific import
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation failures within the DataValidatorNode."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node designed to validate input data against a set of predefined rules.

    This node is crucial for ensuring data integrity and adherence to expected schemas
    before data proceeds further in the orchestration pipeline. It can be configured
    with a dictionary of validation rules, where each rule is a callable function
    that takes a data field's value and returns True if the value is valid, False otherwise.

    If any validation rule fails or a required field is missing, the node raises
    a ValidationError, halting the pipeline and signaling a data integrity issue.
    """

    def __init__(self, validation_rules: Dict[str, Callable[[Any], bool]]):
        """
        Initializes the DataValidatorNode with specific validation rules.

        Args:
            validation_rules (Dict[str, Callable[[Any], bool]]): A dictionary
                where keys represent expected field names in the input data (strings),
                and values are callable validation functions. Each validation function
                should accept the field's value as an argument and return a boolean:
                `True` if the value passes validation, `False` otherwise.
                An empty dictionary means no specific field-level validation will be performed.
        
        Raises:
            TypeError: If `validation_rules` is not a dictionary or if any rule
                       within it is not a callable.
        """
        if not isinstance(validation_rules, dict):
            raise TypeError("Validation rules must be provided as a dictionary.")
        
        for field, rule in validation_rules.items():
            if not callable(rule):
                raise TypeError(f"Validation rule for field '{field}' must be a callable function.")
        
        self._validation_rules = validation_rules
        logger.info(f"DataValidatorNode initialized with {len(self._validation_rules)} validation rules.")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured rules.

        The node iterates through its `_validation_rules`. For each rule, it checks
        if the corresponding field exists in the input `data` (which is expected
        to be a dictionary for field-specific rules). If a field is missing or its
        value fails the associated validation function, a `ValidationError` is raised.

        Args:
            data (Any): The input data to be validated. For field-specific
                        validation rules, this is expected to be a dictionary.
            context (Dict[str, Any]): The current processing context, which can
                                     be used for logging or passing along metadata.

        Returns:
            Any: The original, unmodified data if all validations pass successfully.

        Raises:
            TypeError: If the input `data` is not a dictionary and field-specific
                       validation rules are present.
            ValidationError: If any configured validation rule fails (e.g., a
                             required field is missing, or a field's value does
                             not satisfy its validation function), or if a rule
                             itself raises an unexpected exception.
        """
        if not self._validation_rules:
            logger.debug("No specific field validation rules configured for DataValidator. Passing data through.")
            return data

        if not isinstance(data, dict):
            error_msg = (f"DataValidatorNode expects input data to be a dictionary for field-specific "
                         f"validation but received type: {type(data).__name__}.")
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.debug(f"Initiating validation for node '{self.node_name}' on data with keys: {list(data.keys())}")

        for field_name, validator_func in self._validation_rules.items():
            # Check for existence of the field
            if field_name not in data:
                error_msg = f"Validation failed for node '{self.node_name}': Required field '{field_name}' is missing in the input data."
                logger.warning(error_msg)
                raise ValidationError(error_msg)

            field_value = data[field_name]
            try:
                is_valid = validator_func(field_value)
                if not is_valid:
                    error_msg = (f"Validation failed for node '{self.node_name}', field '{field_name}': "
                                 f"Value '{field_value!r}' did not pass the configured validation rule.")
                    logger.warning(error_msg)
                    raise ValidationError(error_msg)
                else:
                    logger.debug(f"Validation successful for field '{field_name}'.")
            except Exception as e:
                # Catch any unexpected errors within the validator function itself
                error_msg = (f"An unexpected error occurred while applying validation rule for field "
                             f"'{field_name}' in node '{self.node_name}': {e}")
                logger.error(error_msg, exc_info=True)
                raise ValidationError(error_msg) from e

        logger.info(f"All configured validation rules passed successfully for node '{self.node_name}'. Data is considered valid.")
        return data