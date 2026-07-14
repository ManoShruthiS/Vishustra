import logging
from typing import Any, Dict, Callable, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception raised when input data fails validation against defined rules."""
    def __init__(self, message: str, field_name: str = None, value: Any = None):
        """
        Initializes a ValidationError.

        Args:
            message: A human-readable description of the validation failure.
            field_name: The name of the data field that failed validation (if applicable).
            value: The value of the data field that failed validation (if applicable).
        """
        super().__init__(message)
        self.field_name = field_name
        self.value = value
    
    def __str__(self):
        """String representation of the error."""
        details = ""
        if self.field_name:
            details += f" (Field: '{self.field_name}')"
        if self.value is not None:
            details += f" (Value: '{self.value}')"
        return f"{super().__str__()}{details}"

class DataValidator(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against a
    set of predefined rules.

    This node is designed to ensure that data conforms to expected types, formats,
    or business logic before being processed by subsequent nodes in an orchestration
    pipeline. If any validation rule fails, a `ValidationError` is raised,
    halting processing and signaling an issue with the input data.
    """

    def __init__(self, validation_rules: Dict[str, Union[Callable[[Any], bool], List[Callable[[Any], bool]]]]):
        """
        Initializes the DataValidator node with a dictionary of validation rules.

        Args:
            validation_rules: A dictionary defining the validation constraints.
                              - Keys are string names of the data fields to be validated.
                              - Values can be either:
                                - A single callable (validation function) that accepts
                                  the field's value as an argument and returns `True` if
                                  the value is valid, `False` otherwise.
                                - A list of such callables, where all functions in the
                                  list must return `True` for the field to be considered valid.

        Raises:
            TypeError: If `validation_rules` is not a dictionary or contains improperly
                       formatted field names or validation functions.
        """
        if not isinstance(validation_rules, dict):
            raise TypeError("`validation_rules` must be a dictionary.")
        
        for field, rules in validation_rules.items():
            if not isinstance(field, str):
                raise TypeError(f"Validation rule key '{field}' must be a string (field name).")
            
            # Check if rules is a callable or a list of callables
            if not callable(rules) and not (isinstance(rules, list) and all(callable(r) for r in rules)):
                raise TypeError(
                    f"Validation rule for field '{field}' must be a callable or a list of callables."
                )
        
        self.validation_rules = validation_rules
        logger.debug(f"DataValidator node initialized. Monitoring fields: {list(validation_rules.keys())}")

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured rules.

        This method iterates through the `validation_rules` configured during
        initialization. For each specified field, it applies the associated
        validation functions. If any rule fails (returns `False`) or raises
        an exception, a `ValidationError` is raised.

        Args:
            data: The input data to be validated. Expected to be a dictionary,
                  as rules are applied to named fields.
            context: A dictionary containing contextual information relevant to
                     the current pipeline execution. While this node primarily
                     validates `data`, custom validation functions could potentially
                     leverage `context` if designed to do so.

        Returns:
            The original, unchanged `data` dictionary if all validations pass.

        Raises:
            TypeError: If the input `data` is not a dictionary.
            ValidationError: If any configured validation rule fails or a
                             validation function itself raises an exception.
        """
        if not isinstance(data, dict):
            logger.error(f"[{self.node_name}] Input 'data' must be a dictionary for field-based validation. Got: {type(data)}")
            raise TypeError(
                f"DataValidator expects input 'data' to be a dictionary, but received {type(data)}."
            )

        logger.info(f"[{self.node_name}] Starting validation of data fields...")

        for field_name, rules_config in self.validation_rules.items():
            if field_name not in data:
                error_msg = f"Required field '{field_name}' is missing from the input data."
                logger.warning(f"[{self.node_name}] {error_msg}")
                raise ValidationError(error_msg, field_name=field_name)

            field_value = data[field_name]
            
            # Ensure rules_to_apply is always a list for consistent iteration
            rules_to_apply = [rules_config] if callable(rules_config) else rules_config

            for i, rule in enumerate(rules_to_apply):
                try:
                    is_valid = rule(field_value)
                except Exception as e:
                    error_msg = (
                        f"Validation rule #{i+1} for field '{field_name}' raised an unexpected exception: {e}"
                    )
                    logger.error(f"[{self.node_name}] {error_msg}", exc_info=True)
                    raise ValidationError(error_msg, field_name=field_name, value=field_value) from e

                if not is_valid:
                    error_msg = (
                        f"Validation rule #{i+1} failed for field '{field_name}' "
                        f"with value '{field_value}' (type: {type(field_value).__name__})."
                    )
                    logger.warning(f"[{self.node_name}] {error_msg}")
                    raise ValidationError(error_msg, field_name=field_name, value=field_value)

            logger.debug(f"[{self.node_name}] Field '{field_name}' passed all configured validations.")

        logger.info(f"[{self.node_name}] All data validations passed successfully.")
        return data