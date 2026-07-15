import logging
from typing import Any, Dict, Callable, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationException(Exception):
    """Custom exception raised for data validation failures within the DataValidatorNode."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against
    a set of predefined rules.

    This node ensures data integrity and adherence to expected schemas before
    data proceeds further in the orchestration pipeline. Validation rules are
    provided during node initialization.
    """

    def __init__(self, validation_rules: Dict[str, List[Callable[[Any], bool]]]):
        """
        Initializes the DataValidatorNode with a set of validation rules.

        Args:
            validation_rules: A dictionary where keys are field names expected in the
                              input data, and values are lists of callable validation
                              functions. Each validation function should accept a single
                              argument (the field's value) and return `True` if the
                              value is valid, and `False` otherwise.
                              Example:
                              {
                                  "user_id": [lambda x: isinstance(x, int), lambda x: x > 0],
                                  "email": [lambda x: isinstance(x, str), lambda x: "@" in x and "." in x]
                              }
        Raises:
            TypeError: If `validation_rules` is not a dictionary.
        """
        if not isinstance(validation_rules, dict):
            raise TypeError("validation_rules must be a dictionary mapping field names to lists of callables.")
        
        # Ensure all values in the rules dictionary are lists of callables
        for field, rules in validation_rules.items():
            if not isinstance(rules, list):
                raise TypeError(f"Validation rules for field '{field}' must be a list of callables, got {type(rules)}.")
            for i, rule_func in enumerate(rules):
                if not callable(rule_func):
                    raise TypeError(f"Rule at index {i} for field '{field}' is not a callable.")

        self._validation_rules = validation_rules
        logger.debug(f"[{self.node_name}] Initialized with rules for fields: {list(validation_rules.keys())}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "Data Validator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured rules.

        The node iterates through the configured `validation_rules`, applies
        each rule to the corresponding field in the input `data`. If any rule
        fails for any field, a `DataValidationException` is raised.

        Args:
            data: The input data to be validated. Expected to be a dictionary
                  if validation rules are defined for specific fields.
            context: A dictionary containing contextual information for processing.
                     (Currently not used by this node but part of the BaseNode interface).

        Returns:
            The original `data` object if all validations pass successfully.

        Raises:
            TypeError: If the input `data` is not a dictionary and validation rules
                       are configured, or if any rule is malformed.
            DataValidationException: If one or more validation rules fail for the data.
            Exception: For unexpected errors during rule execution.
        """
        logger.info(f"[{self.node_name}] Starting data validation process.")

        if self._validation_rules and not isinstance(data, dict):
            logger.error(
                f"[{self.node_name}] Input data is not a dictionary but validation rules are configured. "
                f"Expected dictionary for field-based validation. Received type: {type(data)}"
            )
            raise TypeError(
                f"DataValidatorNode expects dictionary data when validation rules are provided, "
                f"but received {type(data)}."
            )
        
        if not self._validation_rules:
            logger.debug(f"[{self.node_name}] No validation rules configured. Passing data through without validation.")
            return data

        failed_validations: Dict[str, List[str]] = {}

        for field_name, rules in self._validation_rules.items():
            if field_name not in data:
                # Decide policy for missing fields. For now, assume a field not present
                # simply means its rules aren't applied. If a field is strictly required,
                # a specific rule (e.g., `lambda x: x is not None`) should be added to the field's rules.
                logger.debug(f"[{self.node_name}] Field '{field_name}' not found in input data. Skipping its validation rules.")
                continue

            field_value = data[field_name]
            for i, rule_func in enumerate(rules):
                rule_identifier = rule_func.__name__ if hasattr(rule_func, '__name__') and rule_func.__name__ != '<lambda>' else f"anonymous_rule_{i+1}"
                try:
                    is_valid = rule_func(field_value)
                    if not is_valid:
                        if field_name not in failed_validations:
                            failed_validations[field_name] = []
                        failed_validations[field_name].append(
                            f"Rule '{rule_identifier}' failed for value: '{field_value}'."
                        )
                        logger.warning(
                            f"[{self.node_name}] Validation failed for field '{field_name}' "
                            f"with value '{field_value}'. Rule: '{rule_identifier}'."
                        )
                except Exception as e:
                    # Catch exceptions during rule execution to prevent node failure
                    # and report them as validation failures.
                    if field_name not in failed_validations:
                        failed_validations[field_name] = []
                    failed_validations[field_name].append(
                        f"Rule '{rule_identifier}' encountered an error: {e.__class__.__name__}: {e}"
                    )
                    logger.error(
                        f"[{self.node_name}] Error executing rule '{rule_identifier}' "
                        f"for field '{field_name}' with value '{field_value}': {e}", 
                        exc_info=True
                    )

        if failed_validations:
            error_summary = {field: errors for field, errors in failed_validations.items()}
            error_message = f"Data validation failed. Detailed errors: {error_summary}"
            logger.error(f"[{self.node_name}] {error_message}")
            raise DataValidationException(error_message)
        else:
            logger.info(f"[{self.node_name}] Data validation successful. All rules passed.")
            return data