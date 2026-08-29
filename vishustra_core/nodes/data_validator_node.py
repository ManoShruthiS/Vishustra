import logging
from typing import Any, Dict, Callable, List, Optional

from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class DataValidationException(ValueError):
    """
    Custom exception raised when data fails one or more validation rules within the
    DataValidatorNode.

    This exception can collect multiple error messages if the node is configured
    not to fail on the first error.
    """
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors if errors is not None else []
        logger.error(f"DataValidationException raised: {message}. Errors: {self.errors}")

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that performs data validation based on a list
    of configurable rules.

    Each validation rule is a callable that takes the data and context, and
    raises a ValueError (or a subclass) if the data is invalid for that rule.
    If all rules pass, the original data is returned.
    """

    def __init__(self, validation_rules: List[Callable[[Any, Dict[str, Any]], None]], fail_on_first_error: bool = True):
        """
        Initializes the DataValidatorNode with a set of validation rules.

        Each validation rule is expected to be a callable with the signature:
        `def rule_function(data: Any, context: Dict[str, Any]) -> None:`

        If a rule finds an issue with the data, it should raise a `ValueError`
        (or a more specific subclass of `ValueError`) with a descriptive
        message. If the data passes the rule, the callable should simply return.

        Args:
            validation_rules: A list of callable validation functions.
            fail_on_first_error: If True, the node stops processing rules and
                                 raises `DataValidationException` immediately
                                 upon the first validation failure. If False,
                                 it attempts to run all rules and collects all
                                 error messages before raising a single
                                 `DataValidationException`.

        Raises:
            TypeError: If `validation_rules` is not a list or contains non-callable elements.
        """
        if not isinstance(validation_rules, list):
            raise TypeError("`validation_rules` must be a list of callables.")
        if not all(callable(rule) for rule in validation_rules):
            raise TypeError("All elements in `validation_rules` must be callable.")

        self._validation_rules = validation_rules
        self._fail_on_first_error = fail_on_first_error
        logger.debug(
            f"DataValidatorNode initialized with {len(validation_rules)} rules. "
            f"Fail on first error: {fail_on_first_error}"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying all configured validation rules.

        If any rule fails, a `DataValidationException` is raised, potentially
        containing details about all failed validations if `fail_on_first_error`
        is set to False.

        Args:
            data: The input data to be validated.
            context: A dictionary containing contextual information for the node,
                     which can be used by validation rules.

        Returns:
            The original `data` if all validations pass successfully.

        Raises:
            DataValidationException: If the data fails one or more validation rules.
        """
        node_id = context.get('node_id', self.node_name)
        logger.info(f"Node '{node_id}' ({self.node_name}) starting data validation.")

        failed_validations: List[str] = []
        
        for i, rule in enumerate(self._validation_rules):
            try:
                rule(data, context)
                logger.debug(f"Rule {i+1}/{len(self._validation_rules)} passed for node '{node_id}'.")
            except ValueError as e:
                # Catch specific validation errors raised by the rules
                error_msg = f"Validation rule {i+1} failed for node '{node_id}': {e}"
                logger.warning(error_msg)
                failed_validations.append(error_msg)
                if self._fail_on_first_error:
                    logger.error(
                        f"Validation for node '{node_id}' stopped due to 'fail_on_first_error' policy."
                    )
                    raise DataValidationException(f"Data validation failed: {error_msg}") from e
            except Exception as e:
                # Catch any unexpected errors during rule execution (e.g., programming errors in rules)
                error_msg = (f"An unexpected error occurred during validation rule {i+1} "
                             f"for node '{node_id}': {type(e).__name__}: {e}")
                logger.exception(error_msg) # Log with full traceback for unexpected exceptions
                failed_validations.append(error_msg)
                if self._fail_on_first_error:
                    logger.error(
                        f"Validation for node '{node_id}' stopped due to unexpected error and "
                        f"'fail_on_first_error' policy."
                    )
                    raise DataValidationException(f"Data validation failed unexpectedly: {error_msg}") from e

        if failed_validations:
            full_error_message = (
                f"Data validation failed for node '{node_id}' with {len(failed_validations)} error(s)."
            )
            logger.error(
                f"Validation for node '{node_id}' completed with errors. "
                f"Total failures: {len(failed_validations)}"
            )
            raise DataValidationException(full_error_message, errors=failed_validations)

        logger.info(f"Node '{node_id}' ({self.node_name}) successfully validated data.")
        return data