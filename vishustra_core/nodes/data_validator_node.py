import logging
from typing import Any, Dict, Callable, List

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationException(ValueError):
    """
    Custom exception raised when data fails validation within the DataValidatorNode.
    Inherits from ValueError for broad compatibility and typical validation error semantics.
    """
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a set of predefined rules.
    
    This node takes a list of validation functions in its constructor. Each function
    should accept the data to be validated as its single argument. If a validation
    rule fails, it should raise a ValueError (or a subclass). The node catches
    these errors and re-raises a DataValidationException, providing detailed
    information about which rule failed.

    If all rules pass, the original data is returned unmodified.
    """
    
    # Using a class attribute for the node name for consistency and readability
    _node_name = "DataValidator" 

    def __init__(self, validation_rules: List[Callable[[Any], None]]):
        """
        Initializes the DataValidatorNode with a list of validation rules.

        Each rule is a callable that takes 'data: Any' as its sole argument.
        If a rule determines the data is invalid, it must raise a ValueError
        (or a subclass thereof) with a descriptive error message.

        Args:
            validation_rules: A list of callables, where each callable represents
                              a single validation check. The order of rules is preserved.

        Raises:
            TypeError: If `validation_rules` is not a list, or if any item
                       in the list is not a callable.
        """
        if not isinstance(validation_rules, list):
            raise TypeError("Validation rules must be provided as a list.")
        
        for i, rule in enumerate(validation_rules):
            if not callable(rule):
                raise TypeError(f"Validation rule at index {i} is not a callable. Got: {type(rule)}")
        
        self._validation_rules = validation_rules
        logger.debug(f"[{self.node_name}] Initialized with {len(validation_rules)} validation rules.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return self._node_name

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying all configured validation rules.

        The method iterates through `_validation_rules`. If any rule raises
        a ValueError, the process stops, logs the error, and re-raises
        a DataValidationException. If any other unexpected exception occurs,
        it is also caught and re-raised as DataValidationException.

        Args:
            data: The input data to be validated. This data is passed to each
                  validation rule.
            context: A dictionary containing contextual information for the
                     current execution flow. Not directly used by this node's
                     validation logic but passed for consistency with BaseNode.
        
        Returns:
            The original `data` argument, unmodified, if all validations pass.

        Raises:
            DataValidationException: If any validation rule fails or an unexpected
                                     error occurs during validation.
        """
        logger.info(f"[{self.node_name}] Starting data validation for incoming data.")
        
        for i, rule in enumerate(self._validation_rules):
            try:
                # Execute the validation rule. It's expected to raise ValueError on failure.
                rule(data)
                logger.debug(f"[{self.node_name}] Rule {i+1}/{len(self._validation_rules)} executed successfully.")
            except ValueError as e:
                # Catch anticipated validation failures (e.g., type mismatch, missing key, invalid range)
                error_msg = (
                    f"[{self.node_name}] Data validation failed at rule {i+1} "
                    f"(Rule Type: {type(rule).__name__}): {e}"
                )
                logger.error(error_msg, exc_info=True) # Log with stack trace for detailed debugging
                raise DataValidationException(error_msg) from e
            except Exception as e:
                # Catch any other unexpected errors that might occur within a validation function
                error_msg = (
                    f"[{self.node_name}] An unexpected error occurred during "
                    f"validation rule {i+1} (Rule Type: {type(rule).__name__}): {e}"
                )
                logger.critical(error_msg, exc_info=True) # Critical for unexpected system-level issues
                raise DataValidationException(error_msg) from e
        
        logger.info(f"[{self.node_name}] All validation rules passed. Data is considered valid.")
        return data
