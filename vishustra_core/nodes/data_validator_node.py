import logging
from typing import Any, Dict, Callable

# Assuming the base_node is located at vishustra_core/nodes/base_node.py
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class DataValidationException(ValueError):
    """
    Custom exception raised when data validation fails within the DataValidatorNode.
    Encapsulates the underlying validation error for clearer debugging.
    """
    pass


class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node designed to validate input data against
    a set of predefined rules.

    This node takes a dictionary of validation rules during initialization. Each
    rule is a callable function that receives the input data and is expected
    to raise an exception (e.g., ValueError, TypeError, or DataValidationException)
    if the data fails that specific validation check. If a rule determines the
    data is valid, it should simply return `None` or not raise an exception.

    If any configured validation rule fails, the node logs the error, and raises
    a `DataValidationException`, effectively stopping the processing flow
    for invalid data. If all rules pass successfully, the original input data
    is returned unmodified, indicating it meets all required criteria.
    """

    def __init__(self, validation_rules: Dict[str, Callable[[Any], None]]):
        """
        Initializes the DataValidatorNode with a collection of validation rules.

        Args:
            validation_rules (Dict[str, Callable[[Any], None]]):
                A dictionary where keys are descriptive names for each validation
                rule (e.g., "check_schema", "verify_field_presence"), and values
                are callable functions. Each callable function must accept the
                `data` (Any) as its sole argument. It should raise an exception
                if the data is invalid according to that rule, or simply return
                if the data is valid.
        
        Raises:
            TypeError: If `validation_rules` is not a dictionary, or if any
                       value in the dictionary is not a callable.
        """
        if not isinstance(validation_rules, dict):
            raise TypeError("The 'validation_rules' argument must be a dictionary.")
        if not all(isinstance(k, str) and callable(v) for k, v in validation_rules.items()):
            raise TypeError(
                "All keys in 'validation_rules' must be strings, and all values must be callable functions."
            )

        self._validation_rules = validation_rules
        logger.debug(
            f"[{self.node_name}] Initialized with {len(self._validation_rules)} validation rules."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes all configured validation rules against the input data.

        If any rule raises an exception, this method catches it, logs the failure,
        and re-raises it as a `DataValidationException`. If all rules pass
        without raising exceptions, the original input data is returned.

        Args:
            data (Any): The data payload to be validated.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current processing flow.
                                       (Not directly used by the validator rules
                                       themselves, but available for potential
                                       future extensions).

        Returns:
            Any: The original input `data` if it passes all validation checks.

        Raises:
            DataValidationException: If any of the configured validation rules
                                     fail for the input data.
        """
        logger.info(f"[{self.node_name}] Initiating data validation for incoming data.")

        if not self._validation_rules:
            logger.warning(
                f"[{self.node_name}] No validation rules are configured. "
                "Data will pass through without any checks."
            )
            return data

        for rule_name, validate_func in self._validation_rules.items():
            try:
                logger.debug(f"[{self.node_name}] Applying rule '{rule_name}'...")
                validate_func(data)
                logger.debug(f"[{self.node_name}] Rule '{rule_name}' passed successfully.")
            except Exception as e:
                logger.error(
                    f"[{self.node_name}] Validation rule '{rule_name}' failed for data: {data!r}. "
                    f"Error type: {e.__class__.__name__}, Message: {e}"
                )
                raise DataValidationException(
                    f"Validation failed by rule '{rule_name}': {e}"
                ) from e

        logger.info(f"[{self.node_name}] All validation rules passed. Data is considered valid.")
        return data