import logging
from typing import Any, Dict, List, Callable

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """
    Exception raised when data fails validation within the DataValidator node.
    This helps in distinguishing validation failures from other processing errors.
    """
    pass

class DataValidator(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against
    a set of predefined rules.

    This node ensures data conforms to expected formats, types, or business
    logic before it proceeds to subsequent stages in the orchestration.
    If any validation rule fails, a DataValidationError is raised, stopping
    further processing along that path.
    """

    def __init__(self, validation_rules: List[Callable[[Any, Dict[str, Any]], bool]], name: str = "DataValidatorNode"):
        """
        Initializes the DataValidator node with a list of validation rules.

        Each rule is a callable that receives the `data` and `context` and
        is expected to return `True` if the data passes that rule, `False`
        if it fails, or raise an exception upon critical validation failure.

        Args:
            validation_rules: A list of callable functions. Each function must
                              accept `data` (Any) and `context` (Dict[str, Any])
                              as arguments. It should return `True` for valid data,
                              `False` for invalid data, or raise an exception
                              if a validation condition cannot be met.
            name: An optional, descriptive name for this specific validator
                  instance. Defaults to "DataValidatorNode". This name is used
                  for logging and identification within a workflow.

        Raises:
            TypeError: If `validation_rules` is not a list or contains non-callable
                       elements.
        """
        if not isinstance(validation_rules, list):
            raise TypeError(
                f"Validation rules for '{name}' must be a list, received {type(validation_rules).__name__}."
            )
        for i, rule in enumerate(validation_rules):
            if not callable(rule):
                raise TypeError(
                    f"Validation rule at index {i} for '{name}' is not callable, received {type(rule).__name__}."
                )

        self._validation_rules = validation_rules
        self._instance_name = name
        logger.debug(
            f"DataValidator node '{self.node_name}' initialized with "
            f"{len(validation_rules)} validation rules."
        )

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this DataValidator node instance.
        """
        return self._instance_name

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying all configured validation rules.

        This method iterates through each rule. If a rule returns `False` or
        raises an exception, the data is considered invalid. All identified
        validation failures are collected, and if any exist, a single
        `DataValidationError` is raised detailing all issues.

        Args:
            data: The input data to be validated.
            context: A dictionary containing contextual information relevant
                     to the processing flow, which may include parameters
                     for validation rules.

        Returns:
            The original `data` unmodified if all validation rules pass successfully.

        Raises:
            DataValidationError: If any of the validation rules fail or raise
                                 an exception during their execution.
        """
        logger.info(f"Node '{self.node_name}' starting data validation process.")
        validation_failures: List[str] = []

        for i, rule_callable in enumerate(self._validation_rules):
            # Attempt to get a meaningful name for the rule for logging
            rule_id = getattr(rule_callable, '__name__', f"rule_at_index_{i}")
            logger.debug(f"Node '{self.node_name}' applying rule: '{rule_id}'.")

            try:
                is_valid = rule_callable(data, context)
                if not is_valid:
                    failure_msg = f"Rule '{rule_id}' explicitly returned False, indicating invalid data."
                    validation_failures.append(failure_msg)
                    logger.warning(
                        f"Validation failure in '{self.node_name}': {failure_msg}"
                    )
            except Exception as e:
                # Catch any unexpected exceptions from the rule callable itself
                failure_msg = (
                    f"Rule '{rule_id}' raised an unexpected exception: {e!r}. "
                    "This indicates a critical validation failure."
                )
                validation_failures.append(failure_msg)
                logger.error(
                    f"Validation failure in '{self.node_name}': {failure_msg}",
                    exc_info=True # Log stack trace for unexpected exceptions
                )

        if validation_failures:
            # Aggregate all error messages into a single, comprehensive exception
            combined_error_message = (
                f"Data failed validation in node '{self.node_name}'. "
                f"Detected {len(validation_failures)} failure(s): "
                f"{'; '.join(validation_failures)}"
            )
            logger.error(f"Final validation failed for '{self.node_name}': {combined_error_message}")
            raise DataValidationError(combined_error_message)
        else:
            logger.info(f"Node '{self.node_name}' successfully validated data. Data is valid.")
            return data
