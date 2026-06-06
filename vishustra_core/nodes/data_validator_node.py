import logging
from typing import Any, Dict, List, Callable, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node designed to validate input data against a
    pre-configured set of rules.

    This node ensures that data conforms to expected formats, types, or
    contains necessary information before it proceeds to subsequent processing
    stages in the orchestration flow. It acts as a gatekeeper, preventing
    malformed or incomplete data from affecting downstream nodes.
    """

    def __init__(self, validation_rules: Optional[List[Callable[[Any], None]]] = None):
        """
        Initializes the DataValidatorNode with an optional list of validation
        rules.

        Each rule in `validation_rules` must be a callable that accepts the
        data as its sole argument. If a validation check fails, the callable
        is expected to raise a `ValueError` (or a more specific custom exception)
        to signal the failure. If validation passes for a given rule, the
        callable should simply complete its execution without raising an exception
        and return `None`.

        Args:
            validation_rules: An optional list of callable functions, each
                              representing a specific validation check. If
                              `None` or an empty list is provided, the node
                              will not perform any validations and data will
                              pass through unchecked, with a warning logged.
        """
        self._validation_rules = validation_rules if validation_rules is not None else []
        logger.debug("DataValidatorNode initialized with %d validation rules.", len(self._validation_rules))

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of this node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by applying all configured validation rules
        sequentially.

        If any validation rule fails (i.e., raises an exception), this method
        will catch the exception, log the failure details, and then re-raise
        a `ValueError` encapsulating the specific validation error. This
        ensures that upstream nodes can react to validation failures.
        If all validation rules pass successfully, the original input `data`
        is returned, indicating its validity.

        Args:
            data: The input data payload to be validated.
            context: A dictionary providing contextual information for the node's
                     operation. While not directly used by the validation rules
                     themselves, it's available for potential future extensions
                     or logging purposes.

        Returns:
            The original input `data` if it successfully passes all validation
            checks.

        Raises:
            ValueError: If any configured validation rule fails. The exception
                        message will contain details about the specific rule
                        that caused the failure and its underlying error.
        """
        logger.info("Initiating data validation process with '%s' node.", self.node_name)

        if not self._validation_rules:
            logger.warning(
                "No validation rules configured for '%s' node. Data will pass through unchecked.",
                self.node_name
            )
            return data

        for i, rule_callable in enumerate(self._validation_rules):
            rule_name = getattr(rule_callable, '__name__', str(rule_callable))
            try:
                logger.debug("Applying validation rule #%d: '%s'", i + 1, rule_name)
                rule_callable(data)
            except Exception as e:
                # Catching generic Exception for robustness, but expecting ValueError from rules.
                error_message = (
                    f"Data validation failed by rule '{rule_name}' in '{self.node_name}' node: {e}"
                )
                logger.error(error_message, exc_info=True)
                # Re-raise as a ValueError to standardize error reporting for validation failures.
                raise ValueError(error_message) from e

        logger.info("Data successfully validated by '%s' node. All rules passed.", self.node_name)
        return data