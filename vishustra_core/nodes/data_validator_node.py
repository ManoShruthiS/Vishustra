import logging
from typing import Any, Dict, List, Callable

# Attempt to import BaseNode from the project's core structure.
# A fallback is provided for environments where vishustra_core is not yet installed,
# allowing for standalone testing or initial development.
try:
    from vishustra_core.nodes.base_node import BaseNode
except ImportError:
    # This block provides a mock BaseNode for development/testing outside the
    # full Vishustra framework context. In a deployed Vishustra environment,
    # the primary import path should succeed.
    logging.warning(
        "Could not import BaseNode from 'vishustra_core.nodes.base_node'. "
        "Using a local mock BaseNode for development purposes. "
        "Ensure 'vishustra_core' is correctly installed for production."
    )
    from abc import ABC, abstractmethod
    class BaseNode(ABC):
        @abstractmethod
        def process(self, data: Any, context: Dict[str, Any]) -> Any:
            pass
        @property
        @abstractmethod
        def node_name(self) -> str:
            pass


class ValidationError(Exception):
    """Custom exception raised when data fails validation within the DataValidatorNode."""
    pass

# A type alias for a validation rule. A rule is a callable that takes any data
# and raises an exception (e.g., ValidationError, ValueError) if the data is invalid.
# If the data is valid, the callable should simply return or do nothing.
ValidationRule = Callable[[Any], None]

class DataValidatorNode(BaseNode):
    """
    A processing node designed to validate input data against a predefined set of rules.

    This node is crucial for ensuring data integrity and correctness at various
    stages of an LLM orchestration pipeline. Each rule provided to the validator
    is a callable that, when executed, should raise an exception if the data
    fails its specific validation criterion. If no exception is raised by a rule,
    the data is considered valid according to that rule.
    """

    def __init__(self, rules: List[ValidationRule]):
        """
        Initializes the DataValidatorNode with a collection of validation rules.

        Args:
            rules (List[ValidationRule]): A list of callable validation rules.
                                          Each rule should accept `data: Any` as its
                                          single argument. If validation fails, the rule
                                          is expected to raise an exception (e.g.,
                                          `ValidationError`, `ValueError`). If the data
                                          is valid, the rule should complete without
                                          raising an exception.
        """
        self._rules = rules
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.debug(
            "DataValidatorNode initialized with %d validation rules.", len(self._rules)
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes all configured validation rules against the input data.

        This method iterates through each validation rule. If any rule raises
        an exception, the failure is recorded. After attempting all rules, if
        there were any failures, a comprehensive `ValidationError` is raised
        containing details of all individual validation failures. If all rules
        pass, the original (unmodified) data is returned, signifying successful
        validation.

        Args:
            data (Any): The data payload to be validated. This could be any Python
                        object, such as a dictionary, list, string, or a custom object.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the current pipeline execution. While
                                       not directly used by the generic `DataValidator`
                                       logic itself, it's part of the standard node
                                       signature and can be leveraged by custom rules
                                       if context-dependent validation is required.

        Returns:
            Any: The original `data` if it successfully passes all validation rules.

        Raises:
            ValidationError: If one or more validation rules fail, encapsulating
                             all detected error messages.
        """
        self._logger.info("Initiating data validation for input data (type: %s).", type(data).__name__)
        failed_validations: List[str] = []

        for i, rule in enumerate(self._rules):
            try:
                # Execute the validation rule. A successful rule returns without
                # raising an exception.
                rule(data)
                self._logger.debug("Validation rule %d passed successfully.", i + 1)
            except Exception as e:
                # Catching a broad Exception here to ensure all rule-raised errors
                # are captured, then re-packaging them into our custom ValidationError.
                error_message = f"Validation rule {i + 1} failed: {type(e).__name__} - {e}"
                self._logger.warning(error_message, exc_info=True) # Log exception details for debugging
                failed_validations.append(error_message)

        if failed_validations:
            # Aggregate all error messages into a single, comprehensive ValidationError.
            aggregate_error_message = (
                f"Data validation failed with {len(failed_validations)} errors:\n  "
                + "\n  ".join(failed_validations)
            )
            self._logger.error(aggregate_error_message)
            raise ValidationError(aggregate_error_message)
        else:
            self._logger.info("Data successfully passed all validation rules.")
            return data
