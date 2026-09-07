import logging
from typing import Any, Dict, Callable

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data.

    This node ensures data integrity by applying a user-defined validation callable
    to the incoming data. If the data passes validation, it is returned unchanged.
    If it fails, a ValueError is raised, halting further processing for that data.
    The validation callable can be highly customized to enforce various rules,
    from schema checks to business logic constraints.
    """

    def __init__(self, validation_callable: Callable[[Any, Dict[str, Any]], bool]):
        """
        Initializes the DataValidatorNode with a specific validation function.

        The `validation_callable` will be invoked with the `data` and `context`
        during the `process` method. It is expected to return `True` for valid
        data and `False` for invalid data.

        Args:
            validation_callable (Callable[[Any, Dict[str, Any]], bool]):
                A callable (function or lambda) that takes two arguments:
                1. `data` (Any): The input data to be validated.
                2. `context` (Dict[str, Any]): The current processing context.
                It must return a boolean: `True` if the data is valid, `False` otherwise.

        Raises:
            TypeError: If the provided `validation_callable` is not actually callable.
        """
        if not callable(validation_callable):
            logger.error("Attempted to initialize DataValidatorNode with a non-callable validation_callable.")
            raise TypeError("`validation_callable` must be a callable function or lambda.")
        
        self._validation_callable = validation_callable
        callable_name = getattr(validation_callable, '__name__', 'anonymous_callable')
        logger.debug(f"[{self.node_name}] Initialized with validation callable: '{callable_name}'.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Executes the configured validation callable against the input data.

        If the data passes validation (i.e., the `validation_callable` returns `True`),
        the original data is returned. Otherwise, a `ValueError` is raised.
        Any exceptions occurring within the `validation_callable` itself are
        caught and re-raised as a `ValueError` for consistent error handling.

        Args:
            data (Any): The data payload to be validated.
            context (Dict[str, Any]): The immutable context dictionary for the current run.

        Returns:
            Any: The original `data` if it successfully passes all validation rules.

        Raises:
            ValueError: If the data fails validation or if an unexpected error
                        occurs during the validation process.
        """
        node_identifier = context.get('current_node_id', self.node_name)
        logger.info(f"[{node_identifier}] Starting data validation process.")

        try:
            is_valid = self._validation_callable(data, context)
            
            if is_valid:
                logger.debug(f"[{node_identifier}] Data successfully passed validation.")
                return data
            else:
                error_message = f"[{node_identifier}] Data failed validation checks. Input data deemed invalid."
                logger.warning(error_message)
                raise ValueError(error_message)
                
        except Exception as e:
            # Catching general exceptions from the validation_callable itself
            error_message = (
                f"[{node_identifier}] An unexpected error occurred during "
                f"execution of the validation callable: {e!r}"
            )
            logger.error(error_message, exc_info=True)
            raise ValueError(error_message) from e