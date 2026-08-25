import logging
from typing import Any, Dict, List, Union, Callable, Type, Optional

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidationError(ValueError):
    """Custom exception raised when data validation fails within the node."""
    pass

class DataValidatorNode(BaseNode):
    """
    A processing node responsible for validating input data against a predefined schema
    and a list of required keys.

    This node ensures data integrity and conformity to expected structures before
    further processing, preventing downstream issues.

    Validation rules are specified during the node's initialization:
    - `required_keys`: A list of string keys that must be present in the input data
                       if the data is a dictionary.
    - `schema`: A dictionary mapping data keys to their expected types or more complex
                validation rules. A rule can be a direct type (e.g., `int`) or a
                dictionary with `type` and an optional `validator` (a callable).
                Example:
                {"field_name": int,
                 "another_field": {"type": str, "validator": lambda x: len(x) > 0}}
    """

    def __init__(
        self,
        node_id: str,
        required_keys: Optional[List[str]] = None,
        schema: Optional[Dict[str, Union[Type, Dict[str, Any]]]] = None,
        raise_on_failure: bool = True
    ):
        """
        Initializes the DataValidatorNode with specific validation rules.

        Args:
            node_id: A unique identifier for this specific instance of the validator node.
                     Used in the `node_name` property and for logging.
            required_keys: An optional list of string keys that must exist in the input data.
                           This check is applied if the input `data` is a dictionary.
            schema: An optional dictionary defining the expected types for data fields
                    and/or custom validation logic. Keys are data field names, and values
                    are either a Python type (e.g., `str`, `int`) or a dictionary.
                    If a dictionary, it must contain a 'type' key and can optionally
                    include a 'validator' key, which is a callable that accepts the
                    field's value and returns `True` for valid, `False` for invalid.
            raise_on_failure: If True, a `DataValidationError` is raised immediately
                              upon the first validation failure. If False, the node
                              logs the error and returns `None`, allowing the pipeline
                              to potentially handle failures gracefully without halting.
        """
        self._node_id = node_id
        self._required_keys = required_keys if required_keys is not None else []
        self._schema = schema if schema is not None else {}
        self._raise_on_failure = raise_on_failure
        logger.debug(f"DataValidatorNode '{self._node_id}' initialized with "
                     f"required_keys: {self._required_keys}, schema: {self._schema}.")

    @property
    def node_name(self) -> str:
        """Returns the unique and descriptive name of this validator node instance."""
        return f"DataValidatorNode_{self._node_id}"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured `required_keys` and `schema`.

        Args:
            data: The input data to be validated. Expected to be a dictionary for
                  effective application of `required_keys` and `schema` rules.
            context: A dictionary containing contextual information relevant to the
                     current orchestration run (not directly used for validation rules
                     in this specific node, but part of the standard `BaseNode` signature).

        Returns:
            The original, validated `data` if all checks pass.

        Raises:
            DataValidationError: If `raise_on_failure` is True and any validation check fails.
                                 This includes type mismatches, missing required keys, or
                                 failures of custom validator functions.
        """
        logger.info(f"[{self.node_name}] Starting data validation for incoming data.")

        # Ensure data is a dictionary for schema and required_keys checks
        if not isinstance(data, dict):
            error_msg = (f"[{self.node_name}] Input data must be a dictionary for "
                         f"schema validation. Received type: {type(data).__name__}.")
            logger.error(error_msg)
            if self._raise_on_failure:
                raise DataValidationError(error_msg)
            return None

        # 1. Validate for required keys
        missing_keys = [key for key in self._required_keys if key not in data]
        if missing_keys:
            error_msg = (f"[{self.node_name}] Validation failed: Missing required keys: "
                         f"{', '.join(missing_keys)}.")
            logger.error(error_msg)
            if self._raise_on_failure:
                raise DataValidationError(error_msg)
            return None

        # 2. Validate against schema rules (type checking and custom validators)
        for key, rule in self._schema.items():
            if key not in data:
                logger.debug(f"[{self.node_name}] Key '{key}' defined in schema but "
                             f"not present in data. Skipping schema validation for it.")
                continue # Only validate keys that are actually present

            value = data[key]
            expected_type: Optional[Type] = None
            custom_validator: Optional[Callable[[Any], bool]] = None

            if isinstance(rule, dict):
                expected_type = rule.get('type')
                custom_validator = rule.get('validator')
            else: # Rule is assumed to be a direct type
                expected_type = rule

            # Type validation
            if expected_type and not isinstance(value, expected_type):
                error_msg = (f"[{self.node_name}] Validation failed for key '{key}': "
                             f"Expected type {getattr(expected_type, '__name__', str(expected_type))}, "
                             f"got {type(value).__name__} with value '{value}'.")
                logger.error(error_msg)
                if self._raise_on_failure:
                    raise DataValidationError(error_msg)
                return None

            # Custom validator function execution
            if custom_validator:
                try:
                    if not callable(custom_validator):
                        raise TypeError(f"Custom validator for key '{key}' must be a callable.")
                    if not custom_validator(value):
                        error_msg = (f"[{self.node_name}] Validation failed for key '{key}': "
                                     f"Custom validator returned False for value '{value}'.")
                        logger.error(error_msg)
                        if self._raise_on_failure:
                            raise DataValidationError(error_msg)
                        return None
                except Exception as e:
                    error_msg = (f"[{self.node_name}] Custom validator for key '{key}' "
                                 f"raised an unexpected exception during execution: {type(e).__name__}: {e}")
                    logger.exception(error_msg) # Log full traceback for diagnostic purposes
                    if self._raise_on_failure:
                        raise DataValidationError(error_msg) from e
                    return None

        logger.info(f"[{self.node_name}] Data successfully validated against all rules.")
        return data
