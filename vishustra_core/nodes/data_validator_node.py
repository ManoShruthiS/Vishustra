import logging
from typing import Any, Dict, Tuple, Type, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class DataValidationException(ValueError):
    """
    Custom exception raised when data validation fails within the DataValidatorNode.
    """
    pass


class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against a
    defined schema.

    This node ensures that the structure and types of fields within the input data
    conform to a specified `validation_schema`. It supports checking for required
    fields, their types, and can optionally enforce strict validation to disallow
    any fields not explicitly listed in the schema.
    """

    def __init__(self, validation_schema: Dict[str, Union[Type, Tuple[Type, ...]]], strict: bool = False):
        """
        Initializes the DataValidatorNode with the validation schema and strictness setting.

        The `validation_schema` dictates the expected fields and their types.
        For type checking, it leverages Python's `isinstance` function.

        Args:
            validation_schema (Dict[str, Union[Type, Tuple[Type, ...]]]):
                A dictionary where keys represent expected field names and values
                are the expected Python types (e.g., `str`, `int`, `float`, `list`, `dict`).
                For fields that can accept multiple types (like a union), provide a tuple
                of types (e.g., `(str, int)`).
            strict (bool):
                If `True`, the validator will raise a `DataValidationException` if the
                input `data` contains any fields that are not specified in the
                `validation_schema`. Defaults to `False`.

        Raises:
            TypeError: If the `validation_schema` is not a dictionary or contains
                       invalid key/value types.
        """
        if not isinstance(validation_schema, dict):
            raise TypeError("`validation_schema` must be a dictionary.")
        if not all(isinstance(k, str) for k in validation_schema.keys()):
            raise TypeError("All keys in `validation_schema` must be strings (field names).")
        if not all(isinstance(v, (type, tuple)) for v in validation_schema.values()):
            raise TypeError(
                "All values in `validation_schema` must be type objects "
                "(e.g., `str`, `int`) or tuples of type objects."
            )

        self._validation_schema = validation_schema
        self._strict = strict
        logger.debug(
            f"DataValidatorNode initialized with schema: {self._validation_schema}, strict mode: {self._strict}"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured schema.

        This method expects `data` to be a dictionary. It performs the following
        checks:
        1. Verifies that `data` is a dictionary.
        2. Checks for the presence of all required fields defined in `validation_schema`.
        3. Validates the type of each field against its expected type(s).
        4. If `strict` mode is enabled, it checks for and disallows any fields
           in `data` that are not present in `validation_schema`.

        Args:
            data (Any): The input data to be validated. Expected to be a dictionary.
            context (Dict[str, Any]): The processing context, which may contain
                                      additional runtime information. This node
                                      does not directly use the context for validation.

        Returns:
            Any: The original, validated input data.

        Raises:
            DataValidationException: If the input data fails any of the validation rules.
            TypeError: If the input `data` itself is not a dictionary.
        """
        if not isinstance(data, dict):
            logger.error(
                f"DataValidatorNode received non-dictionary input. Expected dict, got: {type(data).__name__}"
            )
            raise TypeError(
                f"Invalid input data type for '{self.node_name}' node: expected dictionary, "
                f"got {type(data).__name__}"
            )

        logger.info(f"Initiating data validation for node '{self.node_name}' (strict={self._strict}).")

        # 1. Validate required fields and their types
        for field_name, expected_type in self._validation_schema.items():
            if field_name not in data:
                logger.warning(f"Validation failed: Required field '{field_name}' is missing.")
                raise DataValidationException(f"Missing required field: '{field_name}'")

            field_value = data[field_name]
            if not isinstance(field_value, expected_type):
                logger.error(
                    f"Validation failed for field '{field_name}': expected type {expected_type}, "
                    f"got {type(field_value).__name__} with value: {field_value!r}"
                )
                raise DataValidationException(
                    f"Invalid type for field '{field_name}': expected {expected_type}, "
                    f"got {type(field_value).__name__}"
                )
            logger.debug(f"Field '{field_name}' type validated successfully as {type(field_value).__name__}.")

        # 2. Validate for unexpected fields if strict mode is enabled
        if self._strict:
            unexpected_fields = set(data.keys()) - set(self._validation_schema.keys())
            if unexpected_fields:
                logger.warning(
                    f"Validation failed: Unexpected fields found in data: {', '.join(unexpected_fields)}"
                )
                raise DataValidationException(
                    f"Unexpected fields present in data: {', '.join(unexpected_fields)}"
                )
            logger.debug("No unexpected fields found in strict mode, validation successful.")
        else:
            logger.debug("Strict mode is disabled, ignoring unexpected fields.")

        logger.info(f"Data successfully validated by node '{self.node_name}'.")
        return data