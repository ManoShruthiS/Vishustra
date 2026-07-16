import logging
from typing import Any, Dict, Type

# Import BaseNode from the specified project path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a defined schema.

    This node ensures that incoming data conforms to expected types and structure,
    raising an error if validation fails. It primarily focuses on validating
    top-level fields of a dictionary against specified Python types.

    The `validation_schema` specifies expected field names and their corresponding types.
    An optional `strict` mode can be enabled to disallow extra fields not present
    in the schema.
    """

    def __init__(self, validation_schema: Dict[str, Type], strict: bool = False):
        """
        Initializes the DataValidatorNode with a validation schema.

        Args:
            validation_schema (Dict[str, Type]):
                A dictionary defining the expected structure and types of the input data.
                Keys are field names (strings), and values are Python types (e.g., `str`, `int`, `list`, `dict`).
                For example: `{'id': int, 'name': str, 'tags': list}`.
                Note: This validator performs a shallow type check; for nested dictionaries or lists,
                it checks the type of the container but not its contents.
            strict (bool):
                If True, any fields in the input `data` not defined in the `validation_schema`
                will cause a validation error. Defaults to False.

        Raises:
            TypeError: If `validation_schema` is not a dictionary.
            ValueError: If `validation_schema` contains non-string keys or non-type values.
        """
        if not isinstance(validation_schema, dict):
            raise TypeError("validation_schema must be a dictionary.")
        
        for k, v in validation_schema.items():
            if not isinstance(k, str):
                raise ValueError(f"All keys in validation_schema must be strings, but found type {type(k).__name__}.")
            if not isinstance(v, type):
                raise ValueError(f"All values in validation_schema must be Python types (e.g., str, int), but found {v} of type {type(v).__name__}.")

        self._validation_schema = validation_schema
        self._strict = strict
        logger.debug(
            "DataValidatorNode initialized with schema: %s, strict_mode: %s",
            {k: v.__name__ for k, v in self._validation_schema.items()}, self._strict
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured schema.

        Args:
            data (Any): The data to be validated. Expected to be a dictionary
                        if a validation_schema is provided.
            context (Dict[str, Any]): A dictionary containing context information
                                     (not directly used for validation in this node,
                                     but part of the BaseNode interface).

        Returns:
            Any: The original data if validation is successful.

        Raises:
            TypeError: If the input data's overall type does not match expectations (e.g., not a dictionary
                       when a schema is defined), or if a field's type does not match its schema definition.
            ValueError: If required fields are missing, or extra fields are present
                        in strict mode.
        """
        if not isinstance(data, dict):
            error_msg = (
                f"Validation failed by {self.node_name}: "
                f"Expected input data to be a dictionary, but received type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Validate required fields and types
        for field_name, expected_type in self._validation_schema.items():
            if field_name not in data:
                error_msg = (
                    f"Validation failed by {self.node_name}: "
                    f"Required field '{field_name}' is missing in data."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if not isinstance(data[field_name], expected_type):
                error_msg = (
                    f"Validation failed by {self.node_name} for field '{field_name}': "
                    f"Expected type {expected_type.__name__}, but received {type(data[field_name]).__name__} "
                    f"with value '{data[field_name]}'."
                )
                logger.error(error_msg)
                raise TypeError(error_msg)

        # Validate for extra fields in strict mode
        if self._strict:
            extra_fields = set(data.keys()) - set(self._validation_schema.keys())
            if extra_fields:
                error_msg = (
                    f"Validation failed by {self.node_name}: "
                    f"Unexpected extra field(s) found: "
                    f"{', '.join(sorted(extra_fields))} (strict mode enabled)."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        logger.info("%s successfully validated incoming data.", self.node_name)
        return data