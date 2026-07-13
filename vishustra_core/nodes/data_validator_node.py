import logging
from typing import Any, Dict, Type, Union

# Assuming vishustra_core is a package and nodes.base_node is a module within it
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against a defined schema.

    This node ensures that data conforms to expected structure and types before
    further processing, enhancing data integrity and preventing downstream errors.

    The validation schema is defined during the node's initialization, specifying
    required fields, their expected Python types, and optionally, nested schemas
    for dictionary fields.
    """

    def __init__(self, schema: Dict[str, Union[Type, Dict[str, Any]]]):
        """
        Initializes the DataValidatorNode with a specific validation schema.

        The schema is a dictionary where keys represent required field names
        in the input data. The values associated with these keys define the
        validation rules for that field:
        - If the value is a Python `type` (e.g., `str`, `int`, `list`, `dict`),
          the node checks if the corresponding data field's type matches.
        - If the value is another `dict`, it's treated as a nested schema,
          and the node recursively validates the sub-dictionary within the data.

        Args:
            schema: A dictionary defining the validation rules.

        Raises:
            TypeError: If the provided schema is not a dictionary or contains
                       invalid type definitions within its rules.
        """
        if not isinstance(schema, dict):
            raise TypeError("Schema for DataValidatorNode must be a dictionary.")
        self.schema = schema
        logger.info(f"[{self.node_name}] Initialized with schema: {self.schema}")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidator"

    def _validate_field(self, field_name: str, field_value: Any, expected_rule: Union[Type, Dict[str, Any]]) -> None:
        """
        Helper method to validate a single field's value against its expected rule.
        Recursively calls `_validate_data_against_schema` for nested dictionaries.
        """
        if isinstance(expected_rule, type):
            # Validate against a Python type
            if not isinstance(field_value, expected_rule):
                raise ValueError(
                    f"Validation failed for field '{field_name}': "
                    f"Expected type '{expected_rule.__name__}', got '{type(field_value).__name__}'."
                )
        elif isinstance(expected_rule, dict):
            # Validate against a nested schema
            if not isinstance(field_value, dict):
                raise ValueError(
                    f"Validation failed for field '{field_name}': "
                    f"Expected a dictionary for nested schema validation, got '{type(field_value).__name__}'."
                )
            # Recursively validate the sub-dictionary
            self._validate_data_against_schema(field_value, expected_rule)
        else:
            raise TypeError(
                f"Invalid schema rule for field '{field_name}': "
                f"Expected a type or a dictionary, got '{type(expected_rule).__name__}'."
            )

    def _validate_data_against_schema(self, data_to_validate: Any, schema_to_use: Dict[str, Any]) -> None:
        """
        Core validation logic to check `data_to_validate` against `schema_to_use`.
        Handles type checking and ensures all required fields are present.
        """
        if not isinstance(data_to_validate, dict):
            raise TypeError(
                f"Data for schema validation must be a dictionary, "
                f"got '{type(data_to_validate).__name__}'."
            )

        for field_name, expected_rule in schema_to_use.items():
            if field_name not in data_to_validate:
                raise ValueError(f"Required field '{field_name}' is missing from the data.")

            field_value = data_to_validate[field_name]
            self._validate_field(field_name, field_value, expected_rule)

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the node's configured schema.

        If the data successfully passes validation, it is returned unchanged.
        If validation fails due to missing fields, type mismatches, or schema
        misconfiguration, a relevant exception is raised after logging the error.

        Args:
            data: The input data to be validated. Typically expected to be a dictionary.
            context: A dictionary containing arbitrary runtime context. Not directly
                     used for schema definition by this node, but available for
                     potential future extensions or conditional logic.

        Returns:
            The original, validated data if all checks pass.

        Raises:
            TypeError: If the input `data` is not a dictionary (when a schema is defined),
                       or if schema rules are malformed.
            ValueError: If the `data` does not conform to the defined schema
                        (e.g., missing required fields, type mismatches).
            Exception: For any other unexpected errors during the validation process.
        """
        logger.debug(f"[{self.node_name}] Initiating validation for incoming data.")
        try:
            self._validate_data_against_schema(data, self.schema)
            logger.debug(f"[{self.node_name}] Data successfully validated.")
            return data
        except (TypeError, ValueError) as e:
            logger.error(f"[{self.node_name}] Data validation failed: {e}")
            raise  # Re-raise the specific validation error
        except Exception as e:
            logger.critical(
                f"[{self.node_name}] An unexpected critical error occurred during validation: {e}",
                exc_info=True
            )
            raise # Re-raise any critical unexpected errors
