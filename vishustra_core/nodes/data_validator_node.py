import logging
from typing import Any, Dict

# Assuming BaseNode is correctly available in the project's sys.path
from vishustra_core.nodes.base_node import BaseNode

# For robust JSON schema validation, this is a core dependency for this node.
from jsonschema import validate, ValidationError, SchemaError

logger = logging.getLogger(__name__)


class DataValidatorNode(BaseNode):
    """
    A processing node designed to validate input data against a predefined JSON schema.

    This node plays a crucial role in ensuring data integrity and consistency
    within the Vishustra orchestration framework. It prevents malformed or
    unexpected data from propagating further down the processing pipeline,
    thereby enhancing reliability and simplifying downstream node logic.
    """

    def __init__(self, schema: Dict[str, Any]):
        """
        Initializes the DataValidatorNode with a JSON schema.

        Args:
            schema: A dictionary representing the JSON schema to validate data against.
                    This schema should conform to a supported JSON Schema draft (e.g., Draft 7).

        Raises:
            TypeError: If the provided schema is not a dictionary.
            SchemaError: If the provided schema itself is not a valid JSON Schema.
        """
        if not isinstance(schema, dict):
            logger.critical("DataValidatorNode requires a schema of type 'dict'. Received: %s", type(schema).__name__)
            raise TypeError(f"Schema for DataValidatorNode must be a dictionary. Got {type(schema).__name__}.")

        try:
            # Validate the schema itself to catch configuration errors early
            # A simple way to do this with jsonschema is to validate a minimal valid JSON against it.
            # Or, for more thorough schema validation, one might use jsonschema.Draft7Validator.check_schema(schema)
            # For simplicity, we assume 'validate' will throw SchemaError if the schema is fundamentally malformed
            validate({}, schema) # Attempt to validate an empty object against the schema to check schema validity
        except SchemaError as e:
            logger.critical("Provided schema for DataValidatorNode is invalid: %s", e.message)
            raise SchemaError(f"Invalid JSON schema provided for DataValidatorNode: {e.message}") from e
        except Exception as e:
            logger.critical("An unexpected error occurred during schema initialization: %s", e, exc_info=True)
            raise # Propagate unexpected errors

        self._schema = schema
        logger.debug("DataValidatorNode initialized successfully with schema.")

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured JSON schema.

        If the data fails validation, a `ValueError` is raised, detailing the validation error.
        If validation passes, the original input data is returned unchanged.

        Args:
            data: The input data to be validated. This can be any serializable Python object
                  that can be mapped to JSON types.
            context: A dictionary containing contextual information relevant to the current
                     execution flow. While not directly used for validation in this node,
                     it adheres to the `BaseNode` interface.

        Returns:
            The original input data, if it successfully passes validation against the schema.

        Raises:
            ValueError: If the input data does not conform to the predefined schema.
            Exception: For any other unexpected errors encountered during the validation process.
        """
        logger.info("Initiating data validation process using DataValidatorNode.")
        # Log a snippet of data for debugging, careful not to log excessive amounts
        logger.debug("Data sample (first 200 chars) for validation: %s", str(data)[:200])

        try:
            validate(instance=data, schema=self._schema)
            logger.info("Data successfully validated against schema. Proceeding with original data.")
            return data
        except ValidationError as e:
            logger.error("Data validation failed: %s", e.message)
            # Re-raise as a more generic ValueError to abstract away jsonschema specifics
            # from the consuming framework logic, while retaining original error context.
            raise ValueError(f"Input data failed schema validation: {e.message}") from e
        except Exception as e:
            logger.error("An unexpected error occurred during data validation in DataValidatorNode: %s", e, exc_info=True)
            # Re-raise to ensure all unexpected issues are propagated upstream.
            raise
