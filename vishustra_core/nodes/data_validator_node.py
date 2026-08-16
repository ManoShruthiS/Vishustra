import logging
from typing import Any, Dict, Optional, Type, Tuple, Union

# Assuming vishustra_core.nodes.base_node exists at the project root for imports.
# In a real project, this might be a relative import if within the same package,
# or a full package import if installed.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A processing node designed to validate input data against a predefined schema.

    This node ensures data integrity by checking for required fields,
    expected data types, and structural conformity before data progresses
    to subsequent nodes in the Vishustra pipeline. If validation fails,
    an appropriate error is raised to halt further processing of invalid data.
    """

    def __init__(self, validation_schema: Optional[Dict[str, Tuple[Union[Type, Tuple[Type, ...]], bool]]] = None):
        """
        Initializes the DataValidatorNode with an optional validation schema.

        The `validation_schema` dictates the rules for data validation.
        It is a dictionary where:
        - Keys: Represent the expected field names in the input data.
        - Values: Are tuples `(expected_type, is_required)`.
          - `expected_type`: Can be a single Python type (e.g., `str`, `int`, `list`, `dict`)
            or a tuple of types for multiple allowed types (e.g., `(str, int)`).
          - `is_required`: A boolean flag indicating whether the field must be present
            in the input data.

        Example `validation_schema` structure:
        ```python
        {
            "user_id": (str, True),
            "name": (str, True),
            "age": (int, False),  # 'age' is optional
            "email_verified": (bool, True),
            "preferences": (dict, False),
            "status_code": ((str, int), True) # 'status_code' can be string or int
        }
        ```

        Args:
            validation_schema: An optional dictionary defining the validation rules.
                               If None or empty, the node will pass data through without validation.
        """
        self._validation_schema = validation_schema if validation_schema is not None else {}
        logger.debug(f"DataValidatorNode initialized with schema: {self._validation_schema}")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data against the configured schema.

        Args:
            data: The data payload to be validated. If a `validation_schema`
                  is provided, this is typically expected to be a dictionary.
            context: A dictionary containing contextual information relevant
                     to the current processing pipeline. (Not directly used in validation logic).

        Returns:
            The original, unmodified `data` if it successfully passes all
            validation checks.

        Raises:
            TypeError: If the input `data` is not a dictionary when a `validation_schema`
                       is active, indicating a structural mismatch.
            ValueError: If the `data` does not conform to the rules specified
                        in the `validation_schema` (e.g., missing required fields,
                        incorrect data types).
        """
        if not self._validation_schema:
            logger.info("No validation schema provided. Data passed through without validation.")
            return data

        if not isinstance(data, dict):
            error_msg = (
                f"DataValidator expects input data to be a dictionary when a schema is defined, "
                f"but received type: '{type(data).__name__}'. Data: {data}"
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        logger.debug(f"Starting validation for data with node '{self.node_name}'.")

        for field_name, (expected_type, is_required) in self._validation_schema.items():
            if field_name not in data:
                if is_required:
                    error_msg = (
                        f"Validation failed for node '{self.node_name}': "
                        f"Required field '{field_name}' is missing from data."
                    )
                    logger.warning(error_msg)
                    raise ValueError(error_msg)
                else:
                    logger.debug(f"Optional field '{field_name}' not found in data, skipping type check.")
                    continue # Field is optional and missing, proceed to next field

            field_value = data[field_name]

            # Use isinstance which correctly handles tuples of types
            if not isinstance(field_value, expected_type):
                # Format expected types for error message clarity
                expected_type_names = (
                    expected_type.__name__
                    if isinstance(expected_type, type)
                    else ", ".join(t.__name__ for t in expected_type)
                )
                error_msg = (
                    f"Validation failed for node '{self.node_name}' on field '{field_name}': "
                    f"Expected type(s) '{expected_type_names}', but received '{type(field_value).__name__}' "
                    f"with value: '{field_value}'."
                )
                logger.warning(error_msg)
                raise ValueError(error_msg)
            
            logger.debug(
                f"Field '{field_name}' passed validation: "
                f"Value '{field_value}' is of expected type(s) '{type(field_value).__name__}'."
            )

        logger.info(f"Data successfully passed all validation checks for node '{self.node_name}'.")
        return data