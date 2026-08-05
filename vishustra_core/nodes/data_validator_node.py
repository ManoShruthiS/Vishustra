import logging
from typing import Any, Dict, Type

# The project context specifies the import path for BaseNode
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data
    against a predefined schema.

    This node ensures that incoming data conforms to expected types,
    the presence of required fields, and optionally, value ranges or
    length constraints, before the data proceeds to subsequent nodes
    in the orchestration pipeline. By failing early on invalid data,
    it prevents errors from propagating downstream.
    """

    def __init__(self, schema: Dict[str, Dict[str, Any]]):
        """
        Initializes the DataValidatorNode with a validation schema.

        The schema defines the expected structure and validation rules for the data.

        Example schema structure:
        {
            "user_id": {"type": int, "required": True},
            "user_name": {"type": str, "required": True, "min_length": 3, "max_length": 50},
            "age": {"type": int, "required": False, "min_value": 0, "max_value": 120},
            "email": {"type": str, "required": True} # For brevity, regex is not implemented in process method
        }

        Args:
            schema: A dictionary defining the validation rules. Each key represents
                    a field name, and its value is another dictionary containing
                    validation rules for that field (e.g., 'type', 'required',
                    'min_value', 'max_value', 'min_length', 'max_length').
        
        Raises:
            TypeError: If the provided schema is not a dictionary.
        """
        if not isinstance(schema, dict):
            # Log error and raise exception if schema is not a dictionary
            logger.error(f"[{self.__class__.__name__}] Initialization failed: Schema must be a dictionary, but received {type(schema).__name__}.")
            raise TypeError("Schema for DataValidatorNode must be a dictionary.")
        self._schema = schema
        logger.debug(f"[{self.node_name}] Node initialized with schema: {self._schema}")

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this node.
        """
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured schema.

        If validation fails for any field based on the schema rules, a ValueError
        is raised, halting further processing for that data point and indicating
        a data integrity issue. If validation succeeds, the original data is returned.

        Args:
            data: The input data to be validated. This node expects a dictionary
                  for structured validation based on its schema.
            context: A dictionary containing contextual information for the node.
                     (Not directly used in this validator implementation, but
                     available for future extensions, e.g., dynamic schema loading).

        Returns:
            The original, validated data if all checks pass.

        Raises:
            TypeError: If the input 'data' is not a dictionary, which is required
                       for schema-based field validation with this node.
            ValueError: If any field in the 'data' fails to meet the specified
                        validation rules in the schema (e.g., missing required field,
                        incorrect type, out-of-range value, invalid length).
        """
        logger.info(f"[{self.node_name}] Starting data validation for incoming data (type: {type(data).__name__}).")

        # Ensure the input data is a dictionary for schema-based validation
        if not isinstance(data, dict):
            error_msg = (
                f"[{self.node_name}] Validation Error: Input data must be a dictionary "
                f"for schema-based validation, but received {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        # Iterate through each field defined in the schema and apply its rules
        for field_name, rules in self._schema.items():
            is_required = rules.get("required", False)
            expected_type: Type = rules.get("type", Any) # Defaults to Any if not specified
            min_value = rules.get("min_value")
            max_value = rules.get("max_value")
            min_length = rules.get("min_length")
            max_length = rules.get("max_length")
            # regex = rules.get("regex") # Placeholder for future regex implementation

            # 1. Check for missing required fields
            if field_name not in data:
                if is_required:
                    error_msg = (
                        f"[{self.node_name}] Validation failed for field '{field_name}': "
                        f"Required field is missing in the input data."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    # If field is not required and not present, skip further validation for it
                    continue

            field_value = data[field_name]

            # 2. Type validation
            if not isinstance(field_value, expected_type):
                error_msg = (
                    f"[{self.node_name}] Validation failed for field '{field_name}': "
                    f"Expected type {expected_type.__name__}, but received {type(field_value).__name__} "
                    f"with value '{field_value}'."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # 3. Value range validation (applicable to numeric types)
            if isinstance(field_value, (int, float)):
                if min_value is not None and field_value < min_value:
                    error_msg = (
                        f"[{self.node_name}] Validation failed for field '{field_name}': "
                        f"Value {field_value} is less than the minimum allowed {min_value}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if max_value is not None and field_value > max_value:
                    error_msg = (
                        f"[{self.node_name}] Validation failed for field '{field_name}': "
                        f"Value {field_value} is greater than the maximum allowed {max_value}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            # 4. Length validation (applicable to sized types like strings, lists, dicts, tuples, bytes)
            if isinstance(field_value, (str, list, dict, tuple, bytes)):
                if min_length is not None and len(field_value) < min_length:
                    error_msg = (
                        f"[{self.node_name}] Validation failed for field '{field_name}': "
                        f"Length {len(field_value)} is less than the minimum allowed {min_length}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                if max_length is not None and len(field_value) > max_length:
                    error_msg = (
                        f"[{self.node_name}] Validation failed for field '{field_name}': "
                        f"Length {len(field_value)} is greater than the maximum allowed {max_length}."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

            # 5. Regex validation (future enhancement - would require 'import re')
            # if isinstance(field_value, str) and regex:
            #     if not re.match(regex, field_value):
            #         error_msg = (
            #             f"[{self.node_name}] Validation failed for field '{field_name}': "
            #             f"Value '{field_value}' does not match regex pattern '{regex}'."
            #         )
            #         logger.error(error_msg)
            #         raise ValueError(error_msg)

        logger.info(f"[{self.node_name}] Data successfully validated against schema. Returning original data.")
        return data