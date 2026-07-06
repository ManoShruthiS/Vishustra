import logging
import re
from typing import Any, Dict, List, Type, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class InvalidDataError(ValueError):
    """Custom exception raised when data fails validation."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node that validates input data against a set of predefined rules.

    This node is crucial for ensuring data quality and consistency within the
    orchestration pipeline. It supports various validation types including
    required fields, data types, string lengths, numeric ranges, allowed enum values,
    and regular expression patterns. If data fails validation, an `InvalidDataError`
    is raised, halting further processing until the data is rectified.
    """

    def __init__(self, validation_rules: Dict[str, Dict[str, Any]]):
        """
        Initializes the DataValidatorNode with specific validation rules.

        Args:
            validation_rules: A dictionary where keys are field names in the
                              input data (expected to be a dictionary), and values
                              are dictionaries defining validation constraints for
                              that field.

                              Example rule structure:
                              {
                                  "user_id": {"type": int, "required": True, "min_value": 1},
                                  "username": {"type": str, "required": True, "min_length": 3, "max_length": 20},
                                  "email": {"type": str, "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", "required": True},
                                  "age": {"type": int, "min_value": 0, "max_value": 120, "required": False},
                                  "status": {"type": str, "enum": ["active", "inactive", "pending"], "required": True}
                              }
        Raises:
            TypeError: If `validation_rules` is not a dictionary.
        """
        if not isinstance(validation_rules, dict):
            raise TypeError(
                f"[{self.__class__.__name__}] 'validation_rules' must be a dictionary. "
                f"Got {type(validation_rules).__name__}."
            )
        self._validation_rules = validation_rules
        logger.debug(f"[{self.node_name}] Initialized with validation rules for {len(validation_rules)} fields.")

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "DataValidatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against the configured rules.

        If `data` is not a dictionary, a `TypeError` is raised. If any validation
        rule fails for a field, an `InvalidDataError` is raised.

        Args:
            data: The input data to be validated. For structured validation, this
                  is expected to be a dictionary.
            context: A dictionary containing contextual information for the process.
                     Currently not directly used by this node for validation logic,
                     but available for future extensions (e.g., dynamic rules).

        Returns:
            The original data if all validations pass, ensuring immutability.

        Raises:
            TypeError: If the input 'data' is not a dictionary.
            InvalidDataError: If the data fails any configured validation rule.
        """
        logger.info(f"[{self.node_name}] Starting data validation for incoming data.")

        if not isinstance(data, dict):
            error_msg = (
                f"[{self.node_name}] Input 'data' must be a dictionary for structured "
                f"validation. Received type: {type(data).__name__}."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        for field_name, rules in self._validation_rules.items():
            is_required = rules.get("required", False)
            field_value = data.get(field_name)

            if field_value is None:
                if is_required:
                    error_msg = f"[{self.node_name}] Required field '{field_name}' is missing."
                    logger.warning(error_msg)
                    raise InvalidDataError(error_msg)
                else:
                    logger.debug(f"[{self.node_name}] Optional field '{field_name}' is missing, skipping validation checks for it.")
                    continue  # Skip further checks if the field is optional and not present

            # --- Type validation ---
            expected_type = rules.get("type")
            if expected_type and not isinstance(field_value, expected_type):
                error_msg = (
                    f"[{self.node_name}] Field '{field_name}' has incorrect type. "
                    f"Expected '{expected_type.__name__}', got '{type(field_value).__name__}' "
                    f"with value '{field_value}'."
                )
                logger.warning(error_msg)
                raise InvalidDataError(error_msg)

            # --- Length validation (for strings, lists, tuples) ---
            if isinstance(field_value, (str, list, tuple)):
                min_len = rules.get("min_length")
                max_len = rules.get("max_length")
                current_len = len(field_value)
                
                if min_len is not None and current_len < min_len:
                    error_msg = (
                        f"[{self.node_name}] Field '{field_name}' length ({current_len}) "
                        f"is less than minimum required length ({min_len})."
                    )
                    logger.warning(error_msg)
                    raise InvalidDataError(error_msg)
                
                if max_len is not None and current_len > max_len:
                    error_msg = (
                        f"[{self.node_name}] Field '{field_name}' length ({current_len}) "
                        f"exceeds maximum allowed length ({max_len})."
                    )
                    logger.warning(error_msg)
                    raise InvalidDataError(error_msg)
            
            # --- Value range validation (for numbers) ---
            if isinstance(field_value, (int, float)):
                min_val = rules.get("min_value")
                max_val = rules.get("max_value")

                if min_val is not None and field_value < min_val:
                    error_msg = (
                        f"[{self.node_name}] Field '{field_name}' value ({field_value}) "
                        f"is less than minimum allowed value ({min_val})."
                    )
                    logger.warning(error_msg)
                    raise InvalidDataError(error_msg)
                
                if max_val is not None and field_value > max_val:
                    error_msg = (
                        f"[{self.node_name}] Field '{field_name}' value ({field_value}) "
                        f"exceeds maximum allowed value ({max_val})."
                    )
                    logger.warning(error_msg)
                    raise InvalidDataError(error_msg)

            # --- Enum validation ---
            allowed_enum = rules.get("enum")
            if allowed_enum is not None and field_value not in allowed_enum:
                error_msg = (
                    f"[{self.node_name}] Field '{field_name}' value ('{field_value}') "
                    f"is not one of the allowed values: {allowed_enum}."
                )
                logger.warning(error_msg)
                raise InvalidDataError(error_msg)

            # --- Regex pattern validation ---
            pattern = rules.get("pattern")
            if pattern is not None and isinstance(field_value, str):
                if not re.fullmatch(pattern, field_value):
                    error_msg = (
                        f"[{self.node_name}] Field '{field_name}' value ('{field_value}') "
                        f"does not match required pattern: '{pattern}'."
                    )
                    logger.warning(error_msg)
                    raise InvalidDataError(error_msg)

            logger.debug(f"[{self.node_name}] Field '{field_name}' validated successfully.")

        logger.info(f"[{self.node_name}] All validation rules passed for the data.")
        return data
