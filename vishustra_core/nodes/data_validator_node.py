import logging
from typing import Any, Dict, List, Type

# The BaseNode abstract class for all Vishustra processing nodes.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data
    against a set of configurable rules provided in the context.

    This node ensures data integrity and adherence to expected schemas
    before data proceeds to downstream processing stages. Validation rules
    can specify the overall data type, required keys for dictionaries,
    and expected data types for specific keys within dictionaries.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Validates the input data based on rules specified in the context.

        The 'context' dictionary can contain a 'validation_rules' key,
        which is itself a dictionary defining the validation criteria.
        Supported rules within 'validation_rules' include:
        - 'data_type': Type - The expected overall type of the data itself (e.g., str, dict, int).
        - 'required_keys': List[str] - Keys that must be present if 'data' is a dictionary.
        - 'key_types': Dict[str, Type] - Expected types for specific keys if 'data' is a dictionary.

        If validation fails due to the input data not meeting the criteria,
        a `ValueError` is raised. If the validation rules themselves are malformed,
        a `TypeError` is raised.

        Args:
            data: The input data to be validated.
            context: A dictionary containing operational context,
                     including optional 'validation_rules'.

        Returns:
            The original data if validation is successful.

        Raises:
            ValueError: If the data fails any of the specified validation rules.
            TypeError: If the 'validation_rules' dictionary or its sub-rules are malformed.
        """
        validation_rules = context.get("validation_rules")

        if not validation_rules:
            logger.debug(
                "No 'validation_rules' found in context for DataValidatorNode. "
                "Data passed through DataValidatorNode without specific validation."
            )
            return data

        if not isinstance(validation_rules, dict):
            logger.error(
                f"Malformed 'validation_rules' in context: expected dict, "
                f"got {type(validation_rules).__name__}."
            )
            raise TypeError("Validation rules must be a dictionary.")

        logger.debug(f"Starting validation for data using rules: {validation_rules}")

        # 1. Validate overall data type if specified
        expected_data_type: Type = validation_rules.get("data_type")
        if expected_data_type:
            if not isinstance(data, expected_data_type):
                error_msg = (
                    f"Data validation failed: Expected overall data type "
                    f"'{getattr(expected_data_type, '__name__', str(expected_data_type))}', "
                    f"but received type '{type(data).__name__}'."
                )
                logger.warning(error_msg)
                raise ValueError(error_msg)
            logger.debug(f"Overall data type '{getattr(expected_data_type, '__name__', str(expected_data_type))}' validated successfully.")

        # 2. Apply dictionary-specific validations if data is a dictionary
        if isinstance(data, dict):
            # 2a. Validate required keys
            required_keys: List[str] = validation_rules.get("required_keys", [])
            if not isinstance(required_keys, list):
                logger.error(
                    f"Malformed 'required_keys' in validation_rules: expected list, "
                    f"got {type(required_keys).__name__}."
                )
                raise TypeError("Validation rule 'required_keys' must be a list of strings.")

            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                error_msg = (
                    f"Data validation failed: Missing required keys in dictionary: "
                    f"{', '.join(missing_keys)}."
                )
                logger.warning(error_msg)
                raise ValueError(error_msg)
            if required_keys:
                logger.debug(f"All required keys {required_keys} are present.")

            # 2b. Validate key types
            key_types: Dict[str, Type] = validation_rules.get("key_types", {})
            if not isinstance(key_types, dict):
                logger.error(
                    f"Malformed 'key_types' in validation_rules: expected dict, "
                    f"got {type(key_types).__name__}."
                )
                raise TypeError("Validation rule 'key_types' must be a dictionary.")

            for key, expected_type in key_types.items():
                if key in data:
                    if not isinstance(data[key], expected_type):
                        error_msg = (
                            f"Data validation failed: Key '{key}' expected type "
                            f"'{getattr(expected_type, '__name__', str(expected_type))}', "
                            f"but received type '{type(data[key]).__name__}' "
                            f"with value '{data[key]}'."
                        )
                        logger.warning(error_msg)
                        raise ValueError(error_msg)
            if key_types:
                logger.debug(f"Key types for {list(key_types.keys())} validated successfully.")
        
        # 3. Handle cases where dict-specific rules are provided but data is not a dict
        elif validation_rules.get("required_keys") or validation_rules.get("key_types"):
            error_msg = (
                "Data validation failed: Dictionary-specific rules ('required_keys' "
                "or 'key_types') are specified, but the input data is not a dictionary. "
                f"Received type: {type(data).__name__}."
            )
            logger.warning(error_msg)
            raise ValueError(error_msg)

        logger.info("Data validated successfully against specified rules.")
        return data