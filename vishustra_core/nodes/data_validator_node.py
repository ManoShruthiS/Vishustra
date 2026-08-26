import logging
import re
from typing import Any, Dict, List, Union, Callable

# Assuming the project structure places BaseNode in vishustra_core.nodes.base_node
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception raised for data validation failures within the Vishustra framework."""
    pass

class DataValidatorNode(BaseNode):
    """
    A Vishustra processing node responsible for validating input data against
    a set of predefined rules specified in the execution context.

    This node is critical for maintaining data integrity and consistency
    early in the pipeline, preventing malformed or invalid data from
    propagating to downstream processes and causing unexpected errors.

    Validation rules are highly configurable via the 'validation_config'
    key in the context dictionary.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "DataValidator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by validating it against rules defined in the
        'validation_config' within the provided context.

        The 'validation_config' dictionary can specify various rules:
        - 'required_keys': `List[str]` - Keys that must be present if `data` is a dictionary.
        - 'schema': `Dict[str, Dict[str, Any]]` - Defines expected types and optional
          constraints for specific keys in a dictionary (e.g., 'type', 'min_value',
          'max_value', 'min_length', 'max_length', 'regex', 'validator').
        - 'allow_extra_keys': `bool` - If False (default), raises an error if `data`
          (if a dict) contains keys not specified in 'schema' or 'required_keys'.
        - 'type_check': `type` - A general type check for the entire `data` object
          if it's not a dictionary, or if no schema/required_keys are provided.

        Args:
            data: The input data to be validated. Can be of any type.
            context: A dictionary containing operational context, crucially including
                     the 'validation_config' for this node.

        Returns:
            The original data, unmodified, if it passes all specified validation rules.

        Raises:
            ValidationError: If the data fails to meet any of the specified
                             validation requirements.
        """
        validation_config = context.get("validation_config")

        if not validation_config:
            logger.warning(
                f"[{self.node_name}] No 'validation_config' found in context. "
                "Data will be passed through without validation, which may lead to downstream issues."
            )
            return data

        logger.debug(f"[{self.node_name}] Initiating data validation with config: {validation_config}")

        # --- General Type Check (if not a dict or no dict-specific rules) ---
        general_type_check = validation_config.get("type_check")
        if general_type_check and not isinstance(data, dict):
            if not isinstance(data, general_type_check):
                raise ValidationError(
                    f"[{self.node_name}] Data has incorrect type. "
                    f"Expected {general_type_check.__name__}, but received {type(data).__name__}."
                )

        # --- Dictionary-specific Validations ---
        if isinstance(data, dict):
            # Rule 1: Check for required keys
            required_keys = validation_config.get("required_keys", [])
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                raise ValidationError(
                    f"[{self.node_name}] Data is missing one or more required keys: "
                    f"{', '.join(missing_keys)}."
                )

            # Rule 2: Validate against schema for key types and values
            schema = validation_config.get("schema")
            if schema:
                for key, rules in schema.items():
                    if key not in data:
                        # If a key is in schema but not required_keys, it's optional, so skip.
                        # If it was required, it would have been caught by required_keys check.
                        continue

                    value = data[key]
                    expected_type = rules.get("type")

                    # Type check for specific key
                    if expected_type and not isinstance(value, expected_type):
                        raise ValidationError(
                            f"[{self.node_name}] Key '{key}' has an incorrect type. "
                            f"Expected {expected_type.__name__}, but received {type(value).__name__}."
                        )

                    # Value constraints for numeric types
                    if expected_type in (int, float):
                        min_value = rules.get("min_value")
                        if min_value is not None and value < min_value:
                            raise ValidationError(
                                f"[{self.node_name}] Key '{key}' value ({value}) is below "
                                f"the minimum allowed ({min_value})."
                            )
                        max_value = rules.get("max_value")
                        if max_value is not None and value > max_value:
                            raise ValidationError(
                                f"[{self.node_name}] Key '{key}' value ({value}) is above "
                                f"the maximum allowed ({max_value})."
                            )
                    # Length and regex constraints for string types
                    elif expected_type is str:
                        min_length = rules.get("min_length")
                        if min_length is not None and len(value) < min_length:
                            raise ValidationError(
                                f"[{self.node_name}] Key '{key}' string length ({len(value)}) is "
                                f"below the minimum allowed ({min_length})."
                            )
                        max_length = rules.get("max_length")
                        if max_length is not None and len(value) > max_length:
                            raise ValidationError(
                                f"[{self.node_name}] Key '{key}' string length ({len(value)}) is "
                                f"above the maximum allowed ({max_length})."
                            )
                        regex_pattern = rules.get("regex")
                        if regex_pattern:
                            try:
                                if not re.match(regex_pattern, value):
                                    raise ValidationError(
                                        f"[{self.node_name}] Key '{key}' value ('{value}') does not "
                                        f"match the required regex pattern: '{regex_pattern}'."
                                    )
                            except re.error as e:
                                logger.error(
                                    f"[{self.node_name}] Invalid regex pattern provided for key '{key}': "
                                    f"'{regex_pattern}'. Error: {e}", exc_info=True
                                )
                                raise ValidationError(
                                    f"[{self.node_name}] Configuration error: Invalid regex pattern "
                                    f"specified for key '{key}'."
                                ) from e

                    # Custom validator function
                    custom_validator: Callable[[Any], bool] = rules.get("validator")
                    if custom_validator:
                        try:
                            if not callable(custom_validator):
                                raise TypeError("Custom validator must be a callable function.")
                            if not custom_validator(value):
                                raise ValidationError(
                                    f"[{self.node_name}] Key '{key}' failed custom validation."
                                )
                        except ValidationError: # Re-raise custom ValidationErrors directly
                            raise
                        except Exception as e:
                            logger.error(
                                f"[{self.node_name}] Custom validator for key '{key}' "
                                f"encountered an unexpected error: {e}", exc_info=True
                            )
                            raise ValidationError(
                                f"[{self.node_name}] Custom validation for key '{key}' failed "
                                f"due to an internal error."
                            ) from e

            # Rule 3: Check for extra keys if not allowed
            allow_extra_keys = validation_config.get("allow_extra_keys", False)
            if not allow_extra_keys:
                defined_keys = set(schema.keys() if schema else []).union(set(required_keys))
                extra_keys = [key for key in data if key not in defined_keys]
                if extra_keys:
                    raise ValidationError(
                        f"[{self.node_name}] Data contains unsupported extra keys: "
                        f"{', '.join(extra_keys)}. To allow them, set 'allow_extra_keys' to True "
                        f"in the validation configuration."
                    )
        elif validation_config.get("schema") or validation_config.get("required_keys"):
            # If schema or required_keys are provided, but data is not a dict
            raise ValidationError(
                f"[{self.node_name}] Validation configuration expects dictionary data "
                f"(due to 'schema' or 'required_keys' rules), but received data of type "
                f"'{type(data).__name__}'."
            )

        logger.info(f"[{self.node_name}] Data successfully validated against all specified rules.")
        return data
