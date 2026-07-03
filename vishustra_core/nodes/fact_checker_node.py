import logging
from typing import Any, Dict, List, Union

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class FactCheckerNode(BaseNode):
    """
    A processing node that simulates fact-checking of input text.
    It can be configured to use an external API (simulated) or an internal
    list of known facts provided in the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "FactCheckerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input data to perform a fact-check.

        The `data` input is expected to be a dictionary containing at least
        a 'text' key with the content to be fact-checked.

        The `context` can provide configuration for the fact-checking process:
        - 'fact_check_api_url': (Optional[str]) URL of an external fact-checking
          API. If provided, the node will simulate an API call.
        - 'known_facts': (Optional[List[str]]) A list of known facts to
          check against if no API URL is provided.

        Args:
            data: The input data, expected to be a dictionary with a 'text' key.
            context: A dictionary containing operational context and configuration.

        Returns:
            A dictionary containing:
            - 'original_data': The data passed into the node.
            - 'fact_check_result': A dictionary with the fact-checking outcome,
              including 'status' ('verified', 'unverified', 'partially_verified', 'failed'),
              'message', and relevant details.
        """
        logger.debug(f"[{self.node_name}] Starting process for data.")
        
        # Initialize result structure with a default error state
        result: Dict[str, Any] = {
            "original_data": data,
            "fact_check_result": {
                "status": "failed",
                "message": "An unexpected error occurred during fact-checking."
            }
        }

        # Validate input data format
        if not isinstance(data, dict):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected dict, got {type(data).__name__}.")
            result["fact_check_result"].update({
                "message": "Input data must be a dictionary.",
                "error_type": "InvalidInputTypeError"
            })
            return result

        text_to_check: Union[str, None] = data.get("text")
        if not isinstance(text_to_check, str) or not text_to_check.strip():
            logger.error(f"[{self.node_name}] 'text' key missing or not a non-empty string in input data.")
            result["fact_check_result"].update({
                "message": "Input data dictionary must contain a non-empty 'text' string.",
                "error_type": "MissingOrInvalidTextError"
            })
            return result

        fact_check_api_url: Union[str, None] = context.get("fact_check_api_url")
        known_facts: List[str] = context.get("known_facts", [])

        # --- Fact-checking logic ---
        try:
            if fact_check_api_url:
                logger.info(f"[{self.node_name}] Simulating external fact-check API call to: {fact_check_api_url}")
                # In a real-world scenario, 'requests' or an equivalent HTTP client
                # would be used here to call the external API.
                # For this simulation, we'll provide a placeholder response.
                
                simulated_status = "unverified"
                simulated_message = "Content sent to external API for verification (simulated response)."
                
                # Simple keyword-based simulation for a "verified" status
                if "Vishustra" in text_to_check or "LLM orchestration" in text_to_check:
                    simulated_status = "verified"
                    simulated_message = "Content aligns with known Vishustra context (simulated API verification)."
                elif "untrue" in text_to_check.lower() or "false" in text_to_check.lower():
                    simulated_status = "partially_verified" # Or could be 'disputed' / 'false' in a real system
                    simulated_message = "Content contains potentially disputed claims (simulated API response)."


                result["fact_check_result"].update({
                    "status": simulated_status,
                    "message": simulated_message,
                    "api_endpoint": fact_check_api_url
                })
                logger.info(f"[{self.node_name}] External API simulation complete. Status: {simulated_status}")

            elif known_facts:
                logger.info(f"[{self.node_name}] Performing internal fact-check against {len(known_facts)} known facts.")
                verified_facts_found: List[str] = []
                
                for fact in known_facts:
                    if fact.lower() in text_to_check.lower():
                        verified_facts_found.append(fact)
                
                if verified_facts_found:
                    status = "partially_verified"
                    message = f"Found {len(verified_facts_found)} matching known facts in the content. (Internal check)."
                    if len(verified_facts_found) == len(known_facts):
                        status = "verified"
                        message = "All known facts found in the content. (Internal check)."
                else:
                    status = "unverified"
                    message = "No matching known facts found in the content. (Internal check)."

                result["fact_check_result"].update({
                    "status": status,
                    "message": message,
                    "known_facts_checked": known_facts,
                    "verified_facts_found": verified_facts_found
                })
                logger.info(f"[{self.node_name}] Internal fact-check complete. Status: {status}")
            else:
                logger.warning(f"[{self.node_name}] No fact-check API URL or known facts provided in context. Cannot perform verification.")
                result["fact_check_result"].update({
                    "status": "unverified",
                    "message": "No fact-checking configuration found (neither API URL nor internal known facts). Content remains unverified."
                })
            
            # If we reached here, the process completed without critical errors, update status if still 'failed'
            if result["fact_check_result"]["status"] == "failed":
                 result["fact_check_result"].update({
                    "status": "unverified", # Default to unverified if no specific check was done but no hard error either
                    "message": "Fact-checking process completed, but no specific verification method was configured."
                 })

        except Exception as e:
            logger.exception(f"[{self.node_name}] An unexpected exception occurred during fact-checking process.")
            result["fact_check_result"].update({
                "status": "failed",
                "message": f"An unhandled exception occurred: {type(e).__name__} - {e}",
                "error_type": type(e).__name__
            })
            
        logger.debug(f"[{self.node_name}] Process finished. Final status: {result['fact_check_result']['status']}")
        return result