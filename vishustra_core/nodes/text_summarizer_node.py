import logging
from typing import Any, Dict

# Assuming the BaseNode is located at this path within the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TextSummarizerNode(BaseNode):
    """
    A Vishustra processing node that performs abstractive text summarization.

    This node is designed to take a block of text and produce a concise,
    contextually relevant summary. In a production Vishustra deployment,
    this node would integrate with an underlying Language Model (LLM)
    or a dedicated summarization service to perform its task.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "TextSummarizerNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text to generate a summary.

        This method validates the input and, in a real scenario, would invoke
        an external summarization model or LLM. For demonstration, it simulates
        this process by returning a truncated or slightly modified version of
        the input, while adhering to specified length constraints from the context.

        Args:
            data (Any): The input text to be summarized. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       or configuration for the summarization process.
                                       This can include parameters like `min_summary_length`,
                                       `max_summary_length`, or specific model configurations.

        Returns:
            Any: The summarized text as a string.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is empty or too short for meaningful summarization.
        """
        if not isinstance(data, str):
            error_msg = (
                f"Invalid input type for {self.node_name}. "
                f"Expected `str`, but received `{type(data).__name__}`."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not data.strip():
            error_msg = f"Input text for {self.node_name} cannot be empty or whitespace-only."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Retrieve summarization parameters from context with sensible defaults
        min_length = context.get("min_summary_length", 50)
        max_length = context.get("max_summary_length", 300)
        
        logger.debug(
            f"Starting summarization process for {self.node_name}. "
            f"Input length: {len(data)} characters. "
            f"Target min/max summary length: {min_length}/{max_length}."
        )

        # --- Simulated Summarization Logic ---
        # In a real Vishustra integration, this section would contain calls to
        # an LLM API (e.g., OpenAI, Hugging Face, custom model) or a specialized
        # summarization library, configured and authenticated via the context.
        #
        # For this simulation, we'll create a simple "summary" based on truncation
        # and sentence extraction to mimic abstractive output structure.

        sentences = [s.strip() for s in data.split('.') if s.strip()]
        
        # If the input is very short, a 'summary' might just be the original text or a slight truncation.
        if len(data) < min_length * 1.5: # Heuristic: if original text is not much longer than desired min summary.
            logger.warning(
                f"Input text for {self.node_name} is relatively short ({len(data)} chars). "
                f"Returning a truncated version of the original text as a simulated summary."
            )
            simulated_summary = data[:max_length]
            if len(data) > max_length:
                simulated_summary += "..."
        else:
            # Simulate abstractive summary by taking a portion of the original text
            # and adding an ellipsis if truncated, ensuring it respects max_length.
            # This is a very simplistic stand-in for complex LLM output.
            
            # Aim for roughly 1/3 to 1/2 of the original content in sentences, or enough to meet min_length.
            target_sentences_count = max(
                1, 
                min(len(sentences) // 2, len(sentences) - 1) # Cap at half sentences or one less than total
            )
            
            simulated_summary_parts = []
            current_length = 0
            for sentence in sentences:
                if current_length + len(sentence) + 2 > max_length and simulated_summary_parts: # +2 for ". "
                    break
                simulated_summary_parts.append(sentence)
                current_length += len(sentence) + 2 # Account for ". "

            if not simulated_summary_parts: # Fallback if initial loop didn't add anything
                simulated_summary = data[:max_length]
                if len(data) > max_length:
                    simulated_summary += "..."
            else:
                simulated_summary = ". ".join(simulated_summary_parts)
                if len(simulated_summary) < len(data) and len(simulated_summary) < max_length:
                     simulated_summary += "." # Add final period if it was removed by join and not truncated.
                
                # Ensure it adheres to max_length and min_length as much as possible for simulation
                if len(simulated_summary) > max_length:
                    simulated_summary = simulated_summary[:max_length - 3].strip() + "..."
                elif len(simulated_summary) < min_length and len(data) > min_length:
                    # If the simulated summary is too short, try to grab a bit more, or acknowledge.
                    logger.warning(
                        f"Simulated summary for {self.node_name} is shorter than "
                        f"min_summary_length ({min_length} chars), despite longer input. "
                        f"This often indicates a highly concise original text or a very aggressive summarization parameter."
                    )
                    # For a robust simulation, we might try to extend it here if possible.
                    # For simplicity, we just log and return what we have.

        logger.debug(
            f"Successfully simulated summarization for {self.node_name}. "
            f"Final summary length: {len(simulated_summary)} characters."
        )
        return simulated_summary