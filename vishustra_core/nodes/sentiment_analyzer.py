import logging
from typing import Any, Dict, Union

# Assuming BaseNode is available at this path as per project context
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class SentimentAnalyzer(BaseNode):
    """
    A Vishustra processing node that performs sentiment analysis on input text.
    It expects the input 'data' to be either a string or a dictionary
    containing a 'text' key.
    
    This node simulates sentiment analysis using a keyword-based approach.
    In a production environment, this would integrate with a robust NLP library
    (e.g., NLTK, spaCy, or a cloud NLP service API).
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "SentimentAnalyzer"

    def _analyze_sentiment_keywords(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Simulates sentiment analysis using a simple keyword-based approach.
        Assigns a sentiment ('positive', 'negative', 'neutral') and a score
        (ranging approximately from -1.0 to 1.0).
        """
        text_lower = text.lower()
        score = 0.0

        positive_keywords = {
            "good": 0.5, "great": 0.7, "excellent": 0.9, "fantastic": 0.95,
            "love": 0.8, "amazing": 0.85, "happy": 0.6, "wonderful": 0.75,
            "superb": 0.8, "perfect": 0.9, "awesome": 0.7
        }
        negative_keywords = {
            "bad": -0.5, "terrible": -0.7, "poor": -0.6, "disappointing": -0.75,
            "hate": -0.8, "awful": -0.9, "unhappy": -0.65, "dreadful": -0.8,
            "horrible": -0.85, "ugly": -0.5, "worst": -0.9
        }

        for keyword, val in positive_keywords.items():
            if keyword in text_lower:
                score += val
        for keyword, val in negative_keywords.items():
            if keyword in text_lower:
                score += val

        if score > 0.2:
            sentiment = "positive"
        elif score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Scale the score to be within a more standard -1.0 to 1.0 range
        # based on the arbitrary keyword weights. Max theoretical score: ~6.0, Min: ~-6.0
        scaled_score = max(-1.0, min(1.0, score / 6.0)) 

        return {"sentiment": sentiment, "score": scaled_score}

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to determine its sentiment.

        Args:
            data (Any): The input data. Expected to be a string or a dictionary
                        with a 'text' key containing a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       for the processing (currently unused by this node).

        Returns:
            Any: The original data enriched with sentiment information.
                 If input was a string, returns a dictionary: 
                 `{'text': original_string, 'sentiment': ..., 'sentiment_score': ...}`.
                 If input was a dictionary, returns a *copy* of that dictionary 
                 with `sentiment` and `sentiment_score` keys added.

        Raises:
            ValueError: If the input data is not a string or a dictionary
                        with a 'text' key containing a string.
            RuntimeError: If an unexpected error occurs during sentiment analysis.
        """
        input_text = None
        return_data: Dict[str, Any] = {} # Prepare a dictionary for consistent output structure

        if isinstance(data, str):
            input_text = data
            return_data = {"text": data} # Wrap string input in a dict
        elif isinstance(data, dict):
            if "text" in data and isinstance(data["text"], str):
                input_text = data["text"]
                return_data = data.copy() # Work on a copy to avoid modifying original input reference
            else:
                error_msg = (
                    "Input dictionary for SentimentAnalyzer missing 'text' key or 'text' is not a string. "
                    f"Received data: {data!r}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = (
                f"Invalid input type for SentimentAnalyzer. Expected string or dict, got {type(data).__name__!r}. "
                f"Received data: {data!r}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            "Processing sentiment for text (first 75 chars): '%s%s'", 
            input_text[:75], 
            "..." if len(input_text) > 75 else ""
        )

        try:
            sentiment_result = self._analyze_sentiment_keywords(input_text)
            return_data["sentiment"] = sentiment_result["sentiment"]
            return_data["sentiment_score"] = sentiment_result["score"]
            logger.debug(
                "Sentiment analysis complete for text. Result: %s", sentiment_result
            )
            return return_data
        except Exception as e:
            logger.exception(
                "An unexpected error occurred during sentiment analysis for data: %s, Context: %s", 
                data, context
            )
            raise RuntimeError(f"Failed to analyze sentiment: {e}") from e