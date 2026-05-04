import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any

import numpy as np

# Configure logging for the module
logger = logging.getLogger(__name__)
# By default, handlers are not set. The user of the library would configure it.
# We'll set a NullHandler to prevent "No handlers could be found for logger" messages.
logger.addHandler(logging.NullHandler())


class EmbeddingModel(ABC):
    """
    Abstract Base Class for embedding models.

    Concrete implementations should provide methods to generate vector embeddings
    for single texts and batches of texts. These embeddings are crucial for
    calculating semantic similarity.
    """

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """
        Asynchronously generates a numerical vector embedding for a single text string.

        Args:
            text: The input text to be embedded.

        Returns:
            A list of floats representing the embedding vector.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously generates numerical vector embeddings for a batch of text strings.

        Args:
            texts: A list of input texts to be embedded.

        Returns:
            A list of lists of floats, where each inner list is an embedding vector
            corresponding to the input text at the same index.
        """
        pass


class Route:
    """
    Represents a potential routing destination within the Vishustra framework.

    Each route defines a logical path, an explicit target identifier, and
    a set of example phrases that semantically define when this route should be taken.
    """

    def __init__(self, name: str, target: str, examples: List[str], description: Optional[str] = None):
        """
        Initializes a new Route.

        Args:
            name: A unique, human-readable name for the route (e.g., "summarize_document").
            target: An identifier pointing to the actual component to be invoked
                    when this route is matched (e.g., "llm_chain:summarizer", "agent:customer_support").
            examples: A list of natural language phrases that exemplify queries
                      or prompts that should activate this route. These are used
                      to build the semantic index.
            description: An optional longer description of what this route does.
        """
        if not name or not target:
            raise ValueError("Route 'name' and 'target' cannot be empty.")
        if not examples:
            logger.warning(f"Route '{name}' has no examples. It will never be matched by the SemanticRouter.")

        self.name = name
        self.target = target
        self.examples = examples
        self.description = description

    def __repr__(self):
        return f"Route(name='{self.name}', target='{self.target}')"


class RouteMatch:
    """
    Represents a successful match found by the SemanticRouter.

    Contains the matched Route, the similarity score, and the specific example
    from the route that yielded the highest score.
    """

    def __init__(self, route: Route, score: float, matched_example: Optional[str] = None):
        """
        Initializes a RouteMatch.

        Args:
            route: The Route object that was matched.
            score: The similarity score (e.g., cosine similarity) of the match.
            matched_example: The specific example phrase from the route that had
                             the highest similarity to the input query.
        """
        self.route = route
        self.score = score
        self.matched_example = matched_example

    def __repr__(self):
        return (
            f"RouteMatch(route_name='{self.route.name}', target='{self.route.target}', "
            f"score={self.score:.4f}, example='{self.matched_example}')"
        )


class SemanticRouter:
    """
    A sophisticated routing mechanism for Vishustra, designed to direct user queries
    or internal prompts to the most semantically relevant processing path (Route).

    This router leverages an `EmbeddingModel` to transform textual inputs and
    predefined route examples into high-dimensional vectors. It then uses
    cosine similarity to determine the closest match, providing a flexible
    and robust way to orchestrate complex LLM workflows.
    """

    _embeddings: Optional[np.ndarray]  # Stores normalized example embeddings
    _route_map: List[Tuple[Route, str]]  # Maps (Route, example_text) to index in _embeddings

    def __init__(self,
                 embedding_model: EmbeddingModel,
                 routes: List[Route],
                 similarity_threshold: float = 0.75):
        """
        Initializes the SemanticRouter.

        Args:
            embedding_model: An instance of an EmbeddingModel to convert text to vectors.
            routes: A list of Route objects defining the routing possibilities.
            similarity_threshold: The minimum cosine similarity score required for a match
                                  to be considered valid. Scores range from 0 (orthogonal)
                                  to 1 (identical) for normalized vectors. A higher
                                  threshold means stricter matching.
        """
        if not isinstance(embedding_model, EmbeddingModel):
            raise TypeError("embedding_model must be an instance of EmbeddingModel.")
        if not isinstance(routes, list) or not all(isinstance(r, Route) for r in routes):
            raise TypeError("routes must be a list of Route objects.")
        if not 0.0 <= similarity_threshold <= 1.0:
            logger.warning(
                f"Similarity threshold {similarity_threshold} is outside the typical [0.0, 1.0] range for "
                "cosine similarity (where 1 is identical and 0 is orthogonal). Consider adjusting."
            )

        self._embedding_model = embedding_model
        self._routes = routes
        self._similarity_threshold = similarity_threshold
        self._embeddings = None
        self._route_map = []
        self._initialized = False

    async def initialize(self):
        """
        Asynchronously builds the internal vector index from all route examples.
        This method should be called once after instantiation to prepare the router.
        """
        if self._initialized:
            logger.debug("Router index already built.")
            return

        all_example_texts: List[str] = []
        # Store (route_object, example_text) for each embedding
        index_to_route_map: List[Tuple[Route, str]] = []

        for route in self._routes:
            for example in route.examples:
                all_example_texts.append(example)
                index_to_route_map.append((route, example))

        if not all_example_texts:
            logger.warning("No examples provided across all routes. Router will not be able to match any query.")
            self._initialized = True
            return

        logger.info(f"Building router index with {len(all_example_texts)} examples...")
        try:
            embeddings_list = await self._embedding_model.embed_batch(all_example_texts)
            raw_embeddings = np.array(embeddings_list, dtype=np.float32)

            # Normalize embeddings. Handle potential zero norms to prevent NaNs.
            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            self._embeddings = np.where(norms == 0, raw_embeddings, raw_embeddings / norms)

            self._route_map = index_to_route_map
            self._initialized = True
            logger.info(f"Router index built successfully with {self._embeddings.shape[0]} embeddings.")
        except Exception as e:
            logger.error(f"Failed to build router index: {e}", exc_info=True)
            raise

    async def route(self, query: str) -> Optional[RouteMatch]:
        """
        Asynchronously routes a given query to the most semantically similar route.

        If the router has not been initialized, it will attempt to initialize itself.

        Args:
            query: The input query string to be routed.

        Returns:
            A `RouteMatch` object if a suitable route is found above the configured
            similarity threshold, otherwise `None`.
        """
        if not self._initialized:
            logger.warning("Router not initialized. Attempting to initialize now.")
            await self.initialize()

        if self._embeddings is None or self._embeddings.shape[0] == 0:
            logger.warning("Router index is empty. Cannot route query.")
            return None

        logger.debug(f"Attempting to route query: '{query}'")

        query_embedding_list = await self._embedding_model.embed(query)
        query_embedding_np = np.array(query_embedding_list, dtype=np.float32)

        # Normalize the query embedding
        query_norm = np.linalg.norm(query_embedding_np)
        if query_norm == 0:
            logger.warning(f"Query '{query}' resulted in a zero-norm embedding. Cannot calculate similarity.")
            return None
        query_embedding_normalized = query_embedding_np / query_norm

        # Calculate cosine similarities using dot product (since vectors are normalized)
        # np.dot(A, B) for A (M, N) and B (N,) results in (M,)
        similarities = np.dot(self._embeddings, query_embedding_normalized)

        # Find the best match
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        best_route, matched_example_text = self._route_map[best_match_idx]

        logger.debug(
            f"Best match candidate: Route='{best_route.name}', Score={best_score:.4f}, "
            f"Matched example='{matched_example_text}'"
        )

        if best_score >= self._similarity_threshold:
            logger.info(f"Query routed to '{best_route.name}' with score {best_score:.4f}.")
            return RouteMatch(route=best_route, score=best_score, matched_example=matched_example_text)
        else:
            logger.info(
                f"No route found above threshold {self._similarity_threshold:.4f}. "
                f"Best score was {best_score:.4f} for route '{best_route.name}'."
            )
            return None


# --- Example Dummy Embedding Model (for demonstration purposes within Vishustra context) ---
class DummyEmbeddingModel(EmbeddingModel):
    """
    A placeholder embedding model for internal testing and demonstration within Vishustra.
    Generates deterministic, non-semantic embeddings based on a hash of the text.
    In a production Vishustra deployment, this would be replaced by an actual
    LLM-based embedding service (e.g., OpenAI, Cohere, HuggingFace Transformers).
    """

    def __init__(self, vector_dim: int = 128):
        """
        Initializes the DummyEmbeddingModel.

        Args:
            vector_dim: The dimension of the dummy embedding vectors to generate.
        """
        self._vector_dim = vector_dim
        logger.warning(
            "Using DummyEmbeddingModel. This is for testing only and does not "
            "provide meaningful semantic embeddings for real-world applications."
        )

    async def embed(self, text: str) -> List[float]:
        """
        Generates a dummy embedding for a single text.
        The vector is pseudo-random but deterministic for a given text, and normalized.
        """
        if not text:
            return [0.0] * self._vector_dim # Return zero vector for empty text
        
        # Use a consistent seed for reproducibility given the same text
        # Using a hash ensures some variation for different texts
        seed_val = hash(text) % (2**32 - 1)
        rng = np.random.default_rng(seed=seed_val)
        
        vec = rng.random(self._vector_dim) * 2 - 1  # Values between -1 and 1
        
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist() # Return as is if it's a zero vector
        return (vec / norm).tolist() # Normalize to a unit vector

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates dummy embeddings for a batch of texts by calling `embed` for each.
        """
        # For a real model, this would be optimized for batch inference.
        return [await self.embed(text) for text in texts]


# Example usage block, demonstrating how Vishustra might use this module
if __name__ == "__main__":
    # Setup basic logging for the example to make outputs visible
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Set the module's logger to INFO if it's not already configured by basicConfig
    logger.setLevel(logging.INFO)

    async def run_example():
        logger.info("Starting SemanticRouter example for Vishustra...")

        # 1. Instantiate an embedding model (e.g., connecting to OpenAI, HuggingFace locally)
        # For this example, we use the DummyEmbeddingModel.
        embedding_model = DummyEmbeddingModel(vector_dim=128)

        # 2. Define various 'routes' that Vishustra could take
        routes = [
            Route(
                name="summarize_document",
                target="llm_chain:summarizer",
                examples=[
                    "summarize this document for me",
                    "give me a summary of the provided text",
                    "can you condense this information?",
                    "what's the main idea of this article?"
                ],
                description="Routes to an LLM chain optimized for summarizing long documents."
            ),
            Route(
                name="answer_faq",
                target="agent:faq_agent",
                examples=[
                    "what are your operating hours?",
                    "how do I reset my password?",
                    "tell me about your return policy",
                    "where can I find technical support?"
                ],
                description="Routes to a dedicated agent for frequently asked questions."
            ),
            Route(
                name="code_generation",
                target="tool:code_generator",
                examples=[
                    "write a python function to sort a list of numbers",
                    "generate a SQL query to select all active users",
                    "create a javascript snippet for a button click event"
                ],
                description="Routes to a specialized tool for generating code snippets."
            ),
            Route(
                name="sentiment_analysis",
                target="model:sentiment_classifier",
                examples=[
                    "what is the sentiment of this review?",
                    "is this text positive, negative, or neutral?",
                    "analyze the emotional tone of this message"
                ],
                description="Routes to a fine-tuned model for sentiment analysis."
            ),
            Route(
                name="unknown_query_handler",
                target="agent:default_fallback_agent",
                examples=[
                    "I don't know what to ask",
                    "can you help me with anything?",
                    "what else can you do?",
                    "I am lost"
                ],
                description="A fallback route for queries that don't match specific intents."
            )
        ]

        # 3. Instantiate the router with the model and routes
        router = SemanticRouter(
            embedding_model=embedding_model,
            routes=routes,
            similarity_threshold=0.7 # A lower threshold might increase matches with dummy model
        )

        # 4. Initialize the router (builds the internal vector index)
        await router.initialize()

        # 5. Test routing with various queries
        test_queries = [
            "Please provide a quick summary of the report.",  # -> summarize_document
            "I need to know how to change my account password.",  # -> answer_faq
            "Can you write some Java code to reverse a string?",  # -> code_generation
            "How do you feel about this movie review? Is it positive or negative?",  # -> sentiment_analysis
            "What is the capital of France?",  # -> likely unknown_query_handler or None
            "Tell me a joke.",  # -> likely unknown_query_handler or None
            "I need a summary", # -> summarize_document
            "Help!" # -> unknown_query_handler
        ]

        print("\n--- Starting Routing Tests ---")
        for i, query in enumerate(test_queries):
            print(f"\nQUERY {i+1}: '{query}'")
            match = await router.route(query)
            if match:
                print(f"  ✅ Routed to: '{match.route.name}' (Target: '{match.route.target}')")
                print(f"     Score: {match.score:.4f}")
                print(f"     Matched example: '{match.matched_example}'")
            else:
                print(f"  ❌ No suitable route found for query '{query}' above threshold {router._similarity_threshold:.4f}.")
        print("\n--- Routing Tests Complete ---")

    # Run the asynchronous example
    try:
        asyncio.run(run_example())
    except RuntimeError as e:
        if "cannot run an asyncio event loop while another loop is running" in str(e):
            # This handles cases where the script might be run in an environment
            # (like Jupyter notebooks) where an event loop is already active.
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.get_event_loop().run_until_complete(run_example())
        else:
            raise