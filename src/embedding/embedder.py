import time

from openai import APIConnectionError, OpenAI, RateLimitError

MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class Embedder:
    # unlike bge, OpenAI embeddings need no query prefix: queries and
    # documents go through the same call.
    def __init__(self, model_name: str = MODEL_NAME):
        self._client = OpenAI()
        self._model = model_name

    def embed_documents(self, texts: list[str], max_attempts: int = 6) -> list[list[float]]:
        # the SDK's own retries back off for a few seconds at most, which isn't
        # enough when parallel workers keep refilling the 1M TPM window, or
        # when a flaky connection drops a request (APITimeoutError subclasses
        # APIConnectionError).
        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.embeddings.create(model=self._model, input=texts)
                return [item.embedding for item in response.data]
            except (RateLimitError, APIConnectionError):
                if attempt == max_attempts:
                    raise
                time.sleep(15 * attempt)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
