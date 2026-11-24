"""Bounded retries for temporary Google service unavailability."""
import random
import time

from google.genai.errors import APIError


def is_google_unavailable(error):
    """Recognize a Google HTTP 503, including provider-wrapped errors."""
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, APIError) and current.code == 503:
            return True
        current = current.__cause__
    return False


def retry_google_unavailable(operation, *, attempts=4):
    """Retry only Google 503s, using bounded exponential backoff with jitter."""
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:
            if not is_google_unavailable(error) or attempt == attempts - 1:
                raise
            time.sleep(min(4.0, 0.5 * 2 ** attempt) + random.uniform(0, 0.25))
