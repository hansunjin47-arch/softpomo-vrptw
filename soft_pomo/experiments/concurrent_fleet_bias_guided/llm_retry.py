"""Bounded, audited regeneration of failed LLM responses only."""
import time
import urllib.error
from pathlib import Path
from experiments.concurrent_fleet_v9_guided.online import OpenRouterClient
from experiments.concurrent_fleet_pair.io import write


class RetryingClient:
    def __init__(self, folder, validate, *, allow_paid=True, fallback=None):
        self.folder = Path(folder)
        self.validate = validate
        self.allow_paid = allow_paid
        self.fallback = fallback

    def __call__(self, request):
        if self.fallback is not None:
            try:
                payload = self.fallback(request)
                self.validate(request, payload)
                return payload
            except (RuntimeError, ValueError, KeyError, TypeError, AssertionError):
                pass
        errors = []
        for attempt in range(3):
            folder = self.folder / ('attempt_' + str(attempt + 1))
            try:
                payload = OpenRouterClient(folder, allow_paid=self.allow_paid)(request)
                self.validate(request, payload)
                return payload
            except urllib.error.HTTPError as exc:
                if exc.code not in (408, 429) and exc.code < 500:
                    raise
                errors.append(repr(exc))
            except (ValueError, KeyError, TypeError, AssertionError,
                    urllib.error.URLError, TimeoutError, FileExistsError) as exc:
                errors.append(repr(exc))
            write(self.folder / 'last_retry.json', dict(attempt=attempt + 1, errors=errors,
                                                      request=request))
            if attempt < 2:
                time.sleep(15 * (attempt + 1))
        raise RuntimeError('LLM generation failed after 3 audited attempts: ' + str(errors))
