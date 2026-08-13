import io
import http.client
from unittest import mock
import unittest
import urllib.error

from commentgap_scraper.api import forum_info_query, threads_query
from commentgap_scraper.http import HttpClient, RateLimiter


class HttpApiTests(unittest.TestCase):
    def test_query_contains_requested_reply_depth(self):
        query = threads_query(4)
        self.assertEqual(query.count("replies {"), 4)
        self.assertIn("getForumRootPostingsV2", query)
        self.assertIn("stickyPostings", forum_info_query(2))

    def test_rate_limiter_waits_between_start_times(self):
        times = iter([10.0, 10.0, 10.25, 11.0])
        limiter = RateLimiter(1.0, clock=lambda: next(times))
        with mock.patch("commentgap_scraper.http.time.sleep") as sleep:
            limiter.wait()
            limiter.wait()
        sleep.assert_called_once_with(0.75)

    def test_http_client_retries_429_and_honors_retry_after(self):
        error = urllib.error.HTTPError(
            "https://example.test",
            429,
            "rate limited",
            {"Retry-After": "2"},
            io.BytesIO(b""),
        )
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = b"ok"
        client = HttpClient("Research crawler (contact: x@example.org)", max_retries=1)
        with mock.patch("commentgap_scraper.http.urllib.request.urlopen", side_effect=[error, response]), \
             mock.patch.object(client._limiter, "wait"), \
             mock.patch("commentgap_scraper.http.time.sleep") as sleep, \
             mock.patch("commentgap_scraper.http.random.uniform", return_value=0.0):
            self.assertEqual(client.request("https://example.test"), b"ok")
        sleep.assert_called_once_with(2.0)

    def test_http_client_retries_truncated_response(self):
        truncated = mock.MagicMock()
        truncated.__enter__.return_value.read.side_effect = http.client.IncompleteRead(
            b"partial", 20
        )
        complete = mock.MagicMock()
        complete.__enter__.return_value.read.return_value = b"complete"
        client = HttpClient("Research crawler (contact: x@example.org)", max_retries=1)
        with mock.patch(
            "commentgap_scraper.http.urllib.request.urlopen",
            side_effect=[truncated, complete],
        ), mock.patch.object(client._limiter, "wait"), mock.patch(
            "commentgap_scraper.http.time.sleep"
        ) as sleep, mock.patch(
            "commentgap_scraper.http.random.uniform", return_value=0.0
        ):
            self.assertEqual(client.request("https://example.test"), b"complete")
        sleep.assert_called_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
