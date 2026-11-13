"""Offline regression tests: API errors never change sampling conditions."""
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import VRPTWLLMModule as llm


class SamplingTests(unittest.TestCase):
    def test_closing_brace_repair(self):
        from SoftClusterLLMModule import parse_all_clusters_confidence
        text = json.dumps({'confidence': {'1': {'1': 0.7, '0': 0.2}}})
        expected = parse_all_clusters_confidence(text, [[1]])
        self.assertEqual(parse_all_clusters_confidence(text[:-1], [[1]]), expected)
        self.assertEqual(parse_all_clusters_confidence('{"confidence": {"1": {"1":', [[1]]), ({}, {}))

    def invoke(self, responses):
        sent = []
        def fake(request, **kwargs):
            sent.append(json.loads(request.data))
            value = responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return io.BytesIO(json.dumps({'choices': [{'finish_reason': value,
                'message': {'content': '{"confidence":[]}'}}]}).encode())
        with patch.object(llm, 'OPENROUTER_API_KEY', 'test'), patch.object(llm.urllib.request, 'urlopen', fake), patch('time.sleep'):
            try:
                llm.query_llm('fixed prompt', 'vendor/test', response_format={'type': 'json_object'})
            except RuntimeError:
                return sent, True
        return sent, False

    def test_length_fails_without_resampling(self):
        sent, failed = self.invoke(['length'])
        self.assertTrue(failed)
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]['temperature'], 0.0)

    def test_network_retry_is_identical(self):
        sent, failed = self.invoke([urllib.error.URLError('temporary'), 'stop'])
        self.assertFalse(failed)
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0], sent[1])
        self.assertFalse(sent[0]['provider']['allow_fallbacks'])


if __name__ == '__main__':
    unittest.main()
