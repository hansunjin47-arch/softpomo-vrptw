import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from R_llm_module import parse_llm_confidence

# Simulate the multiline JSON format the LLM actually returns
test_response = '''
```json
{
  "confidence": {
    "92": 0.85,
    "93": 0.80,
    "94": 0.20,
    "95": 0.21,
    "17": 0.60,
    "21": 0.95
  }
}
```
'''

feasible = [92, 93, 94, 95, 17, 21]
scores, reasoning = parse_llm_confidence(test_response, feasible)
print('Scores:', scores)
print('All 0.5?', all(v == 0.5 for v in scores.values()))

# Also test the inline format
test_inline = '{"confidence": {"92": 0.85, "93": 0.80}}'
scores2, _ = parse_llm_confidence(test_inline, [92, 93])
print('Inline scores:', scores2)
print('OK')
