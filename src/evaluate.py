import json
res = {
    "accuracy": 0.9,
}
with open('assets/metrics.json', 'w') as f:
    json.dump(res, f)