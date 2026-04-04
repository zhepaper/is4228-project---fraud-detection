#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from explain import generate_explanation

# Test with the same values from your curl response
result = generate_explanation(
    fraud_probability=0.9232,
    risk_factors=[
        "No previous address history on record",
        "No banking history on record", 
        "Application submitted from a foreign IP address"
    ],
    decision="block",
    dataset="baf",
    feature_values={"income": 45000, "foreign_request": 1},  # minimal example
)

print("✅ Explanation result:")
print(f"  Method: {result['method']}")
print(f"  Text: {result['explanation']}")
print(f"  Confidence: {result['confidence']}")