import sys
import json
import time
import random

def analyze_content(content, input_type):
    # Simulate processing time
    time.sleep(1.5)
    
    # Simple demo analysis
    is_url = input_type == "url"
    content_length = len(content)
    
    # Demo logic - in a real app, you'd implement actual analysis here
    confidence = random.randint(65, 95)
    is_real = confidence > 80
    
    return {
        "prediction": "Likely Real" if is_real else "Potentially Fake",
        "type": "real" if is_real else "fake",
        "confidence": confidence
    }

if __name__ == "__main__":
    # Read input from stdin
    input_data = sys.stdin.read()
    try:
        data = json.loads(input_data)
        content = data.get("content", "")
        input_type = data.get("inputType", "text")
        
        result = analyze_content(content, input_type)
        
        # Output result as JSON
        print(json.dumps(result))