#!/usr/bin/env python3
# test_gemini_models.py
import google.generativeai as genai

API_KEY = "AIzaSyCzxLnZ00biLcjHXDQcK0-x0FhsfIi7ZAM"  # Your actual key

def test_all_models_for_status(): # Renamed the function for clarity
    genai.configure(api_key=API_KEY)
    
    models_to_test = [
        'gemini-2.5-flash',       # SUCCESS (tested)
        'gemini-2.5-pro',         # Should succeed
        'gemini-2.5-flash-lite',  # Should succeed
        'gemini-1.5-flash-latest', # Should FAIL (404)
        'gemini-1.5-flash',       # Should FAIL (404)
        'gemini-1.0-pro-latest',  # Should FAIL (404)
        'gemini-pro',             # Should FAIL (404)
        'models/gemini-1.5-flash-latest' # Should FAIL (404)
    ]
    
    # We will use a list to collect the results instead of returning immediately
    results = [] 

    print("--- Starting Full Model Availability Check ---")
    for model_name in models_to_test:
        try:
            print(f"Testing: {model_name}")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'Hello World'")
            print(f"✅ SUCCESS: {model_name}")
            # Do NOT return here, just append the success
            results.append((model_name, "SUCCESS")) 
        except Exception as e:
            print(f"❌ FAILED: {model_name} - {e}")
            results.append((model_name, f"FAILED: {e}"))
    
    print("--- Check Complete ---")
    return results # Return all results at the end

if __name__ == "__main__":
    all_results = test_all_models_for_status()
    print("\n🎉 Summary of All Tests:")
    for name, status in all_results:
        print(f"* {name}: {status}")