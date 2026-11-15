"""
run.py — VitalizeHealth entry point
Run with:  python run.py
"""

import os
import sys

# Load .env variables BEFORE importing the app
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_anthropic_api_key_here":
        print("⚠️  No ANTHROPIC_API_KEY set — AI plan will use ML offline fallback.")
        print("   To enable AI plans: edit .env and add your key.\n")
    else:
        print("✅  Anthropic API key detected — full AI plans enabled.\n")

    print(f"🚀  VitalizeHealth running at → http://localhost:{port}")
    print("    Press Ctrl+C to stop\n")

    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=False)
