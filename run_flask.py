"""
Run the Flask web application.

Usage:
    python run_flask.py
"""

import os
import sys
import importlib.util

if __name__ == '__main__':
    # Add project root to path so frontend/app.py can import from src
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Load the Flask app from frontend/app.py
    app_path = os.path.join(project_root, 'frontend', 'app.py')
    spec = importlib.util.spec_from_file_location("app", app_path)
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    app = app_module.app

    print("=" * 60)
    print("Starting Continuous Prompting Flask App")
    print("=" * 60)
    print("\nOpen your browser to: http://localhost:5000")
    print("\nFeatures:")
    print("   - Real-time updates (no page reloads)")
    print("   - Live stock prices")
    print("   - Live charts")
    print("   - LLM responses as they happen")
    print("   - Portfolio tracking")
    print("\n" + "=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

