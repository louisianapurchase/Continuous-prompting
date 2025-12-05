"""
Unified launcher for Continuous Prompting Framework.

Usage:
    python run.py              # Run Flask web interface (default)
    python run.py --web        # Run Flask web interface
    python run.py --cli        # Run CLI interface
    python run.py --cli --max-iterations 50  # CLI with options
"""

import os
import sys
import argparse
import importlib.util


def run_web():
    """Run the Flask web application."""
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


def run_cli():
    """Run the CLI interface."""
    from src.app.cli import main
    sys.exit(main())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Continuous Prompting Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Run web interface (default)
  python run.py --web              # Run web interface
  python run.py --cli              # Run CLI interface
  python run.py --cli --max-iterations 50
        """
    )

    parser.add_argument(
        '--web',
        action='store_true',
        help='Run Flask web interface (default)'
    )

    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run CLI interface'
    )

    args, unknown = parser.parse_known_args()

    # Default to web if neither specified
    if not args.cli and not args.web:
        args.web = True

    if args.cli:
        # Pass remaining args to CLI
        sys.argv = [sys.argv[0]] + unknown
        run_cli()
    else:
        run_web()
