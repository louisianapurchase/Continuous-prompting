"""
Quick launcher for Web interface.
Usage: python run_web.py
This will launch: streamlit run src/app/web.py
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app/web.py"])

