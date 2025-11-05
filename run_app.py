import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")

def main():
    # Check if requirements are installed
    try:
        import streamlit
        import sklearn
        import pandas
    except ImportError:
        print("Installing requirements...")
        install_requirements()
    
    # Run the Streamlit app
    os.system("streamlit run app.py")

if __name__ == "__main__":
    main()