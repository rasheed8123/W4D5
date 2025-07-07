#!/usr/bin/env python3
"""
Setup script for the Intelligent Document Chunking Agent.
This script helps install dependencies and set up the environment.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible!")
    return True

def install_requirements():
    """Install Python requirements."""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def install_spacy_model():
    """Install spaCy English model."""
    return run_command("python -m spacy download en_core_web_sm", "Installing spaCy English model")

def create_directories():
    """Create necessary directories."""
    directories = ["data/raw", "data/processed", "models", "chroma_db"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")

def main():
    """Main setup function."""
    print("🧠 Setting up Intelligent Document Chunking Agent")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Install spaCy model
    if not install_spacy_model():
        print("⚠️  Warning: Failed to install spaCy model. Some features may not work properly.")
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("1. Run the application: streamlit run app.py")
    print("2. Open your browser and navigate to the provided URL")
    print("3. Click 'Initialize Pipeline' to get started")
    print("4. Upload documents and start chunking!")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 