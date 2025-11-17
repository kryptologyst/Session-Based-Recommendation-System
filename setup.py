#!/usr/bin/env python3
"""Setup script for session-based recommendation system."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Session-Based Recommendation System")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing dependencies"),
        ("pip install -e .", "Installing package in development mode"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"❌ Setup failed at: {description}")
            sys.exit(1)
    
    # Create necessary directories
    directories = ["data", "results", "models", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Generate sample data
    print("\n🔄 Generating sample data...")
    try:
        subprocess.run([
            sys.executable, "scripts/train.py", 
            "--generate_data", 
            "--config", "configs/default.yaml",
            "--output_path", "results"
        ], check=True)
        print("✅ Sample data generated successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Sample data generation failed, but setup is complete")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the interactive demo:")
    print("   streamlit run scripts/demo.py")
    print("\n2. Train models:")
    print("   python scripts/train.py --generate_data")
    print("\n3. Run tests:")
    print("   pytest tests/")
    print("\n4. Explore the notebook:")
    print("   jupyter notebook notebooks/session_rec_analysis.ipynb")


if __name__ == "__main__":
    main()
