#!/usr/bin/env python3
"""
Launch the ML Trading Dashboard
Run: python launch_dashboard.py
"""

import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed"""
    required = ['streamlit', 'plotly', 'pandas', 'numpy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"📦 Installing missing dependencies: {', '.join(missing)}")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q'] + missing,
            check=True
        )
        print("✅ Dependencies installed")


def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching ML Trading Dashboard...")
    print("")
    print("📊 Dashboard will open at: http://localhost:8501")
    print("📝 Press Ctrl+C to stop")
    print("")
    
    # Get the script path
    script_dir = Path(__file__).parent
    dashboard_path = script_dir / "ml" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Error: {dashboard_path} not found")
        return False
    
    # Launch streamlit
    try:
        subprocess.run(
            [sys.executable, '-m', 'streamlit', 'run', str(dashboard_path)],
            cwd=str(script_dir),
            check=True
        )
        return True
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped")
        return True
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False


if __name__ == '__main__':
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Check dependencies
    check_dependencies()
    
    # Launch dashboard
    success = launch_dashboard()
    sys.exit(0 if success else 1)
