#!/usr/bin/env python3
"""
Dashboard Test Script - Verify all components are working
Run: python test_dashboard.py
"""

import sys
import subprocess
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported"""
    print("🧪 Testing imports...")
    
    required_packages = {
        'streamlit': 'Streamlit',
        'plotly': 'Plotly',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
    }
    
    failed = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            failed.append(package)
    
    return len(failed) == 0, failed


def test_files():
    """Test that all required files exist"""
    print("\n🧪 Testing files...")
    
    base_path = Path(__file__).parent
    required_files = [
        'ml/dashboard.py',
        'ml/dashboard_utils.py',
        'ml/training_manager.py',
        'requirements-dashboard.txt',
        'DASHBOARD_README.md',
        'DASHBOARD_FEATURES.md',
        'launch_dashboard.py',
    ]
    
    failed = []
    for file in required_files:
        file_path = base_path / file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} (NOT FOUND)")
            failed.append(file)
    
    return len(failed) == 0, failed


def test_modules():
    """Test that Python modules are valid"""
    print("\n🧪 Testing Python modules...")
    
    base_path = Path(__file__).parent
    modules = [
        'ml/dashboard.py',
        'ml/dashboard_utils.py',
        'ml/training_manager.py',
    ]
    
    failed = []
    for module in modules:
        file_path = base_path / module
        
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            compile(code, str(file_path), 'exec')
            print(f"  ✅ {module} (syntax OK)")
        except SyntaxError as e:
            print(f"  ❌ {module} (syntax error: {e})")
            failed.append(module)
    
    return len(failed) == 0, failed


def test_training_manager():
    """Test TrainingManager instantiation"""
    print("\n🧪 Testing TrainingManager...")
    
    try:
        from ml.training_manager import TrainingManager, TrainingStatus, TrainingProgress, TrainingMetrics
        
        # Create instance
        manager = TrainingManager()
        print(f"  ✅ TrainingManager created")
        
        # Check initial state
        assert manager.progress.status == TrainingStatus.IDLE, "Initial status should be IDLE"
        print(f"  ✅ Initial status: {manager.progress.status.value}")
        
        # Check callbacks
        assert len(manager._progress_callbacks) == 0, "No callbacks initially"
        
        def dummy_callback(progress):
            pass
        
        manager.add_progress_callback(dummy_callback)
        assert len(manager._progress_callbacks) == 1, "Should have 1 callback"
        print(f"  ✅ Callbacks working")
        
        return True, []
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False, [str(e)]


def test_dashboard_utils():
    """Test dashboard utilities"""
    print("\n🧪 Testing Dashboard Utils...")
    
    try:
        from ml.dashboard_utils import (
            DataValidator, ResultsFormatter, FileManager,
            ChartBuilder, ProgressTracker, NotificationManager
        )
        
        print(f"  ✅ DataValidator")
        print(f"  ✅ ResultsFormatter")
        print(f"  ✅ FileManager")
        print(f"  ✅ ChartBuilder")
        print(f"  ✅ ProgressTracker")
        print(f"  ✅ NotificationManager")
        
        # Test ProgressTracker
        duration = ProgressTracker.format_duration(3661)
        assert "h" in duration and "m" in duration, f"Duration format wrong: {duration}"
        print(f"  ✅ Duration formatting: {duration}")
        
        return True, []
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False, [str(e)]


def main():
    """Run all tests"""
    print("=" * 50)
    print("🧪 Dashboard Component Tests")
    print("=" * 50)
    
    results = {}
    
    # Test imports
    success, failed = test_imports()
    results['imports'] = (success, failed)
    if not success and failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print(f"   Install with: pip install -r requirements-dashboard.txt")
        return False
    
    # Test files
    success, failed = test_files()
    results['files'] = (success, failed)
    if not success:
        print(f"\n❌ Missing files: {', '.join(failed)}")
        return False
    
    # Test Python modules
    success, failed = test_modules()
    results['modules'] = (success, failed)
    if not success:
        return False
    
    # Test TrainingManager (only if ML modules exist)
    try:
        success, failed = test_training_manager()
        results['training_manager'] = (success, failed)
        if not success:
            print(f"\n⚠️  TrainingManager test failed (ML modules may not be set up)")
    except ImportError:
        print("\n⚠️  Skipping TrainingManager test (ML modules not available)")
        results['training_manager'] = (None, ['ML modules not available'])
    
    # Test dashboard utils
    try:
        success, failed = test_dashboard_utils()
        results['dashboard_utils'] = (success, failed)
    except ImportError:
        print("\n⚠️  Skipping dashboard utils test (dependencies missing)")
        results['dashboard_utils'] = (None, ['Dependencies not installed'])
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for s, _ in results.values() if s is True)
    failed = sum(1 for s, _ in results.values() if s is False)
    skipped = sum(1 for s, _ in results.values() if s is None)
    
    print(f"\n✅ Passed:  {passed}/{len(results)}")
    print(f"❌ Failed:  {failed}/{len(results)}")
    print(f"⏭️  Skipped: {skipped}/{len(results)}")
    
    if failed == 0:
        print("\n" + "=" * 50)
        print("✨ All tests passed! Dashboard is ready to use.")
        print("=" * 50)
        print("\n🚀 To launch the dashboard, run:")
        print("   python launch_dashboard.py")
        return True
    else:
        print("\n" + "=" * 50)
        print("❌ Some tests failed. Please fix issues above.")
        print("=" * 50)
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
