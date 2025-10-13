#!/usr/bin/env python3
"""
🚀 SYMBIOAI QUICK TEST LAUNCHER
Easy access to the comprehensive test suite
"""

import sys
import os
from pathlib import Path

# Add test suite to path
test_suite_dir = Path(__file__).parent / 'symbioai_test_suite'
sys.path.insert(0, str(test_suite_dir))

# Import and run the master test runner
from symbioai_test_suite.run_all_tests import main

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║        🚀 SYMBIOAI COMPREHENSIVE TEST SUITE LAUNCHER 🚀            ║
║                                                                    ║
║  Professional testing infrastructure for advanced AI validation    ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

Quick Commands:
  python quick_test.py              → Run all tests
  python quick_test.py --tier tier1 → Quick sanity (< 2 min)
  python quick_test.py --tier tier2 → Module tests (2-10 min)

""")
    
    # Change to test suite directory
    os.chdir(test_suite_dir)
    
    # Run the main test suite
    main()
