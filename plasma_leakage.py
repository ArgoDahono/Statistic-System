# plasma_leakage.py - Python equivalent of C++ module
"""
Plasma Leakage Module - Python Implementation
Equivalent to C++ plasma_leakage module for fast computations
"""

# Import the fast calculators
from Trial import LeakageIndexCalculator, StatisticalCalculator

# Expose classes for compatibility
PlasmaLeakageProcessor = LeakageIndexCalculator
PlasmaLeakageMonteCarlo = LeakageIndexCalculator

# Additional compatibility exports
__version__ = "1.0.0"
__all__ = ['LeakageIndexCalculator', 'StatisticalCalculator', 'PlasmaLeakageProcessor', 'PlasmaLeakageMonteCarlo']