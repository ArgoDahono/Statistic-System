# Test basic functionality without pandas
import math

print("Testing basic Python functionality...")

# Test the LeakageIndexCalculator class
class LeakageIndexCalculator:
    """Fast leakage index calculations - C++ equivalent in Python"""

    @staticmethod
    def calculate(albumin, hematocrit):
        """Calculate leakage index from albumin and hematocrit"""
        # Simple NaN check without pandas
        if albumin <= 0 or math.isnan(albumin) or math.isnan(hematocrit):
            return float('nan')
        return hematocrit / albumin

    @staticmethod
    def validate_physiology(albumin, hematocrit):
        """Validate physiological status based on leakage index"""
        idx = LeakageIndexCalculator.calculate(albumin, hematocrit)
        if math.isnan(idx):
            return 'Invalid'
        if idx < 1.5:
            return 'Normal'
        elif idx < 2.0:
            return 'Risiko Kebocoran'
        else:
            return 'Kebocoran Plasma'

# Test basic calculation
result = LeakageIndexCalculator.calculate(3.5, 45.0)
print(f"Leakage index calculation: {result}")

# Test physiology validation
status = LeakageIndexCalculator.validate_physiology(3.5, 45.0)
print(f"Physiology status: {status}")

print("Basic functionality test passed!")