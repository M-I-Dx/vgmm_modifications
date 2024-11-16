import sys
import os

# Add the src directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import traceback

# Import test functions
from unit_test import (
    test_data_loader_transformer_initialization_and_partitions,
    test_scalable_vgmm_normalizer_initialization
)
from sanity_check import test_combined_transform_inverse_transform

def run_tests():
    """
    Run all tests and report success or failure.
    """
    tests = [
        ("Test ScalableVGMMNormalizer Initialization", test_scalable_vgmm_normalizer_initialization),
        ("Test DataLoaderTransformer Initialization and Partitions", test_data_loader_transformer_initialization_and_partitions),
        ("Test Combined Transform and Inverse Transform", test_combined_transform_inverse_transform),
    ]

    print("\nRunning Tests...\n")
    all_successful = True

    for test_name, test_func in tests:
        try:
            print(f"Running: {test_name}...")
            test_func()
            print(f"SUCCESS: {test_name}\n")
        except Exception as e:
            all_successful = False
            print(f"FAILURE: {test_name}")
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            print()

    if all_successful:
        print("All tests passed successfully!")
    else:
        print("Some tests failed. Please check the details above.")

if __name__ == "__main__":
    run_tests()