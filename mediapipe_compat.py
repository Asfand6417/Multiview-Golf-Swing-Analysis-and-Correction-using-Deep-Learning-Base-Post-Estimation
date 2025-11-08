"""
Compatibility layer for mediapipe to work with Python 3.9 and TensorFlow
This module addresses the 'unhashable type: list' error that occurs when 
importing mediapipe with certain TensorFlow versions on Python 3.9
"""
import os
import sys
import importlib
import types

# Set environment variable to disable oneDNN custom operations
# This can help with compatibility issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def patch_typing():
    """
    Patch the typing module to handle the unhashable list issue
    """
    import typing
    
    # Save the original _remove_dups_flatten function
    original_remove_dups = typing._remove_dups_flatten
    
    # Create a patched version that handles lists safely
    def patched_remove_dups_flatten(parameters):
        try:
            return original_remove_dups(parameters)
        except TypeError as e:
            if "unhashable type: 'list'" in str(e):
                # Convert lists to tuples which are hashable
                params = []
                for p in parameters:
                    if isinstance(p, list):
                        params.append(tuple(p))
                    else:
                        params.append(p)
                return original_remove_dups(params)
            raise
    
    # Apply the patch
    typing._remove_dups_flatten = patched_remove_dups_flatten

# Apply patches before importing mediapipe
patch_typing()

# Import mediapipe safely
import mediapipe as mp

# Export mediapipe namespace
sys.modules[__name__] = mp


def solutions():
    return None