# Mediapipe Compatibility Fix for Python 3.9

## Issue Description

When running the GolfPose application with Python 3.9, you may encounter the following error:

```
TypeError: unhashable type: 'list'
```

This error occurs in the Python typing module when importing mediapipe, which in turn imports TensorFlow. The specific issue happens when the typing module tries to handle Union and Optional types with lists as parameters.

## Root Cause

The error is caused by a compatibility issue between:
- Python 3.9's typing module
- TensorFlow's type annotations
- Mediapipe's dependency on TensorFlow

In some versions of TensorFlow, there are type annotations that use lists as parameters to Union or Optional types. Since lists are mutable (unhashable), they cannot be used in sets, which is what the typing module tries to do when processing these types.

## Solution

A compatibility layer (`mediapipe_compat.py`) has been created to address this issue. This layer:

1. Sets the `TF_ENABLE_ONEDNN_OPTS` environment variable to `0` to disable oneDNN custom operations
2. Patches the typing module to safely handle lists in type annotations by converting them to tuples
3. Imports mediapipe after applying the patch
4. Exports the mediapipe namespace so it can be used as a drop-in replacement

## How to Use

Instead of directly importing mediapipe:

```python
import mediapipe as mp  # May cause errors in Python 3.9
```

Use the compatibility layer:

```python
import mediapipe_compat as mp  # Safe for Python 3.9
```

## Applying to Other Scripts

If you encounter the same error in other scripts, simply replace the mediapipe import with the compatibility layer import.

## Alternative Solutions

If this compatibility layer doesn't resolve your issue, you might consider:

1. Downgrading to Python 3.8
2. Upgrading or downgrading TensorFlow and mediapipe to versions known to work together
3. Using a virtual environment with specific package versions

## Technical Details

The patch works by intercepting the `_remove_dups_flatten` function in the typing module, which is where the error occurs. When it encounters a list (which is unhashable), it converts it to a tuple (which is hashable) before proceeding with the original function.