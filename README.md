# An SMPL Implementation

This is an implementation of the SMPL body model for NumPy, PyTorch and TensorFlow.
It is similar to https://github.com/CalciferZh/SMPL, with two main differences:

- we use batched arrays in all frameworks,
- we provide a fast way to compute only the joint locations but not the vertex locations

There are also minor rearrangements of the computations, with heavy use of Einstein summation for clarity and speed.

Usage:

```python
from smpl.smpl_numpy import SMPL

body_model = SMPL(model_dir='...', gender='neutral'):

# The pose is a concatenation of rotation vectors for the 24 body parts
# (a rotation vector is one whose direction is the rotation axis
# and length the rotation angle in radians).
pose = np.random.rand((1, 72))

# The shape is expressed in SMPL's 10-dimensional shape space
pose = np.random.rand((1, 10))

result = body_model(pose, shape)
result['vertices']  # vertex locations, shape [1, 6890, 3]
result['joints']  # joint locations, shape [1, 24, 3]
result['orientations']  # global orientation of body parts as rotation matrices, shape [1, 24, 3, 3]

# For extra speed, we can skip computing the vertices if only joints are needed
result = body_model(pose, shape, return_vertices=False)
result['joints']  # joint locations, shape [1, 24, 3]
result['orientations']  # global orientation of body parts as rotation matrices, shape [1, 24, 3, 3]
```

To use TensorFlow or PyTorch, use `from smpl.smpl_tensorflow import SMPL` or `from smpl.smpl_pytorch import SMPL`.

## Model files

This code requires the SMPL model files which can be obtained from https://smpl.is.tue.mpg.de after agreeing to the SMPL license.
