"""Documented conventions used across falsify.

These constants exist so the rest of the codebase has a single place to point
at when a reviewer asks "what convention is this in?". None of them are
behavioral defaults — they are documentation.
"""

# Quaternion order used everywhere in falsify and FiGS: x, y, z, w (scipy convention).
QUATERNION_ORDER = "xyzw"

# The FiGS dynamics state vector layout.
DRONE_STATE_LAYOUT = ("px", "py", "pz", "vx", "vy", "vz", "qx", "qy", "qz", "qw")

# Frame "convention" tags (informational, used by Frame.convention).
RIGHT_HANDED = "right_handed"
LEFT_HANDED = "left_handed"

# Common axis conventions, captured as documentation strings.
NED_AXES = "x = North, y = East, z = Down"
ZUP_AXES = "x = forward (scene-relative), y = left, z = up"
OPENGL_CAMERA_AXES = "x = right, y = up, z = backward (out of the image)"
OPENCV_CAMERA_AXES = "x = right, y = down, z = forward (into the image)"
