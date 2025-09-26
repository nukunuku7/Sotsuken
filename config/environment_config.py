# environment_config.py (360° Projection Environment)

# Units in meters
# This configuration models a room with 3 screens and 3 projectors (360-degree setup)

environment = {
    "screens": [
        {
            "id": "screen1",
            "center": [0.0, 1.5, 0.0],        # Front screen center
            "normal": [1.0, 0.0, 0.0],        # Facing +X direction
            "radius": 1.5                     # Curved screen radius
        },
        {
            "id": "screen2",
            "center": [0.0, -1.5, 0.0],       # Rear screen center
            "normal": [-1.0, 0.0, 0.0],       # Facing -X direction
            "radius": 1.5
        },
        {
            "id": "screen3",
            "center": [0.0, 0.0, 1.5],        # Top screen center
            "normal": [0.0, 1.0, 0.0],        # Facing +Y (vertical up)
            "radius": 1.5
        }
    ],

    "projectors": [
        {
            "id": "proj1",
            "position": [-3.0, 1.5, 1.2],     # Projector for screen1
            "fov": 100.0                      # Horizontal field of view
        },
        {
            "id": "proj2",
            "position": [-3.0, -1.5, 1.2],    # Projector for screen2
            "fov": 100.0
        },
        {
            "id": "proj3",
            "position": [-3.0, 0.0, 2.0],     # Projector for screen3
            "fov": 100.0
        }
    ],

    "mirror": {
        "enabled": True,
        "center": [-1.5, 0.0, 1.5],           # Spherical mirror center
        "radius": 0.6                        # Radius of curvature (convex)
    }
}