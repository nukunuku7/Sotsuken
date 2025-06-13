# create_obj.py
import numpy as np
from scipy.optimize import minimize

# スクリーンのパラメータ
radius = 2204
center_height = 1610

def optimize_positions():
    projector_init = np.array([0.0, 100.0, 50.0])
    mirror_init = np.array([0.0, 50.0, 50.0])
    relative_vector = mirror_init - projector_init

    def cost_function(projector_pos):
        mirror_pos = projector_pos + relative_vector
        screen_center = np.array([0, 0, center_height])
        mirror_to_screen = screen_center - mirror_pos
        mirror_to_screen /= np.linalg.norm(mirror_to_screen)
        projector_to_mirror = mirror_pos - projector_pos
        projector_to_mirror /= np.linalg.norm(projector_to_mirror)
        alignment = -np.dot(projector_to_mirror, mirror_to_screen)
        return 1 - alignment

    def constraint_x_fixed(projector_pos): return projector_pos[0]
    def constraint_mirror_y_upper(projector_pos): return 1500 - (projector_pos[1] + relative_vector[1])
    def constraint_mirror_y_lower(projector_pos): return (projector_pos[1] + relative_vector[1]) - 0
    def constraint_projector_y(projector_pos): return projector_pos[1] - 1500

    constraints = [
        {'type': 'eq', 'fun': constraint_x_fixed},
        {'type': 'ineq', 'fun': constraint_mirror_y_upper},
        {'type': 'ineq', 'fun': constraint_mirror_y_lower},
        {'type': 'ineq', 'fun': constraint_projector_y}
    ]
    bounds = [(0, 0), (1500, None), (None, None)]

    res = minimize(cost_function, projector_init, method='SLSQP', bounds=bounds, constraints=constraints)
    projector = res.x
    mirror = projector + relative_vector
    return projector, mirror

def to_local_coordinates(pos):
    return pos - np.array([0, 0, 0])

def create_objects():
    projector, mirror = optimize_positions()
    projector_local = to_local_coordinates(projector)
    mirror_local = to_local_coordinates(mirror)

    objects = {
        "projector": {
            "position": projector_local.tolist(),
            "type": "projector"
        },
        "mirror": {
            "position": mirror_local.tolist(),
            "type": "mirror"
        }
    }
    
    return objects

if __name__ == "__main__":
    objects = create_objects()
    print("Projector Position:", objects["projector"]["position"])
    print("Mirror Position:", objects["mirror"]["position"])
    print("Objects Created Successfully")