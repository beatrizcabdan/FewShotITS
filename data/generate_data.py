import numpy as np
import os
import pandas as pd
import os
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
import random
import math
import zCurve as z


def compute_relative_acceleration(pos_prev, pos_current, pos_next, dt):
    # Compute velocity vectors
    vx1 = (pos_current[0] - pos_prev[0]) / dt
    vy1 = (pos_current[1] - pos_prev[1]) / dt
    vx2 = (pos_next[0] - pos_current[0]) / dt
    vy2 = (pos_next[1] - pos_current[1]) / dt

    # Compute the change in velocity (acceleration vector)
    ax = (vx2 - vx1) / dt
    ay = (vy2 - vy1) / dt

    # Velocity direction (unit vector) at the middle position
    v_mag = math.hypot(vx1, vy1)
    if v_mag == 0:
        return 0.0, 0.0  # No motion

    # Tangential direction (unit vector)
    tx = vx1 / v_mag
    ty = vy1 / v_mag

    # Normal direction (perpendicular to tangential)
    nx = -ty
    ny = tx

    # Project acceleration onto tangential and normal directions
    a_tangential = ax * tx + ay * ty
    a_normal = ax * nx + ay * ny

    return a_normal, a_tangential


def generate_turning_data_3(length=30, left=True, initial_angle=math.pi/2, maxangle=math.pi/2, initial_x=0, initial_y=0, initial_speed=5, changespeed=0, noise_std=0.01):
    padding = 3
    length += padding
    noise_x = np.random.normal(0, noise_std, size=length)
    noise_y = np.random.normal(0, noise_std, size=length)

    positions_x = np.ones(length)*initial_x
    positions_y = np.ones(length)*initial_y
    turning_angle = np.ones(length) * initial_angle
    speed = np.ones(length) * initial_speed
    acceleration_x = np.zeros(length)
    acceleration_y = np.zeros(length)

    delta_t = 0.01  # Time step in microseconds


    for i in range(2, length):
        if padding < i < length:
            turning_angle[i] -= (i - padding) / (length - padding) * maxangle * (-1 if left else 1)
        elif i >= length - padding:
            turning_angle[i] = turning_angle[length-padding-1]
        speed[i] += changespeed * i / length

        positions_x[i] = positions_x[i-1] + math.cos(turning_angle[i]) * speed[i] * delta_t
        positions_y[i] = positions_y[i-1] + math.sin(turning_angle[i]) * speed[i] * delta_t

        acceleration_x[i], acceleration_y[i] = compute_relative_acceleration((positions_x[i-2], positions_y[i-2]), (positions_x[i-1], positions_y[i-1]), (positions_x[i], positions_y[i]), delta_t*10)

    positions_x += noise_x
    positions_y += noise_y
    speed += noise_x / 3 + noise_y / 3
    acceleration_x += noise_x / 5
    acceleration_y += noise_y / 5
    return positions_x[padding:], positions_y[padding:], turning_angle[padding:], speed[padding:], acceleration_x[padding:], acceleration_y[padding:]


def generate_sequence(class_type, noise_std, length=30):
    base = np.ones(length) * 50
    if class_type == 'normal':
        sequence = base
    elif class_type == 'lefturn':
        base[length // 2] += 15 + np.random.normal(0, 3)
        sequence = base
    elif class_type == 'rightturn':
        base[length // 2] -= 15 + np.random.normal(0, 3)
        sequence = base
    elif class_type == 'noisy':
        pattern = np.tile([5, -5], length // 2)
        sequence = base + pattern
    else:
        sequence = base

    # Add noise
    noise = np.random.normal(0, noise_std, size=length)
    return sequence + noise

def concatenate_trajectory_variables(positions_x, positions_x2, positions_y, positions_y2, turning_angle, turning_angle2, speed, speed2, acceleration_x, acceleration_x2, acceleration_y, acceleration_y2):
    positions_x = np.concatenate((positions_x, positions_x2))
    positions_y = np.concatenate((positions_y, positions_y2))
    turning_angle = np.concatenate((turning_angle, turning_angle2))
    speed = np.concatenate((speed, speed2))
    acceleration_x = np.concatenate((acceleration_x, acceleration_x2))
    acceleration_y = np.concatenate((acceleration_y, acceleration_y2))

    return positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y


def save_sequences(output_dir="data/data_csv", num_samples=1000, noise_std=2.0):
    os.makedirs(output_dir, exist_ok=True)
    classes = ['normal', 'lefturn', 'rightturn', 'noisy', 'ra', 'lanechange']
    for cls in classes:
        for i in range(num_samples):
            if cls == "normal":
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = generate_turning_data_3(left=False, initial_angle=0, maxangle=0, initial_x=0, initial_y=0, initial_speed=30, changespeed=3, noise_std=0)
            elif cls == "lefturn":
                maxangle = math.pi / 2 - math.pi / 4 * (i/num_samples)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = generate_turning_data_3(left=True, initial_angle=0, maxangle=maxangle, initial_x=0, initial_y=0, initial_speed=30, changespeed=3, noise_std=noise_std)
            elif cls == "rightturn":
                maxangle = math.pi / 2 - math.pi / 4 * (i/num_samples)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = generate_turning_data_3(left=False, initial_angle=0, maxangle=maxangle, initial_x=0, initial_y=0, initial_speed=30, changespeed=3, noise_std=noise_std)
            elif cls == "noisy":
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = generate_turning_data_3(left=False, initial_angle=0, maxangle=0, initial_x=0, initial_y=0, initial_speed=30, changespeed=3, noise_std=noise_std*5+0.01)
            elif cls == "lanechange":
                maxangle = math.pi / 4 - math.pi / 6 * (i/num_samples)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = generate_turning_data_3(left=False, initial_angle=0, maxangle=0, initial_x=0, initial_y=0, initial_speed=30, changespeed=0, noise_std=noise_std, length=5)
                positions_x2, positions_y2, turning_angle2, speed2, acceleration_x2, acceleration_y2 = generate_turning_data_3(left=True, initial_angle=turning_angle[-1], maxangle=maxangle, initial_x=positions_x[-1], initial_y=positions_y[-1], initial_speed=speed[-1], changespeed=1, noise_std=noise_std, length=10)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = concatenate_trajectory_variables(positions_x, positions_x2, positions_y, positions_y2, turning_angle, turning_angle2, speed, speed2, acceleration_x, acceleration_x2, acceleration_y, acceleration_y2)

                positions_x2, positions_y2, turning_angle2, speed2, acceleration_x2, acceleration_y2 = generate_turning_data_3(left=False, initial_angle=turning_angle[-1], maxangle=0, initial_x=positions_x[-1], initial_y=positions_y[-1], initial_speed=speed[-1], changespeed=1, noise_std=0, length=5)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = concatenate_trajectory_variables(positions_x, positions_x2, positions_y, positions_y2, turning_angle, turning_angle2, speed, speed2, acceleration_x, acceleration_x2, acceleration_y, acceleration_y2)
                positions_x2, positions_y2, turning_angle2, speed2, acceleration_x2, acceleration_y2 = generate_turning_data_3(left=False, initial_angle=turning_angle[-1], maxangle=1.1*maxangle, initial_x=positions_x[-1], initial_y=positions_y[-1], initial_speed=speed[-1], changespeed=1, noise_std=noise_std, length=5)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = concatenate_trajectory_variables(positions_x, positions_x2, positions_y, positions_y2, turning_angle, turning_angle2, speed, speed2, acceleration_x, acceleration_x2, acceleration_y, acceleration_y2)
                positions_x2, positions_y2, turning_angle2, speed2, acceleration_x2, acceleration_y2 = generate_turning_data_3(left=False, initial_angle=turning_angle[-1], maxangle=0, initial_x=positions_x[-1], initial_y=positions_y[-1], initial_speed=speed[-1], changespeed=0, noise_std=0, length=5)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = concatenate_trajectory_variables(positions_x, positions_x2, positions_y, positions_y2, turning_angle, turning_angle2, speed, speed2, acceleration_x, acceleration_x2, acceleration_y, acceleration_y2)
            elif cls == "ra":
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = generate_turning_data_3(left=False, initial_angle=0, maxangle=0, initial_x=0, initial_y=0, initial_speed=31, changespeed=-1, noise_std=0, length=3)
                positions_x2, positions_y2, turning_angle2, speed2, acceleration_x2, acceleration_y2 = generate_turning_data_3(left=False, initial_angle=turning_angle[-1], maxangle=math.pi / 2, initial_x=positions_x[-1], initial_y=positions_y[-1], initial_speed=speed[-1], changespeed=2, noise_std=0, length=4)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = concatenate_trajectory_variables(positions_x, positions_x2, positions_y, positions_y2, turning_angle, turning_angle2, speed, speed2, acceleration_x, acceleration_x2, acceleration_y, acceleration_y2)

                maxangle = 6 * math.pi / 5 - 2 * math.pi / 5 * (i / num_samples)
                positions_x2, positions_y2, turning_angle2, speed2, acceleration_x2, acceleration_y2 = generate_turning_data_3(left=True, initial_angle=turning_angle[-1], maxangle=maxangle, initial_x=positions_x[-1], initial_y=positions_y[-1], initial_speed=speed[-1], changespeed=1, noise_std=noise_std, length=16)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = concatenate_trajectory_variables(positions_x, positions_x2, positions_y, positions_y2, turning_angle, turning_angle2, speed, speed2, acceleration_x, acceleration_x2, acceleration_y, acceleration_y2)

                positions_x2, positions_y2, turning_angle2, speed2, acceleration_x2, acceleration_y2 = generate_turning_data_3(left=False, initial_angle=turning_angle[-1], maxangle=math.pi / 3, initial_x=positions_x[-1], initial_y=positions_y[-1], initial_speed=speed[-1], changespeed=0, noise_std=0, length=4)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = concatenate_trajectory_variables(positions_x, positions_x2, positions_y, positions_y2, turning_angle, turning_angle2, speed, speed2, acceleration_x, acceleration_x2, acceleration_y, acceleration_y2)
                positions_x2, positions_y2, turning_angle2, speed2, acceleration_x2, acceleration_y2 = generate_turning_data_3(left=False, initial_angle=turning_angle[-1], maxangle=0, initial_x=positions_x[-1], initial_y=positions_y[-1], initial_speed=speed[-1], changespeed=0, noise_std=0, length=3)
                positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y = concatenate_trajectory_variables(positions_x, positions_x2, positions_y, positions_y2, turning_angle, turning_angle2, speed, speed2, acceleration_x, acceleration_x2, acceleration_y, acceleration_y2)


            # seq = [z.interlace(int(x+100), int(y+100), bits_per_dim=14) for x, y in zip(acceleration_x, acceleration_y)] #SFC-encoded sensor signals
            seq = [z.interlace(int(a + 10), int(b + 10), int(c * 4 + 12), int(d/5), int(e + 15), int(f + 15), bits_per_dim=8) for a, b, c, d, e, f in zip(positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y)]
            seq = [[a, b, c, d, e, f, g] for a, b, c, d, e, f, g in zip(positions_x, positions_y, turning_angle, speed, acceleration_x, acceleration_y, seq)]  # original sensor signals
            df = pd.DataFrame(seq, columns=["positions_x", "positions_y", "turning_angle", "speed", "acceleration_x", "acceleration_y", "sfc_encoded"])
            df.to_csv(os.path.join(output_dir, f"{cls}_{i}.csv"), index=False, float_format='%.4f')


if __name__ == "__main__":
    noise_std = 0.01
    save_sequences(noise_std=noise_std, output_dir="data/data_csv")  # synthetic training set generator
    save_sequences(noise_std=noise_std, output_dir="data/test_csv")  # synthetic training set generator
