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
    noise_x = np.random.normal(0, noise_std, size=length)
    noise_y = np.random.normal(0, noise_std, size=length)

    positions_x = np.zeros(length)
    positions_y = np.zeros(length)
    turning_angle = np.ones(length) * initial_angle
    speed = np.ones(length) * initial_speed
    acceleration_x = np.zeros(length)
    acceleration_y = np.zeros(length)

    # Initialize variables
    positions_x[0], positions_y[0] = initial_x, initial_y  # Start position

    delta_t = 0.01  # Time step in microseconds
    padding = 4

    for i in range(2, length):
        if padding < i < length - padding:
            turning_angle[i] -= (i - padding) / (length - 2 * padding) * maxangle * (-1 if left else 1)
        elif i >= length - padding:
            turning_angle[i] = turning_angle[length-padding-1]
        speed[i] += (2 * i / length) * changespeed

        positions_x[i] = positions_x[i-1] + math.cos(turning_angle[i]) * speed[i] * delta_t + noise_x[i]
        positions_y[i] = positions_y[i-1] + math.sin(turning_angle[i]) * speed[i] * delta_t + noise_y[i]

        acceleration_x[i], acceleration_y[i] = compute_relative_acceleration((positions_x[i-2], positions_y[i-2]), (positions_x[i-1], positions_y[i-1]), (positions_x[i], positions_y[i]), delta_t*10)

    return positions_x, positions_y, turning_angle, speed, turning_angle, acceleration_x, acceleration_y


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


def save_sequences(output_dir="data/data_csv", num_samples=20, noise_std=2.0):
    left = False  # default
    maxangle = 0  # default

    os.makedirs(output_dir, exist_ok=True)
    classes = ['normal', 'lefturn', 'rightturn', 'noisy']
    for cls in classes:
        if cls == "normal":
            maxangle = 0
            noise_use = 0
        elif cls == "lefturn":
            maxangle = math.pi / 2
            left = True
            noise_use = noise_std
        elif cls == "rightturn":
            maxangle = math.pi / 2
            left = False
            noise_use = noise_std
        elif cls == "noisy":
            maxangle = 0
            noise_use = noise_std * 10
        for i in range(num_samples):
            positions_x, positions_y, turning_angle, speed, turning_angle, acceleration_x, acceleration_y = generate_turning_data_3(left=left, initial_angle=0, maxangle=maxangle, initial_x=0, initial_y=0, initial_speed=7, changespeed=0, noise_std=noise_use)
            # seq = generate_sequence(cls, noise_std)
            # seq = np.random.normal(0, noise_std, size=30) #randomsignal
            # seq = acceleration_x #sensor signal
            seq = [z.interlace(int(x*10+100), int(y*10+100), bits_per_dim=14) for x, y in zip(acceleration_x, acceleration_y)] #SFC-encoded sensor signals

            df = pd.DataFrame(seq, columns=["value"])
            df.to_csv(os.path.join(output_dir, f"{cls}_{i}.csv"), index=False)


if __name__ == "__main__":
    save_sequences(noise_std=0.01)  # synthetic sequence generator