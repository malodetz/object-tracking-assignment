import os
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Define functions outside the loop
def get_point_on_random_side(width, height):
    side = random.randint(0, 4)
    if side == 0:
        x = random.randint(0, width)
        y = 0
    elif side == 1:
        x = random.randint(0, width)
        y = height
    elif side == 2:
        x = 0
        y = random.randint(0, height)
    else:
        x = width
        y = random.randint(0, height)
    return x, y


def fun(x, a, b, c, d):
    return a * x + b * x**2 + c * x**3 + d


def objective(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f


def check_track(track, width, height):
    if all(el["x"] == track[0]["x"] for el in track):
        return False
    if all(el["y"] == track[0]["y"] for el in track):
        return False
    if not all(el["x"] >= 0 and el["x"] <= width for el in track):
        return False
    if not all(el["y"] >= 0 and el["y"] <= height for el in track):
        return False
    if (2 > track[0]["x"] > (width - 2) and 2 > track[0]["y"] > (width - 2)) or (
        2 > track[-1]["x"] > (width - 2) and 2 > track[-1]["y"] > (width - 2)
    ):
        return False
    return True


def add_track_to_tracks(
    track, tracks, id, bb_skip_percent, random_range, cb_width, cb_height
):
    for i, p in enumerate(track):
        # a chance that detector missed object
        if random.random() < bb_skip_percent:
            bounding_box = []
        else:
            bounding_box = [
                p["x"]
                - int(cb_width / 2)
                + random.randint(-random_range, random_range),
                p["y"] - cb_height + random.randint(-random_range, random_range),
                p["x"]
                + int(cb_width / 2)
                + random.randint(-random_range, random_range),
                p["y"] + random.randint(-random_range, random_range),
            ]

        if i < len(tracks):
            tracks[i]["data"].append(
                {
                    "cb_id": id,
                    "bounding_box": bounding_box,
                    "x": p["x"],
                    "y": p["y"],
                    "track_id": None,
                }
            )
        else:
            tracks.append(
                {
                    "frame_id": len(tracks) + 1,
                    "data": [
                        {
                            "cb_id": id,
                            "bounding_box": bounding_box,
                            "x": p["x"],
                            "y": p["y"],
                            "track_id": None,
                        }
                    ],
                }
            )
    return tracks


def generate_track_file(tracks_amount, random_range, bb_skip_percent, output_dir="."):
    # Set up environment
    width = 1000
    height = 800
    tracks = []
    i = 0
    cb_width = 120
    cb_height = 100

    # Generate tracks
    while i < tracks_amount:
        x, y = np.array([]), np.array([])
        p = get_point_on_random_side(width, height)
        x = np.append(x, p[0])
        y = np.append(y, p[1])
        x = np.append(x, random.randint(200, width - 200))
        y = np.append(y, random.randint(200, height - 200))
        x = np.append(x, random.randint(200, width - 200))
        y = np.append(y, random.randint(200, height - 200))
        p = get_point_on_random_side(width, height)
        x = np.append(x, p[0])
        y = np.append(y, p[1])
        num = random.randint(20, 50)
        coef, _ = curve_fit(fun, x, y)
        track = [
            {"x": int(x), "y": int(y)}
            for x, y in zip(
                np.linspace(x[0], x[-1], num=num),
                fun(np.linspace(x[0], x[-1], num=num), *coef),
            )
        ]
        if check_track(track, width, height):
            tracks = add_track_to_tracks(
                track, tracks, i, bb_skip_percent, random_range, cb_width, cb_height
            )
            i += 1

    # Create a filename based on parameters
    bb_skip_str = str(int(bb_skip_percent * 100))
    filename = f"track_{tracks_amount}_{random_range}_{bb_skip_str}.py"
    filepath = os.path.join(output_dir, filename)

    # Write the track data to a file
    with open(filepath, "w") as f:
        f.write(f"country_balls_amount = {tracks_amount}\n")
        f.write(f"track_data = {tracks}\n")

    return filename


# Main script
if __name__ == "__main__":
    # Define parameter combinations
    tracks_amounts = [5, 10, 15]
    random_ranges = [2, 5, 10]
    bb_skip_percents = [0.1, 0.25, 0.75]

    # Generate all combinations
    for tracks_amount in tracks_amounts:
        for random_range in random_ranges:
            for bb_skip_percent in bb_skip_percents:
                filename = generate_track_file(
                    tracks_amount, random_range, bb_skip_percent
                )
                print(f"from {filename[:-3]} import country_balls_amount, track_data")

    print("All track files generated successfully!")
