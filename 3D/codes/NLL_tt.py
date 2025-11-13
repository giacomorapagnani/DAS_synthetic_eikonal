import os
import subprocess
import numpy as np

def write_grid2time_input(filename, vel_model, phase, origin, n, spacing, event_xyz, out_dir):
    x0, y0, z0 = origin
    xs, ys, zs = event_xyz
    os.makedirs(out_dir, exist_ok=True)

    with open(filename, 'w') as f:
        f.write("CONTROL 1 12345\n")
        f.write("TRANS NONE\n")
        f.write(f"GTFILES {vel_model} {os.path.join(out_dir, phase)} {phase} 0\n")
        f.write("GTMODE GRID3D\n")
        f.write("MESSAGE 2\n")
        f.write(f"GTSRCE SRCE_{phase} XYZ {xs} {ys} {zs}\n")
        f.write(f"GT_GRID {x0} {y0} {z0} {n} {n} {n} {spacing} {spacing} {spacing}\n")
        f.write("GT_METHOD GT_PLFD\n")
        f.write(f"GT_PHASE {phase}\n")
        f.write("END\n")

def run_grid2time(input_file):
    result = subprocess.run(["Grid2Time", input_file], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Grid2Time failed with exit code {result.returncode}")

def read_time_file(filename, n):
    with open(filename, 'rb') as f:
        f.read(256)  # skip header
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape((n, n, n))

def compute_travel_times(event_xyz, vel_model, n=50, spacing=1.0, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    origin = (0.0, 0.0, 0.0)

    # Onde P
    input_P = os.path.join(out_dir, "P.in")
    write_grid2time_input(input_P, vel_model, "P", origin, n, spacing, event_xyz, out_dir)
    run_grid2time(input_P)
    ttp = read_time_file(os.path.join(out_dir, "P.time"), n)

    # Onde S
    input_S = os.path.join(out_dir, "S.in")
    write_grid2time_input(input_S, vel_model, "S", origin, n, spacing, event_xyz, out_dir)
    run_grid2time(input_S)
    tts = read_time_file(os.path.join(out_dir, "S.time"), n)

    return ttp, tts

if __name__ == "__main__":
    event_xyz = (0.0, 0.0, 5.0)  # evento a 5 km di profondità
    vel_mod_dir= "../codes/"
    vel_model = "model3D.mod"
    n = 50
    spacing = 1.0
    ttp, tts = compute_travel_times(event_xyz, vel_model, n, spacing)

    print("Travel time P:", ttp.shape, np.nanmin(ttp), np.nanmax(ttp))
    print("Travel time S:", tts.shape, np.nanmin(tts), np.nanmax(tts))
