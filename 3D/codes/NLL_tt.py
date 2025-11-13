import os
import subprocess
import numpy as np
import struct

def write_nlloc_control_file(filename, grid_dir, vel_model, phase, x0, y0, z0, n, spacing):
    """
    Crea un file di configurazione NonLinLoc per il calcolo delle travel time.
    """
    with open(filename, 'w') as f:
        f.write(f"CONTROL 1 12345\n")
        f.write(f"TRANS SDCART 0.0 0.0 0.0\n")  # sistema di coordinate (da adattare)
        f.write(f"VGGRID {grid_dir}/{phase}  {x0} {y0} {z0} {n} {n} {n} {spacing} {spacing} {spacing}\n")
        f.write(f"VGOUT {grid_dir}/{phase}.time\n")
        f.write(f"VGTYPE TIME\n")
        f.write(f"VGGRIDTYPE CARTESIAN\n")
        f.write(f"VGPHASE {phase}\n")
        f.write(f"VGVELOCITYMODEL {vel_model}\n")
        f.write(f"VGMAXTRAVELTIME 30.0\n")
        f.write(f"END\n")

def run_nlloc(control_file):
    """Esegue il comando NLLoc per il calcolo delle travel time."""
    subprocess.run(["Grid2Time", control_file], check=True)

def read_time_file(filename, n):
    """Legge il file .time binario e restituisce una matrice numpy 3D."""
    with open(filename, 'rb') as f:
        header = f.read(256)  # header NLLoc
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape((n, n, n))

def compute_travel_times(event_xyz, vel_model, n, spacing, out_dir="NLL_output"):
    """
    Calcola travel time P e S per un evento (x, y, z) usando NonLinLoc.
    """
    os.makedirs(out_dir, exist_ok=True)
    x0, y0, z0 = event_xyz

    # P
    ctrl_P = os.path.join(out_dir, "time_P.in")
    write_nlloc_control_file(ctrl_P, out_dir, vel_model, "P", x0, y0, z0, n, spacing)
    run_nlloc(ctrl_P)
    ttp = read_time_file(os.path.join(out_dir, "P.time"), n)

    # S
    ctrl_S = os.path.join(out_dir, "time_S.in")
    write_nlloc_control_file(ctrl_S, out_dir, vel_model, "S", x0, y0, z0, n, spacing)
    run_nlloc(ctrl_S)
    tts = read_time_file(os.path.join(out_dir, "S.time"), n)

    return ttp, tts


if __name__ == "__main__":
    event_xyz = (0.0, 0.0, 5.0)  # km
    
    vel_mod_dir="../VEL_MOD/"
    vel_model = "model3D.mod"      # file modello di velocità
    
    grid_length = 50  #km
    spacing = 1.0 # km
    n = int( round(grid_length/spacing) )

    ttp, tts = compute_travel_times(event_xyz, vel_model, n, spacing)

    print("Travel time P shape:", ttp.shape)
    print("Travel time S shape:", tts.shape)
    np.save("travel_time_P.npy", ttp)
    np.save("travel_time_S.npy", tts)