import numpy as np

def generate_focal_mechanisms(resolution_deg):
    """
    Generate focal mechanisms (strike, dip, rake) using a 3D Fibonacci spiral.
    The number of samples N is determined by the 3D angular resolution.
    """
    # Define total angular volume and approximate number of samples
    strike_range = 360
    dip_range = 90
    rake_range = 360
    total_volume = strike_range * dip_range * rake_range
    voxel_volume = resolution_deg**3
    n = int(total_volume / voxel_volume)

    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    inv_phi2 = 1 / (phi**2)
    inv_phi3 = 1 / (phi**3)

    indices = np.arange(n)
    # Quasi-uniform 3D Fibonacci lattice
    x = np.mod(indices * inv_phi2, 1)
    y = np.mod(indices * inv_phi3, 1)
    z = np.mod(indices * (1/phi), 1)

    # Map to focal mechanism parameters
    strike = 360 * x
    dip = 90 * y
    rake = 360 * z - 180  # -180 to 180

    # Use a structured NumPy array for compact, efficient storage
    focals = np.zeros(n, dtype=[('strike', 'f4'), ('dip', 'f4'), ('rake', 'f4')])
    focals['strike'] = strike.astype('f4')
    focals['dip'] = dip.astype('f4')
    focals['rake'] = rake.astype('f4')

    return focals

def save_focals(focals, filename="focals_3d.npz"):
    np.savez_compressed(filename, focals=focals)

def load_focals(filename="focals_3d.npz"):
    return np.load(filename, allow_pickle=False)['focals']


# Example usage:
if __name__ == "__main__":
    focals = generate_focal_mechanisms(90)
    print(f"Generated {len(focals)} focal mechanisms.")
    print(focals[:5])
    save_focals(focals, "focals_3d_5deg.npz")
    loaded = load_focals("focals_3d_5deg.npz")
    print("Reloaded:", loaded[:5])
