import numpy as num

class Angles_NLL:
    """
    Read NonLinLoc ANGLE grids (ANGLE1, ANGLE2)
    """

    def __init__(self, db_path, hdr_filename):
        self.db_path = db_path
        self.hdr_filename = hdr_filename
        self._read_header()

    def _read_header(self):
        hdr = open(f"{self.db_path}/{self.hdr_filename}").readlines()
        for line in hdr:
            if line.startswith("NX"):
                self.nx = int(line.split()[1])
            elif line.startswith("NY"):
                self.ny = int(line.split()[1])
            elif line.startswith("NZ"):
                self.nz = int(line.split()[1])
            elif line.startswith("DX"):
                self.dx = float(line.split()[1])
            elif line.startswith("DY"):
                self.dy = float(line.split()[1])
            elif line.startswith("DZ"):
                self.dz = float(line.split()[1])
            elif line.startswith("XORIG"):
                self.ox = float(line.split()[1])
            elif line.startswith("YORIG"):
                self.oy = float(line.split()[1])
            elif line.startswith("ZORIG"):
                self.oz = float(line.split()[1])

    def load_angles(self, phase, precision="single"):
        """
        Load ANGLE grids for phase P or S
        Returns:
            angle1, angle2 : 3D numpy arrays
        """
        fname = f"{self.db_path}/angle.{phase}.time"
        dtype = num.float32 if precision == "single" else num.float64

        data = num.fromfile(fname, dtype=dtype)

        data = data.reshape(self.nz, self.ny, self.nx, 2)

        angle1 = data[..., 0]  # take-off angle
        angle2 = data[..., 1]  # azimuth

        return angle1, angle2
