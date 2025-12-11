import math

class FocalMechanismSampler:
    """
    Efficient generator of (strike, dip, rake) triplets for N events using
    Fibonacci/golden-ratio additive sequences (low-discrepancy).

    Usage:
        samp = FocalMechanismSampler(N)
        strike, dip, rake = samp.next()      # repeated N times, or
        for s, d, r in samp:                 # iterator style

    Conventions:
      - strike: azimuth in degrees [0, 360), measured clockwise from North.
      - dip   : degrees [0, 90].
      - rake  : degrees [-180, 180).
    """

    def __init__(self, N, start_index: int = 0):
        if N <= 0:
            raise ValueError("N must be positive.")
        self.N = int(N)
        # sequence index (0 .. N-1)
        self.i = int(start_index) % self.N

        # three incommensurate step sizes (irrationals related to golden ratio)
        sqrt5 = math.sqrt(5.0)
        phi = (1.0 + sqrt5) / 2.0              # golden ratio ~1.618...
        # use fractional steps in (0,1); these are "Fibonacci-related" and well-spread
        self.alpha = (phi - 1.0)               # 1/phi ≈ 0.6180339887  (good low-discrepancy)
        self.beta  = (math.sqrt(2.0) - 1.0)    # ≈0.41421356  (incommensurate with alpha)
        self.gamma = (math.e - 2.0)            # ≈0.7182818   (another irrational)

        # small precomputed constants
        self._two_pi = 2.0 * math.pi

    @staticmethod
    def _frac(x):
        """Return fractional part in [0,1)."""
        return x - math.floor(x)

    def _index_frac_triplet(self, idx):
        """Return three fractional coordinates in [0,1) for given index."""
        # Additive recurrence (cheap) -- ensures no array allocations
        a = self._frac(idx * self.alpha)
        b = self._frac(idx * self.beta)
        c = self._frac(idx * self.gamma)
        return a, b, c

    def _triplet_to_sdr(self, a, b, c):
        """
        Map three uniform fractions (a,b,c) to strike,dip,rake.

        - strike: uniform on [0,360)
        - dip: sampled so plane normals are uniform on the sphere (use inverse CDF)
        - rake: uniform on [-180,180)
        """
        # strike: uniform [0,360)
        strike = 360.0 * a

        # dip: sample unit-sphere normal's z component uniformly in [-1,1]
        # using b => nz = 1 - 2*b, then dip = arccos(|nz|) in degrees in [0,90]
        nz = 1.0 - 2.0 * b
        # numeric safety
        if nz > 1.0: nz = 1.0
        if nz < -1.0: nz = -1.0
        dip = math.degrees(math.acos(abs(nz)))  # 0..90

        # rake: uniform in [-180, 180)
        rake = 360.0 * (c - 0.5)
        # normalize to [-180,180)
        if rake >= 180.0:
            rake -= 360.0
        if rake < -180.0:
            rake += 360.0

        return strike, dip, rake

    def next(self):
        """Return the next (strike, dip, rake) and advance the internal index."""
        if self.i >= self.N:
            raise StopIteration("All N samples have been produced.")
        a, b, c = self._index_frac_triplet(self.i)
        s, d, r = self._triplet_to_sdr(a, b, c)
        self.i += 1
        return s, d, r

    # Python iterator protocol
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self, start_index: int = 0):
        """Reset or set the index (useful if you want deterministic restart)."""
        self.i = int(start_index) % self.N

    def resolution_estimate(self):
        """
        Return estimates of the sampling resolution.

        - 'angular_spacing_rad' : characteristic angular spacing (radians) on a sphere
            ~ sqrt(4*pi/N) (approx linear scale of nearest-neighbour spacing).
        - 'angular_spacing_deg' : same in degrees.
        - 'param_space_cell_deg' : cubic-root resolution in the naive (strike,dip,rake) box [deg].
            That is: ( (360 * 90 * 360) / N )^(1/3) degrees.
        - 'notes' : brief text noting interpretation.
        """
        N = float(self.N)
        area_per_point = 4.0 * math.pi / N          # sr (steradians)
        # approximate linear angular scale (radians)
        ang_rad = math.sqrt(area_per_point)
        ang_deg = math.degrees(ang_rad)
        # naive parameter-space cubic cell size (degrees)
        V = 360.0 * 90.0 * 360.0
        cell_deg = (V / N) ** (1.0 / 3.0)

        return {
            "angular_spacing_rad": ang_rad,
            "angular_spacing_deg": ang_deg,
            "param_space_cell_deg": cell_deg,
            "notes": (
                "angular_spacing ~ characteristic angular separation between plane normals "
                "(useful for spherical coverage). param_space_cell_deg is a naive cubic-root "
                "resolution for the 3D box (strike [deg], dip [deg], rake [deg]). "
                "Dip sampling accounts for sphere-area weighting."
            )
        }

if __name__ == "__main__":
    nsources = 500
    samp = FocalMechanismSampler(nsources)
    for idx in range(nsources):
        strike, dip, rake = samp.next()
        print(f'fm number: {idx}, strike: {strike}, dip: {dip}, rake: {rake}')

    print(f'\nfocal sphere resolution: {samp.resolution_estimate()} [deg]')
