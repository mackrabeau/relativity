"""
Relativistic physics engine.

Flat-spacetime approximation with linearised Schwarzschild sources.
Metric (isotropic weak-field, G = c = 1):
    g = diag[ -(1-h), (1+h), (1+h), (1+h) ]
    h = Σ  r_s_i / |r - r_i|   (superposition, clamped away from horizon)

Integration variable: proper time τ for each body.
"""

import numpy as np

C = 1.0          # speed of light (geometrised units)
G = 1.0          # gravitational constant


class Body:
    """Massive body on a worldline x^μ(τ)."""

    def __init__(self, name, mass, pos3, vel3, radius=1.0, color="#ffffff"):
        self.name   = name
        self.mass   = float(mass)
        self.r_s    = 2.0 * self.mass      # Schwarzschild radius
        self.radius = float(radius)        # visual / collision radius
        self.color  = color

        pos3 = np.asarray(pos3, dtype=float)
        vel3 = np.asarray(vel3, dtype=float)

        self.x   = np.zeros(4)             # 4-position  [t, x, y, z]
        self.x[1:] = pos3
        self.u   = self._init_u(vel3)      # 4-velocity  [u^t, u^x, u^y, u^z]
        self.tau = 0.0                     # accumulated proper time
        self.trail: list = []

    def _init_u(self, vel3):
        v2 = float(np.dot(vel3, vel3))
        if v2 >= 1.0:
            raise ValueError(f"{self.name}: speed {v2**0.5:.3f}c ≥ c")
        gamma = 1.0 / (1.0 - v2) ** 0.5
        u = np.zeros(4)
        u[0]  = gamma * C
        u[1:] = gamma * vel3
        return u

    # ── helpers ────────────────────────────────────────────────────────────

    def spatial_pos(self):
        return self.x[1:4].copy()

    def coord_vel(self):
        """Coordinate 3-velocity  v^i = u^i / u^t."""
        return self.u[1:] / self.u[0]

    def speed(self):
        v = self.coord_vel()
        return float(np.dot(v, v) ** 0.5)

    def lorentz_gamma(self):
        return float(self.u[0] / C)


# ── Metric & Christoffel (analytical) ──────────────────────────────────────

def _h_and_grad(pos3, sources, skip=None):
    """Compute  h = Σ r_s/r  and  ∂_i h  (3-vector) at spatial position pos3."""
    h   = 0.0
    dh  = np.zeros(3)
    for src in sources:
        if src is skip:
            continue
        dr = pos3 - src.x[1:4]
        r  = max(float(np.linalg.norm(dr)), 1.1 * src.r_s)
        h  += src.r_s / r
        dh += -src.r_s * dr / r ** 3      # ∂_i(r_s/r) = -r_s Δx_i / r³
    h = min(h, 0.95)                       # keep metric non-degenerate
    return h, dh


def christoffel(x, sources, skip=None):
    """
    Analytical Christoffel symbols  Γ^μ_{αβ}  for the isotropic weak-field metric.
    Non-zero components (quasi-static: ∂_t g = 0):
        Γ^t_{ti} = Γ^t_{it} = -∂_i h / [2(1-h)]
        Γ^i_{tt}             = -∂_i h / [2(1+h)]
        Γ^i_{jk}             = (δ_{ik}∂_j h + δ_{ij}∂_k h - δ_{jk}∂_i h) / [2(1+h)]
    Returns ndarray of shape (4, 4, 4).
    """
    h, dh = _h_and_grad(x[1:4], sources, skip)
    ft = 0.5 / (1.0 - h)
    fs = 0.5 / (1.0 + h)

    G = np.zeros((4, 4, 4))
    for i in range(3):
        ii = i + 1
        G[0,  0, ii] = -dh[i] * ft        # Γ^t_{ti}
        G[0, ii,  0] = -dh[i] * ft        # Γ^t_{it}
        G[ii,  0,  0] = -dh[i] * fs       # Γ^i_{tt}
        for j in range(3):
            jj = j + 1
            for k in range(3):
                kk = k + 1
                G[ii, jj, kk] = fs * (
                    (1 if i == k else 0) * dh[j]
                    + (1 if i == j else 0) * dh[k]
                    - (1 if j == k else 0) * dh[i]
                )
    return G


def geodesic_accel(x, u, sources, skip=None):
    """du^μ/dτ = -Γ^μ_{αβ} u^α u^β"""
    Gam = christoffel(x, sources, skip)
    return -np.einsum("mab,a,b->m", Gam, u, u)


# ── Integrator ──────────────────────────────────────────────────────────────

def rk4_step(body, sources, dtau):
    """Advance body by dtau in proper time (RK4)."""
    x0, u0 = body.x.copy(), body.u.copy()

    def deriv(x, u):
        return u, geodesic_accel(x, u, sources, skip=body)

    k1x, k1u = deriv(x0, u0)
    k2x, k2u = deriv(x0 + 0.5 * dtau * k1x, u0 + 0.5 * dtau * k1u)
    k3x, k3u = deriv(x0 + 0.5 * dtau * k2x, u0 + 0.5 * dtau * k2u)
    k4x, k4u = deriv(x0 + dtau * k3x,        u0 + dtau * k3u)

    body.x   = x0 + (dtau / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    body.u   = u0 + (dtau / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
    body.tau += dtau


def renormalize(body, sources):
    """Re-enforce  u^μ u_μ = -c²  by rescaling u^t (drift correction)."""
    h, _ = _h_and_grad(body.x[1:4], sources, skip=body)
    h = min(h, 0.95)
    g_tt    = -(1.0 - h)
    g_ii    =  (1.0 + h)
    spatial = g_ii * float(np.dot(body.u[1:], body.u[1:]))
    u_t_sq  = (-C ** 2 - spatial) / g_tt
    if u_t_sq > 0:
        body.u[0] = u_t_sq ** 0.5


# ── Doppler colour ──────────────────────────────────────────────────────────

def doppler_color(body, observer2d=(0.0, 0.0)):
    """
    RGBA colour encoding relativistic Doppler shift toward observer.
    Blue = blueshift (approaching), Red = redshift (receding).
    """
    src  = body.x[1:3]
    obs  = np.asarray(observer2d, dtype=float)
    rvec = obs - src
    dist = float(np.linalg.norm(rvec))
    if dist < 1e-10:
        return (1.0, 1.0, 1.0, 1.0)

    r_hat = rvec / dist
    v     = body.coord_vel()[:2]
    v_r   = float(np.dot(v, r_hat))               # >0 → moving toward observer
    v_r   = float(np.clip(v_r, -0.9999, 0.9999))

    # f_obs/f_emit = sqrt((1 + v_r) / (1 - v_r))
    doppler = ((1.0 + v_r) / (1.0 - v_r)) ** 0.5
    shift   = float(np.clip(np.log(doppler), -2.0, 2.0)) / 2.0   # −1..+1

    if shift >= 0.0:                               # blueshift
        return (1.0 - shift, 1.0 - shift, 1.0, 0.9)
    else:                                          # redshift
        s = -shift
        return (1.0, 1.0 - s, 1.0 - s, 0.9)


# ── Collision ───────────────────────────────────────────────────────────────

def bodies_overlap(b1, b2, scale=1.0):
    dr = b1.x[1:4] - b2.x[1:4]
    return float(np.linalg.norm(dr)) < scale * (b1.radius + b2.radius)


def elastic_bounce(b1, b2, restitution=0.75):
    """Arcade elastic bounce along the collision normal."""
    dr   = b1.x[1:4] - b2.x[1:4]
    dist = float(np.linalg.norm(dr))
    if dist < 1e-10:
        return
    n    = dr / dist
    v1   = b1.coord_vel()
    v2   = b2.coord_vel()
    vn   = float(np.dot(v1 - v2, n))
    if vn >= 0:                                    # already separating
        return

    m1 = b1.mass + 1e-6
    m2 = b2.mass + 1e-6
    j  = -(1.0 + restitution) * vn / (1.0 / m1 + 1.0 / m2)

    b1.u[1:4] += (j / m1) * n * (b1.u[0] / C)
    b2.u[1:4] -= (j / m2) * n * (b2.u[0] / C)

    # push apart to prevent sticking
    overlap = (b1.radius + b2.radius) - dist + 0.01
    b1.x[1:4] +=  0.5 * overlap * n
    b2.x[1:4] -=  0.5 * overlap * n
