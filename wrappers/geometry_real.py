"""Our real-data fan-beam geometry for RBG-Diff (Mayo Siemens / GE).

Takes RBG-Diff onto real clinical projection data: loads a geometry json, builds
``torch_radon`` fan-beam operators on the 256^2 / FOV=330 recon grid, reconstructs
a full-view mu image from a real measured sinogram, and re-points a net's radon /
image_radon / num_full_views to the real geometry for training and evaluation.
It re-points a live net instance's radon ops so ``generate_sparse_and_full_ct``
(inherited from ``DiffusionSparseWrapper``) produces real-geometry recons.
Depends only on numpy / torch / torch_radon.

Two functions:
  * PREPROCESSING (real measured sinogram -> GT mu image): detector flip + scale +
    real geometry (``full_recon_from_sino``).
  * TRAINING / EVAL re-degrade (GT mu image -> sparse via forward+FBP): real
    geometry only, since the image is already in mu units (``patch_net_real``).
"""
import json
import types
import numpy as np
import torch
from torch_radon import RadonFanbeam

MU_WATER = 0.192


# ----------------------------- geometry json -----------------------------
def load_geom(path):
    with open(path) as f:
        g = json.load(f)
    for k in ("rotview", "nu", "du", "dso", "dsd", "hu_factor", "angle_offset",
              "recon_size", "fov", "mu_water"):
        assert k in g, f"geom json missing key: {k}"
    return g


def pixel_scale(geom):
    """mm per pixel on our recon grid."""
    return geom["fov"] / geom["recon_size"]


# ----------------------------- angle helpers -----------------------------
def uniform_angles(rotview, offset):
    return (np.linspace(0, 2 * np.pi, rotview, endpoint=False) + offset).astype(np.float32)


def realeven_angles(rotview, V, offset):
    """V angles evenly subsampled from the real (uniform) full set over 2pi. For V | rotview this
    is exactly the uniform-V subset (Siemens 2304/1152 & 72/36/18); for non-divisor V (GE 984) it
    is the nearest-even approximation. Single, reviewer-defensible rule (realeven)."""
    full = np.linspace(0, 2 * np.pi, rotview, endpoint=False)
    idx = np.round(np.linspace(0, rotview, V, endpoint=False)).astype(int) % rotview
    return (full[idx] + offset).astype(np.float32)


# ----------------------------- operators -----------------------------
def _fanbeam(geom, angles):
    s = pixel_scale(geom)
    return RadonFanbeam(geom["recon_size"], angles,
                        source_distance=geom["dso"] / s,
                        det_distance=(geom["dsd"] - geom["dso"]) / s,
                        det_count=geom["nu"],
                        det_spacing=geom["du"] / s)


def build_fanbeam(geom, V):
    """sparse (or full if V==rotview) fan operator at V views, real geometry, realeven angles."""
    return _fanbeam(geom, realeven_angles(geom["rotview"], V, geom["angle_offset"]))


def data_circle_radius_mm(geom):
    """Reconstructable-circle radius (mm). 256^2 square corner (fov/2*sqrt2) must be < this to
    avoid forward-projection truncation -> iterative ring self-bootstrapping explosion."""
    half_fan = np.arctan((geom["nu"] * geom["du"] / 2.0) / geom["dsd"])
    return geom["dso"] * np.sin(half_fan)


def square_corner_mm(geom):
    return geom["fov"] / 2.0 * np.sqrt(2.0)


# ----------------------------- HU / mu -----------------------------
def mu2HU(mu, mu_water=MU_WATER):
    return (mu - mu_water) / mu_water * 1000.0


def HU2mu(hu, mu_water=MU_WATER):
    return hu / 1000.0 * mu_water + mu_water


# ----------------------------- preprocessing recon -----------------------------
def real_full_angles(geom, meta_angles):
    """The REAL acquisition angles for the full-view recon: meta['angles'][:rotview] + angle_offset.
    These are uniform-STEP but start at the scan's true start angle a0 = meta['angles'][0] (NOT 0)."""
    return (np.asarray(meta_angles, dtype=np.float64)[:geom["rotview"]] + geom["angle_offset"]).astype(np.float32)


def full_recon_from_sino(geom, sino2d_np, meta_angles, device="cuda"):
    """REAL measured sinogram (rotview, nu) -> mu image (1,1,H,W). flip(axis) + scale + real-geom FBP.
    MUST use the REAL acquisition angles `meta_angles` (from the rebinned tif) — the sinogram rows correspond
    to those angles. Using UNIFORM angles would ROTATE the recon by the scan start angle a0=meta_angles[0]
    (e.g. Siemens abd a0=-5.825 rad => ~26deg rotation vs vendor)."""
    s = pixel_scale(geom)
    scale = (geom["mu_water"] / geom["hu_factor"]) / s
    flip_axis = geom.get("flip_axis", None)
    sino = np.copy(np.flip(sino2d_np, axis=flip_axis)) if flip_axis is not None else np.ascontiguousarray(sino2d_np)
    op = _fanbeam(geom, real_full_angles(geom, meta_angles))
    t = torch.from_numpy(sino.astype(np.float32))[None, None].to(device) * scale
    return op.backprojection(op.filter_sinogram(t, "ram-lak"))


def forward_has_nan(geom, mu_img, V=None):
    """dummy forward-projection NaN check (torch_radon RadonFanbeam.forward NaNs on some angle sets).
    Returns True if the forward output has NaN/inf -> caller should SKIP the case (never auto-shift)."""
    V = geom["rotview"] if V is None else V
    op = build_fanbeam(geom, V)
    with torch.no_grad():
        sino = op.forward(mu_img)
    return bool(torch.isnan(sino).any() or torch.isinf(sino).any())


# ----------------------------- training / eval monkeypatch -----------------------------
def patch_net_real(net, geom, num_full_views=None, cache_ops=True):
    """Re-point net.radon / net.image_radon / net.num_full_views to the REAL fan geometry so the
    net's own generate_sparse_and_full_ct produces real-geometry sparse/full recons. Uses EXACT real
    params + realeven angles. NO flip / NO scale here (the image is already mu; flip/scale belong to
    preprocessing only). num_full_views defaults to the geometry's real rotview. Patches THIS instance
    only (also call on ema_net). Returns net."""
    off = geom["angle_offset"]
    nfv = geom["rotview"] if num_full_views is None else num_full_views
    _cache = {}

    def _op(V):
        if not cache_ops:
            return _fanbeam(geom, realeven_angles(geom["rotview"], V, off))
        if V not in _cache:
            _cache[V] = _fanbeam(geom, realeven_angles(geom["rotview"], V, off))
        return _cache[V]

    def image_radon(self, ct_image, num_views=None, angle_bias=0):
        V = self.num_full_views if num_views is None else num_views
        return _op(V).forward(ct_image)

    def radon(self, sinogram, num_views=None, angle_bias=0):
        V = self.num_full_views if num_views is None else num_views
        op = _op(V)
        return op.backprojection(op.filter_sinogram(sinogram, "ram-lak"))

    net.num_full_views = nfv
    net.image_radon = types.MethodType(image_radon, net)
    net.radon = types.MethodType(radon, net)
    net._real_geom = geom
    return net
