#!/usr/bin/env python
# coding: utf-8

# In[8]:


"""
Week 1 Exercise: Vectorise a PIV Post-Processing Pipeline
==========================================================
Instructions:
  - Do NOT edit the LOOP versions (they are your reference)
  - Fill in every section marked  TODO  in the NUMPY versions
  - Run the file when done — it will benchmark and compare both versions
  - Target: >=5x speedup

Run with:
    python piv_week1_exercise.py
"""

import numpy as np
import time


# =============================================================================
# SYNTHETIC DATA — do not change this
# =============================================================================

def generate_piv_field(nx=64, ny=64, noise_level=0.05, seed=42):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    XX, YY = np.meshgrid(x, y)
    U = np.sin(XX) * np.cos(YY) + rng.normal(0, noise_level, (ny, nx))
    V = -np.cos(XX) * np.sin(YY) + rng.normal(0, noise_level, (ny, nx))
    n_outliers = int(0.03 * nx * ny)
    oi = rng.integers(0, ny, n_outliers)
    oj = rng.integers(0, nx, n_outliers)
    U[oi, oj] = rng.uniform(-5, 5, n_outliers)
    V[oi, oj] = rng.uniform(-5, 5, n_outliers)
    return U, V

def generate_ensemble(n_frames=30, nx=64, ny=64):
    return [generate_piv_field(nx=nx, ny=ny, seed=i) for i in range(n_frames)]


# =============================================================================
# EXERCISE 1 OF 4: VORTICITY
# =============================================================================
# What it computes: omega_z = dV/dx - dU/dy
# The loop version manually steps through every grid point and applies
# forward/central/backward finite differences depending on position.
# Hint: np.gradient does all of this in one call.

# --- LOOP VERSION (do not edit) ---
def vorticity_loop(U, V, dx=1.0, dy=1.0):
    ny, nx = U.shape
    omega = [[0.0] * nx for _ in range(ny)]
    for i in range(ny):
        for j in range(nx):
            if j == 0:
                dvdx = (V[i][j+1] - V[i][j]) / dx
            elif j == nx - 1:
                dvdx = (V[i][j] - V[i][j-1]) / dx
            else:
                dvdx = (V[i][j+1] - V[i][j-1]) / (2*dx)
            if i == 0:
                dudy = (U[i+1][j] - U[i][j]) / dy
            elif i == ny - 1:
                dudy = (U[i][j] - U[i-1][j]) / dy
            else:
                dudy = (U[i+1][j] - U[i-1][j]) / (2*dy)
            omega[i][j] = dvdx - dudy
    return np.array(omega)

# --- NUMPY VERSION (you fill this in) ---
def vorticity_numpy(U, V, dx=1.0, dy=1.0):
    """
    TODO: replace the two lines below with np.gradient calls.

    np.gradient(field, spacing, axis=...)  returns the gradient along an axis.
      axis=0 means rows    → that's the y direction → dU/dy
      axis=1 means columns → that's the x direction → dV/dx

    You need two gradient calls, then subtract.
    """

    # TODO: compute dV/dx using np.gradient
    dvdx = np.gradient(V, dx, axis=1)  # replace this line

    # TODO: compute dU/dy using np.gradient
    dudy = np.gradient(U, dy, axis=0)  # replace this line

    # TODO: return dvdx - dudy
    return dvdx-dudy # replace this line


# =============================================================================
# EXERCISE 2 OF 4: TURBULENT KINETIC ENERGY (TKE)
# =============================================================================
# What it computes:
#   1. Mean U and V across the ensemble (time-average)
#   2. Fluctuation at each point: u' = U - U_mean,  v' = V - V_mean
#   3. TKE = average of  0.5 * (u'^2 + v'^2)  across all frames
#
# The loop version iterates over frames, then rows, then columns.
# Hint: stack all frames into a 3D array first, then use np.mean.

# --- LOOP VERSION (do not edit) ---
def tke_loop(fields):
    n = len(fields)
    ny, nx = fields[0][0].shape
    U_mean = [[0.0]*nx for _ in range(ny)]
    V_mean = [[0.0]*nx for _ in range(ny)]
    for U, V in fields:
        for i in range(ny):
            for j in range(nx):
                U_mean[i][j] += U[i][j]
                V_mean[i][j] += V[i][j]
    for i in range(ny):
        for j in range(nx):
            U_mean[i][j] /= n
            V_mean[i][j] /= n
    TKE = [[0.0]*nx for _ in range(ny)]
    for U, V in fields:
        for i in range(ny):
            for j in range(nx):
                u_prime = U[i][j] - U_mean[i][j]
                v_prime = V[i][j] - V_mean[i][j]
                TKE[i][j] += 0.5 * (u_prime**2 + v_prime**2)
    for i in range(ny):
        for j in range(nx):
            TKE[i][j] /= n
    return np.array(TKE), np.array(U_mean), np.array(V_mean)

# --- NUMPY VERSION (you fill this in) ---
def tke_numpy(fields):
    """
    TODO: use np.stack and np.mean to replace all the loops.

    Step 1 — stack all U arrays into one 3D array of shape (n_frames, ny, nx):
        U_stack = np.stack([U for U, V in fields], axis=0)

    Step 2 — compute the mean across axis=0 (the frames axis):
        U_mean = np.mean(U_stack, axis=0)     # shape: (ny, nx)

    Step 3 — subtract mean from each frame to get fluctuations:
        U_fluct = U_stack - U_mean            # broadcasting handles this

    Step 4 — compute TKE = mean over frames of 0.5*(u'^2 + v'^2)
    """

    # TODO: stack all U fields into shape (n_frames, ny, nx)
    U_stack = np.stack([U for U, V in fields], axis=0) # replace

    # TODO: stack all V fields the same way
    V_stack = np.stack([V for U, V in fields], axis=0)  # replace

    # TODO: compute mean U and mean V across frames (axis=0)
    U_mean =  np.mean(U_stack, axis=0)   # replace
    V_mean =  np.mean(V_stack, axis=0)   # replace

    # TODO: compute fluctuations (deviation from mean)
    U_fluct = U_stack-U_mean  # replace
    V_fluct = V_stack-V_mean  # replace

    # TODO: compute TKE = mean over frames of 0.5*(u'^2 + v'^2)
    TKE = np.mean(0.5*(U_fluct**2+V_fluct**2), axis=0)      # replace

    return TKE, U_mean, V_mean


# =============================================================================
# EXERCISE 3 OF 4: OUTLIER DETECTION (Normalised Median Test)
# =============================================================================
# What it computes:
#   For every vector, extract its 8 neighbours in a 3x3 window.
#   Compare the vector's magnitude to the median of those neighbours.
#   Flag it as an outlier if the normalised residual exceeds a threshold.
#
# This is the hardest exercise — the loop version has 5 nested loops.
# The NumPy approach uses np.lib.stride_tricks or scipy to build a
# "windowed" view of the array, then np.median on that.
#
# Hint: look up np.pad and np.lib.stride_tricks.sliding_window_view

# --- LOOP VERSION (do not edit) ---
def detect_outliers_loop(U, V, threshold=2.0):
    ny, nx = U.shape
    outlier_mask = [[False] * nx for _ in range(ny)]
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            neighbours_u = []
            neighbours_v = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    neighbours_u.append(U[i+di][j+dj])
                    neighbours_v.append(V[i+di][j+dj])
            med_u = sorted(neighbours_u)[len(neighbours_u)//2]
            med_v = sorted(neighbours_v)[len(neighbours_v)//2]
            res_u = [abs(u - med_u) for u in neighbours_u]
            res_v = [abs(v - med_v) for v in neighbours_v]
            med_res_u = max(sorted(res_u)[len(res_u)//2], 0.1)
            med_res_v = max(sorted(res_v)[len(res_v)//2], 0.1)
            r_u = abs(U[i][j] - med_u) / med_res_u
            r_v = abs(V[i][j] - med_v) / med_res_v
            if (r_u**2 + r_v**2)**0.5 > threshold:
                outlier_mask[i][j] = True
    return np.array(outlier_mask)

# --- NUMPY VERSION (you fill this in) ---
def detect_outliers_numpy(U, V, threshold=2.0):
    """
    TODO: use sliding_window_view to extract all 3x3 neighbourhoods at once.

    Step 1 — pad U and V by 1 on each edge (so border vectors get neighbours):
        U_pad = np.pad(U, pad_width=1, mode='edge')

    Step 2 — create a sliding window view of shape (ny, nx, 3, 3):
        from numpy.lib.stride_tricks import sliding_window_view
        windows_U = sliding_window_view(U_pad, (3, 3))
        # windows_U[i, j] is the 3x3 neighbourhood centred on U[i, j]

    Step 3 — reshape each window to 9 values, remove the centre (index 4):
        flat = windows_U.reshape(ny, nx, 9)
        neighbours = np.delete(flat, 4, axis=2)   # shape: (ny, nx, 8)

    Step 4 — compute median of neighbours at every point:
        med = np.median(neighbours, axis=2)        # shape: (ny, nx)

    Step 5 — compute normalised residual for the centre vector vs median,
              then apply threshold.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    ny, nx = U.shape

    # TODO: pad both U and V
    U_pad = np.pad(U, pad_width=1, mode='edge')  # replace
    V_pad = np.pad(V, pad_width=1, mode='edge')  # replace

    # TODO: create sliding window views
    windows_U = sliding_window_view(U_pad, (3, 3))  # replace
    windows_V = sliding_window_view(V_pad, (3, 3)) # replace

    # TODO: flatten windows and remove centre element (index 4)
    flat = windows_U.reshape(ny, nx, 9)
    neighbours_U =  np.delete(flat, 4, axis=2)     # replace
    flat2 = windows_V.reshape(ny, nx, 9)
    neighbours_V = np.delete(flat2, 4, axis=2)  # replace

    # TODO: median of neighbours
    med_U = np.median(neighbours_U, axis=2)   # replace
    med_V = np.median(neighbours_V, axis=2) 

    # TODO: median of residuals (for normalisation), floor at 0.1
    med_res_U = np.maximum(np.median(np.abs(neighbours_U - med_U[..., None]), axis=2), 0.1)
    med_res_V =  np.maximum(np.median(np.abs(neighbours_V - med_V[..., None]), axis=2), 0.1)


    # TODO: normalised residual for centre vector
    r_u = np.abs(U - med_U) / med_res_U
    r_v = np.abs(V - med_V) / med_res_V

    # TODO: return boolean mask where combined residual > threshold
    return (r_u + r_v) > threshold


# =============================================================================
# EXERCISE 4 OF 4: OUTLIER REPLACEMENT
# =============================================================================
# What it computes:
#   For each flagged outlier, replace it with the mean of its valid neighbours.
#
# Hint: this is the hardest to fully vectorise (valid neighbours vary per point).
# A good partial solution: use np.where and a convolution for the mean,
# then apply only where mask is True.

# --- LOOP VERSION (do not edit) ---
def replace_outliers_loop(U, V, mask):
    ny, nx = U.shape
    U_clean = [row[:] for row in U.tolist()]
    V_clean = [row[:] for row in V.tolist()]
    for i in range(ny):
        for j in range(nx):
            if mask[i][j]:
                valid_u, valid_v = [], []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < ny and 0 <= nj < nx and not mask[ni][nj]:
                            valid_u.append(U[ni][nj])
                            valid_v.append(V[ni][nj])
                if valid_u:
                    U_clean[i][j] = sum(valid_u) / len(valid_u)
                    V_clean[i][j] = sum(valid_v) / len(valid_v)
    return np.array(U_clean), np.array(V_clean)

# --- NUMPY VERSION (you fill this in) ---
def replace_outliers_numpy(U, V, mask):
    """
    TODO: use scipy.ndimage.uniform_filter as a neighbourhood mean,
    then use np.where to apply replacements only at outlier locations.

    Step 1 — compute neighbourhood mean using a 3x3 uniform filter:
        from scipy.ndimage import uniform_filter
        U_mean_local = uniform_filter(U, size=3)

    Step 2 — replace outlier positions only:
        U_clean = np.where(mask, U_mean_local, U)

    Note: uniform_filter includes the centre pixel in the mean, which
    introduces a small error at outlier locations (you're averaging in
    the bad value). Acceptable for this exercise — note it in a comment.
    """
    from scipy.ndimage import uniform_filter

    # TODO: compute local neighbourhood mean for U and V
    U_local_mean =  uniform_filter(U, size=3)  # replace
    V_local_mean =  uniform_filter(V, size=3)  # replace

    # TODO: replace outliers with local mean, keep valid vectors unchanged
    U_clean = np.where(mask, U_local_mean, U)  # replace
    V_clean = np.where(mask, V_local_mean, V)  # replace

    return U_clean, V_clean


# =============================================================================
# BENCHMARK — runs automatically when you execute this file
# =============================================================================

def run_benchmark():
    N_FRAMES = 30
    NX, NY = 64, 64

    print("=" * 60)
    print("Week 1 Benchmark: Loop vs NumPy")
    print(f"Ensemble: {N_FRAMES} frames | Grid: {NX}x{NY}")
    print("=" * 60)

    fields = generate_ensemble(n_frames=N_FRAMES, nx=NX, ny=NY)
    U0, V0 = fields[0]

    results = {}

    # ── Exercise 1: Vorticity ──────────────────────────────────────────────
    print("\n[ Exercise 1: Vorticity ]")

    t0 = time.perf_counter()
    for U, V in fields:
        omega_loop = vorticity_loop(U, V)
    loop_time = time.perf_counter() - t0
    print(f"  Loop  : {loop_time:.4f}s")

    try:
        t0 = time.perf_counter()
        for U, V in fields:
            omega_np = vorticity_numpy(U, V)
        np_time = time.perf_counter() - t0

        if omega_np is None:
            print("  NumPy : not implemented yet (returns None)")
        else:
            max_err = np.abs(omega_np - omega_loop).max()
            speedup = loop_time / np_time
            print(f"  NumPy : {np_time:.4f}s  |  speedup: {speedup:.1f}x  |  max error: {max_err:.6f}")
            results['vorticity'] = speedup
    except Exception as e:
        print(f"  NumPy : error — {e}")

    # ── Exercise 2: TKE ───────────────────────────────────────────────────
    print("\n[ Exercise 2: TKE ]")

    t0 = time.perf_counter()
    TKE_loop, Um_loop, Vm_loop = tke_loop(fields)
    loop_time = time.perf_counter() - t0
    print(f"  Loop  : {loop_time:.4f}s")

    try:
        t0 = time.perf_counter()
        TKE_np, Um_np, Vm_np = tke_numpy(fields)
        np_time = time.perf_counter() - t0

        if TKE_np is None:
            print("  NumPy : not implemented yet (returns None)")
        else:
            max_err = np.abs(TKE_np - TKE_loop).max()
            speedup = loop_time / np_time
            print(f"  NumPy : {np_time:.4f}s  |  speedup: {speedup:.1f}x  |  max error: {max_err:.6f}")
            results['tke'] = speedup
    except Exception as e:
        print(f"  NumPy : error — {e}")

    # ── Exercise 3: Outlier Detection ─────────────────────────────────────
    print("\n[ Exercise 3: Outlier Detection ]")

    t0 = time.perf_counter()
    for U, V in fields:
        mask_loop = detect_outliers_loop(U, V)
    loop_time = time.perf_counter() - t0
    print(f"  Loop  : {loop_time:.4f}s")

    try:
        t0 = time.perf_counter()
        for U, V in fields:
            mask_np = detect_outliers_numpy(U, V)
        np_time = time.perf_counter() - t0

        if mask_np is None:
            print("  NumPy : not implemented yet (returns None)")
        else:
            agreement = (mask_np == mask_loop).mean() * 100
            speedup = loop_time / np_time
            print(f"  NumPy : {np_time:.4f}s  |  speedup: {speedup:.1f}x  |  agreement: {agreement:.1f}%")
            results['outlier_detect'] = speedup
    except Exception as e:
        print(f"  NumPy : error — {e}")

    # ── Exercise 4: Outlier Replacement ───────────────────────────────────
    print("\n[ Exercise 4: Outlier Replacement ]")

    mask = detect_outliers_loop(U0, V0)
    t0 = time.perf_counter()
    for U, V in fields:
        U_c_loop, V_c_loop = replace_outliers_loop(U, V, mask)
    loop_time = time.perf_counter() - t0
    print(f"  Loop  : {loop_time:.4f}s")

    try:
        t0 = time.perf_counter()
        for U, V in fields:
            U_c_np, V_c_np = replace_outliers_numpy(U, V, mask)
        np_time = time.perf_counter() - t0

        if U_c_np is None:
            print("  NumPy : not implemented yet (returns None)")
        else:
            speedup = loop_time / np_time
            print(f"  NumPy : {np_time:.4f}s  |  speedup: {speedup:.1f}x")
            results['outlier_replace'] = speedup
    except Exception as e:
        print(f"  NumPy : error — {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if results:
        avg = sum(results.values()) / len(results)
        print(f"Exercises complete: {len(results)}/4")
        print(f"Average speedup   : {avg:.1f}x")
        if avg >= 5.0:
            print("TARGET MET: >=5x speedup achieved")
        else:
            print(f"Keep going: need {5.0:.1f}x, currently at {avg:.1f}x")
    else:
        print("No exercises implemented yet — fill in the TODOs above.")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()


# In[ ]:





# In[ ]:




