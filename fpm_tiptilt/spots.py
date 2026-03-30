import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit


img = mpimg.imread("/Users/popenoe1/Documents/fastrSHWFS/fpm_tiptilt/tt_stitch.png")
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, "tt_stitch.png")
img = mpimg.imread(img_path)

cropped = img[:, 260:-550]
img=cropped
'''
plt.imshow(cropped, cmap="gray")
plt.axis("off")
plt.show()
'''

pixel_scale = 5.86 #microns per pixel
half_size = 20

# ---------- Centroiding  ----------

if img.ndim == 3:
    img_gray = img.mean(axis=2)
else:
    img_gray = img

img_smooth = ndi.gaussian_filter(img_gray, sigma=2)
threshold = img_gray.max() * 0.75
binary = img_gray > threshold
labels, num_labels = ndi.label(binary)
print("Number of detected spots:", num_labels)

sums = ndi.sum(img_smooth, labels, index=range(1, num_labels + 1))
centroids = ndi.center_of_mass(img_smooth, labels, index=range(1, num_labels + 1))
indices = np.argsort(sums)[::-1]

N = min(500, num_labels) 
all_top_indices = indices[:N]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_smooth, cmap="gray")
ax.set_axis_off()

for rank, idx in enumerate(all_top_indices):
    y, x = centroids[idx]
    ax.plot(x, y, "r+", markersize=10, markeredgewidth=2)
    ax.text(x + 3, y + 3, str(rank + 1), color="red", fontsize=8)

plt.tight_layout()
plt.show()

all_top_indices = indices[:N]

bad_ranks_1based = [54, 73, 72, 69, 64, 60, 1, 67, 49, 68, 59, 61, 66]
bad_ranks_0based = [r - 1 for r in bad_ranks_1based]

top_indices = [
    idx for rank, idx in enumerate(all_top_indices)
    if rank not in bad_ranks_0based
]

print(np.shape(img_smooth))

# ---------- FWHM Calculation ----------

def fwhm_1d(x, y):
    """
    Compute FWHM of a 1D peak y(x).
    Returns FWHM in x units, or np.nan if it fails.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.size == 0:
        return np.nan

    y_max = y.max()
    if y_max <= 0:
        return np.nan

    half_max = y_max / 2.0

    above = np.where(y >= half_max)[0]
    if len(above) < 2:
        return np.nan

    left_idx = above[0]
    right_idx = above[-1]
    return x[right_idx] - x[left_idx]


def extract_spot_region(img_gray, x0, y0, half_size=half_size):
    """
    Extract a square region around (x0, y0) from img_gray.
    x0, y0 are pixel coordinates (x=col, y=row) in the full image.
    Returns subimg and local center (x0_local, y0_local).
    """
    h, w = img_gray.shape

    x0 = int(round(x0))
    y0 = int(round(y0))

    x_min = max(0, x0 - half_size)
    x_max = min(w, x0 + half_size + 1)
    y_min = max(0, y0 - half_size)
    y_max = min(h, y0 + half_size + 1)

    subimg = img_gray[y_min:y_max, x_min:x_max]

    x0_local = x0 - x_min
    y0_local = y0 - y_min

    return subimg, x0_local, y0_local

# ---------------- Gaussian fitting ----------------

def gaussian_1d(x, A, x0, sigma, offset):
    """1D Gaussian: A * exp(-(x - x0)^2 / (2 sigma^2)) + offset"""
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

def fit_gaussian_to_profile(x, y):
    """
    Fit a 1D Gaussian to a 1D profile y(x).
    Returns (A, x0, sigma, offset, fwhm_pix) or all np.nan on failure.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if y.size < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Initial guesses
    A0 = y.max() - y.min()
    offset0 = y.min()
    x0_0 = x[np.argmax(y)]
    # crude width guess
    sigma0 = max(1.0, 0.25 * len(x))

    p0 = [A0, x0_0, sigma0, offset0]

    try:
        popt, pcov = curve_fit(gaussian_1d, x, y, p0=p0)
        A, x0_fit, sigma_fit, offset_fit = popt
        fwhm_pix = 2.3548 * sigma_fit  # FWHM = 2*sqrt(2*ln2) * sigma ≈ 2.3548 * sigma
        return A, x0_fit, sigma_fit, offset_fit, fwhm_pix
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
def spot_fwhm_from_centroid(img_gray, centroid, half_size=half_size, pixel_scale=pixel_scale):
    """
    Compute FWHM for a spot around a given centroid, using Gaussian fits only.
    Returns:
        fwhm_x_pix, fwhm_y_pix, fwhm_x_phys, fwhm_y_phys
    """
    cy, cx = centroid 

    # extract subimage around the centroid
    subimg, x0_local, y0_local = extract_spot_region(
        img_gray,
        cx, 
        cy,  
        half_size=half_size,
    )

    # 1D profiles through the center
    profile_x = subimg[y0_local, :]
    profile_y = subimg[:, x0_local]

    x_coords = np.arange(profile_x.size)
    y_coords = np.arange(profile_y.size)

    # Gaussian fit FWHM along x and y
    A_x, x0_fit_x, sigma_x, offset_x, fwhm_x_gauss = fit_gaussian_to_profile(x_coords, profile_x)
    A_y, x0_fit_y, sigma_y, offset_y, fwhm_y_gauss = fit_gaussian_to_profile(y_coords, profile_y)

    fwhm_x_pix = fwhm_x_gauss
    fwhm_y_pix = fwhm_y_gauss

    fwhm_x_phys = fwhm_x_pix * pixel_scale
    fwhm_y_phys = fwhm_y_pix * pixel_scale

    return fwhm_x_pix, fwhm_y_pix, fwhm_x_phys, fwhm_y_phys


# ---------------- results ----------------

skip_k = 1  # number of brightest detections to skip
fwhm_results = []

for rank, idx in enumerate(top_indices[skip_k:], start=skip_k):
    centroid = centroids[idx]

    fwhm_x_pix, fwhm_y_pix, fwhm_x_phys, fwhm_y_phys = spot_fwhm_from_centroid(
        img_gray,
        centroid,
        half_size=half_size,
        pixel_scale=pixel_scale,
    )

    fwhm_results.append({
        "spot_index": int(idx),        
        "rank": int(rank),              
        "centroid_y": float(centroid[0]),
        "centroid_x": float(centroid[1]),
        "fwhm_x_pix": float(fwhm_x_pix),
        "fwhm_y_pix": float(fwhm_y_pix),
        "fwhm_x_phys": float(fwhm_x_phys),
        "fwhm_y_phys": float(fwhm_y_phys),
    })


y_line = 211 # choose where row should fall in pixels
chosen_ranks_1based = [53,20,45,43]  # for example

# Remove any that are in the bad list
chosen_ranks_1based = [
    r for r in chosen_ranks_1based
    if r not in bad_ranks_1based
]

chosen_ranks_0based = [r - 1 for r in chosen_ranks_1based]

chosen_fwhm_results = []

for rank0, rank1 in zip(chosen_ranks_0based, chosen_ranks_1based):
    # index into the original brightness order
    idx = all_top_indices[rank0]
    centroid = centroids[idx]

    fwhm_x_pix, fwhm_y_pix, fwhm_x_phys, fwhm_y_phys = spot_fwhm_from_centroid(
        img_gray,
        centroid,
        half_size=half_size,
        pixel_scale=pixel_scale,
    )

    chosen_fwhm_results.append({
        "brightness_rank_1based": int(rank1),
        "centroid_index": int(idx),
        "label_id": int(idx + 1),
        "centroid_y": float(centroid[0]),
        "centroid_x": float(centroid[1]),
        "fwhm_x_pix": float(fwhm_x_pix),
        "fwhm_y_pix": float(fwhm_y_pix),
        "fwhm_x_phys": float(fwhm_x_phys),
        "fwhm_y_phys": float(fwhm_y_phys),
    })

results_ratio = []

for r in chosen_fwhm_results:
    centroid_y = r["centroid_y"]
    dist_signed = centroid_y - y_line
    dist_abs = abs(dist_signed)

    fwhm_x = r["fwhm_x_pix"]*0.966
    fwhm_y = r["fwhm_y_pix"]*3.947

    # use the larger of the two FWHMs
    fwhm_used = max(fwhm_x, fwhm_y)
    ratio = dist_abs / fwhm_used

    results_ratio.append({
        "rank": r["brightness_rank_1based"],
        "centroid_y": centroid_y,
        "distance_signed": dist_signed,
        "distance_abs": dist_abs,
        "fwhm_x_pix": fwhm_x,
        "fwhm_y_pix": fwhm_y,
        "fwhm_used_pix": fwhm_used,
        "ratio_dist_over_fwhm_used": ratio,
    })

    print(
        f"Rank {r['brightness_rank_1based']}: "
        f"centroid_y={centroid_y:.1f}, "
        f"distance to y={y_line} is {dist_signed:.2f} px (abs {dist_abs:.2f} px), "
        f"FWHM_x={fwhm_x:.2f} px, FWHM_y={fwhm_y:.2f} px, "
        f"used FWHM={fwhm_used:.2f} px, "
        f"dist / used FWHM = {ratio:.4f}"
    )

ratios = [r["ratio_dist_over_fwhm_used"] for r in results_ratio]
print("Mean dist / used FWHM =", np.mean(ratios))


scale_x = 0.966   # 1920 / 1988, for example
scale_y = 3.947   # 1200 / 304, for example

# Compute per-spot FWHM using the larger of the two scaled values
fwhm_used_all_pix = [
    max(r["fwhm_x_pix"] * scale_x,
        r["fwhm_y_pix"] * scale_y)
    for r in fwhm_results
    if np.isfinite(r["fwhm_x_pix"]) and np.isfinite(r["fwhm_y_pix"])
]

overall_fwhm_used_pix = np.mean(fwhm_used_all_pix) if len(fwhm_used_all_pix) > 0 else np.nan
print(f"Overall average FWHM using larger scaled axis (all spots, pixels): {overall_fwhm_used_pix:.3f}")

print(np.max(fwhm_used_all_pix))


f_mm = 60.0        # focal length
D_mm = 0.3         # subaperture size
lambda_mm = 0.000633
mm_per_pix = 0.00568

N_res = f_mm / D_mm
s_res_mm = N_res * lambda_mm
s_res_pix = s_res_mm / mm_per_pix

print("N_res =", N_res)
print("Size of one resolution element (mm):", s_res_mm)
print("Size of one resolution element (pixels):", s_res_pix)

scale_x = 0.966
scale_y = 3.947

FWHM_expected_pix = 22.3

fwhm_used_all_pix = [
    max(r["fwhm_x_pix"] * scale_x,
        r["fwhm_y_pix"] * scale_y)
    for r in fwhm_results
    if np.isfinite(r["fwhm_x_pix"]) and np.isfinite(r["fwhm_y_pix"])
]

overall_fwhm_used_pix = np.mean(fwhm_used_all_pix)
print(f"Overall average measured FWHM (pixels): {overall_fwhm_used_pix:.2f}")
print(f"Expected resolution element size (pixels): {FWHM_expected_pix:.2f}")
print(f"Measured / expected = {overall_fwhm_used_pix / FWHM_expected_pix:.3f}")


# ---------------- plots  ----------------

idx_real = top_indices[4]      # choose which brightness rank you want to inspect
centroid_real = centroids[idx_real]

subimg, x0_local, y0_local = extract_spot_region(
    img_gray,
    centroid_real[1],  # x
    centroid_real[0],  # y
    half_size=half_size,
)

# 1D profiles through center
profile_x = subimg[y0_local, :]
profile_y = subimg[:, x0_local]

x_coords = np.arange(profile_x.size)
y_coords = np.arange(profile_y.size)

# Gaussian fits
A_x, x0_fit_x, sigma_x, offset_x, fwhm_x_gauss = fit_gaussian_to_profile(x_coords, profile_x)
A_y, x0_fit_y, sigma_y, offset_y, fwhm_y_gauss = fit_gaussian_to_profile(y_coords, profile_y)

# evaluate fitted Gaussians for plotting
fit_profile_x = gaussian_1d(x_coords, A_x, x0_fit_x, sigma_x, offset_x)
fit_profile_y = gaussian_1d(y_coords, A_y, x0_fit_y, sigma_y, offset_y)

'''
# plot chosen ones
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_smooth, cmap="gray")
ax.axhline(y=y_line, color="cyan", linestyle="--", linewidth=1)
ax.set_axis_off()

for rank0, rank1 in zip(chosen_ranks, chosen_ranks_1based):
    idx = top_indices[rank0]
    cy, cx = centroids[idx]
    ax.plot(cx, cy, "r+", markersize=10, markeredgewidth=2)
    ax.text(cx + 3, cy + 3, str(rank1), color="yellow", fontsize=10)

plt.tight_layout()
plt.show()
'''

# X profile with FWHM line
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(subimg, cmap="gray")
plt.scatter([x0_local], [y0_local], c="r", s=20)
plt.title(f"Spot around centroid (rank index {idx_real})")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.plot(x_coords, profile_x, "k.-", label="data")
plt.plot(x_coords, fit_profile_x, "r-", label="Gaussian fit")

A_x, x0_fit_x, sigma_x, offset_x, fwhm_x_gauss
fwhm_x = fwhm_x_gauss
half_x = offset_x + A_x / 2.0
x_left = x0_fit_x - fwhm_x / 2.0
x_right = x0_fit_x + fwhm_x / 2.0

plt.axhline(half_x, color="b", linestyle="--", linewidth=1, label="half max")
plt.plot(
    [x_left, x_right],
    [half_x, half_x],
    color="b",
    linewidth=3,
    label=f"FWHM = {fwhm_x:.2f} px",
)
plt.title(f"Profile along x - FWHM_gauss={fwhm_x_gauss:.2f} px")
plt.legend()



# Y profile with FWHM line
plt.subplot(1, 3, 3)
plt.plot(y_coords, profile_y, "k.-", label="data")
plt.plot(y_coords, fit_profile_y, "r-", label="Gaussian fit")

A_y, x0_fit_y, sigma_y, offset_y, fwhm_y_gauss
fwhm_y = fwhm_y_gauss
half_y = offset_y + A_y / 2.0
y_left = x0_fit_y - fwhm_y / 2.0
y_right = x0_fit_y + fwhm_y / 2.0

plt.axhline(half_y, color="b", linestyle="--", linewidth=1, label="half max")

plt.plot(
    [y_left, y_right],
    [half_y, half_y],
    color="b",
    linewidth=3,
    label=f"FWHM = {fwhm_y:.2f} px",
)

plt.title(f"Profile along y - FWHM_gauss={fwhm_y_gauss:.2f} px")
plt.legend()
plt.tight_layout()
plt.show()

'''
num_spots_to_plot = 73   # how many spots to inspect
start_rank = skip_k      # start from skip_k, or 0 if you want to include the brightest

spot_indices_to_plot = top_indices[start_rank:start_rank + num_spots_to_plot]

for i, idx_real in enumerate(spot_indices_to_plot, start=1):
    centroid = centroids[idx_real]
    cy, cx = centroid

    # extract subimage and profiles
    subimg, x0_local, y0_local = extract_spot_region(
        img_gray,
        cx,
        cy,
        half_size=half_size,
    )

    profile_x = subimg[y0_local, :]
    profile_y = subimg[:, x0_local]

    x_coords = np.arange(profile_x.size)
    y_coords = np.arange(profile_y.size)

    # fit Gaussians
    A_x, x0_fit_x, sigma_x, offset_x, fwhm_x_gauss = fit_gaussian_to_profile(x_coords, profile_x)
    A_y, x0_fit_y, sigma_y, offset_y, fwhm_y_gauss = fit_gaussian_to_profile(y_coords, profile_y)

    # generate fitted profiles (if fit failed, these may be nan)
    fit_profile_x = gaussian_1d(x_coords, A_x, x0_fit_x, sigma_x, offset_x)
    fit_profile_y = gaussian_1d(y_coords, A_y, x0_fit_y, sigma_y, offset_y)

    plt.figure(figsize=(12, 4))

    # 1) Subimage
    plt.subplot(1, 3, 1)
    plt.imshow(subimg, cmap="gray")
    plt.scatter([x0_local], [y0_local], c="r", s=20)
    plt.title(f"Spot around centroid (index {idx_real})")
    plt.axis("off")

    # 2) X profile with FWHM line
    plt.subplot(1, 3, 2)
    plt.plot(x_coords, profile_x, "k.-", label="data")
    if np.isfinite(fwhm_x_gauss):
        plt.plot(x_coords, fit_profile_x, "r-", label="Gaussian fit")

        fwhm_x = fwhm_x_gauss
        half_x = offset_x + A_x / 2.0
        x_left = x0_fit_x - fwhm_x / 2.0
        x_right = x0_fit_x + fwhm_x / 2.0

        plt.axhline(half_x, color="b", linestyle="--", linewidth=1, label="half max")
        plt.plot(
            [x_left, x_right],
            [half_x, half_x],
            color="b",
            linewidth=3,
            label=f"FWHM = {fwhm_x:.2f} px",
        )
        plt.title(f"Profile x - FWHM_gauss={fwhm_x_gauss:.2f} px")
    else:
        plt.title("Profile x - fit failed")

    plt.legend()

    # 3) Y profile with FWHM line
    plt.subplot(1, 3, 3)
    plt.plot(y_coords, profile_y, "k.-", label="data")
    if np.isfinite(fwhm_y_gauss):
        plt.plot(y_coords, fit_profile_y, "r-", label="Gaussian fit")

        fwhm_y = fwhm_y_gauss
        half_y = offset_y + A_y / 2.0
        y_left = x0_fit_y - fwhm_y / 2.0
        y_right = x0_fit_y + fwhm_y / 2.0

        plt.axhline(half_y, color="b", linestyle="--", linewidth=1, label="half max")
        plt.plot(
            [y_left, y_right],
            [half_y, half_y],
            color="b",
            linewidth=3,
            label=f"FWHM = {fwhm_y:.2f} px",
        )

        plt.title(f"Profile y - FWHM_gauss={fwhm_y_gauss:.2f} px")
    else:
        plt.title("Profile y - fit failed")

    plt.legend()
    plt.tight_layout()
    plt.show()
'''

'''
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(subimg, cmap="gray")
plt.scatter([x0_local], [y0_local], c="r", s=20)
plt.title(f"Spot around centroid (rank index {idx_real})")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.plot(x_coords, profile_x, "k.-", label="data")
plt.plot(x_coords, fit_profile_x, "r-", label="Gaussian fit")
plt.title(f"Profile along x - FWHM_gauss={fwhm_x_gauss:.2f} px")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(y_coords, profile_y, "k.-", label="data")
plt.plot(y_coords, fit_profile_y, "r-", label="Gaussian fit")
plt.title(f"Profile along y - FWHM_gauss={fwhm_y_gauss:.2f} px")
plt.legend()

plt.tight_layout()
plt.show()
'''
