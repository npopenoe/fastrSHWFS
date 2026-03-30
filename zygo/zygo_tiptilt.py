import scipy.io as sio
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt 
from scipy.ndimage import rotate
import numpy as np
from PIL import Image
from matplotlib.patches import Circle
from scipy.signal import correlate2d
import numpy as np
import matplotlib.pyplot as plt

#load mat
script_dir = dirname(__file__)
mat_fname1 = pjoin(script_dir, 'tiptilt_mask1.mat')
mat_fname2 = pjoin(script_dir, 'tiptilt_mask1.mat')

mat_contents1 = sio.loadmat(mat_fname1, spmatrix=False)
mat_contents2 = sio.loadmat(mat_fname2, spmatrix=False)

# print(mat_contents1.keys()) --> dict_keys(['__header__', '__version__', '__globals__', 'header', 'phase'])

phase1 = mat_contents1['phase']
phase2 = mat_contents2['phase']


# ---------------- Mask Parameters ----------------
subap_size = 300  # microns
block_size = 59  # pixels
subap_num = 5  # number of subaps in zygo scan


# ------------------------------ DESIGN ------------------------------
script_dir = dirname(__file__)
mat_fname = pjoin(script_dir, 'tiptilt_only.mat')
mat_contents = sio.loadmat(mat_fname, spmatrix=False)
mask_bit = mat_contents['difftiptilt_SHWFS_mask']
height_um = mat_contents['difftiptilt_SHWFS_max_height_in_microns'][0, 0]
step_um = mat_contents['difftiptilt_SHWFS_step_size_in_microns'][0, 0]
pixel_size_um = subap_size / block_size # microns per pixel


def convert_bits_microns(mask_bit, height_um):
    mask = (mask_bit / np.max(mask_bit)) * height_um
    return mask
mask = 2*(convert_bits_microns(mask_bit, height_um))

def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = 613 - (cropx // 2)
    starty = 1184 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]

cropped_mask = crop_center(mask, 295, 295)

mask_img = Image.fromarray(cropped_mask)
mask_resized = mask_img.resize((1024, 1024), resample=Image.NEAREST)
tt = np.array(mask_resized)
tt_centered = tt - np.mean(tt)

# ------------------------------ ZYGO ------------------------------

def rotate_image(image, angle_deg):
    rotated_image = rotate(image, angle_deg, reshape=False, order=1)
    return rotated_image

angle_deg = 268.7 # determined from visual inspection
rotated_zygo = rotate_image(phase2, angle_deg)

def crop_zygo(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)+1 #offsets are done visually by checking perimeter values at subap edges
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]

cropped_zygo = crop_zygo(rotated_zygo, 905,905)

zygo_img = Image.fromarray(cropped_zygo)
zygo_resized = zygo_img.resize((1024, 1024) ,resample=Image.NEAREST)
zygo = np.array(zygo_resized)
zygo_centered = zygo - np.mean(zygo)


# ------------------------------ SUBTRACTION ------------------------------

subtracted = tt_centered - zygo_centered
subtracted_nm = subtracted * 1000

# cross sec plots
center_y = zygo_centered.shape[0] // 2
center_x = zygo_centered.shape[1] // 3
y_slice = zygo_centered[center_y, :]    
x_slice = zygo_centered[:, center_x]

center_y_design = tt_centered.shape[0] // 2
center_x_design = tt_centered.shape[1] // 3
y_slice_design = tt_centered[center_y, :]    
x_slice_design = tt_centered[:, center_x]

center_y_sub = subtracted.shape[0] // 2
center_x_sub = subtracted.shape[1] // 3
y_slice_sub = subtracted[center_y, :]    
x_slice_sub = subtracted[:, center_x]

# choose a subap to do rms calc
ystart_pixel = 23 #pixels
yend_pixel = 221
xstart_pixel = 88
xend_pixel = 280
middle_of_subap_x = (x_slice_sub[xstart_pixel:xend_pixel+1]) * 1000  # nanometers
middle_of_subap_y = (y_slice_sub[ystart_pixel:yend_pixel+1]) * 1000

rms_whole_mask = np.sqrt(np.mean(subtracted_nm**2))
rms_middle_subap_x = np.sqrt(np.mean(middle_of_subap_x**2))
rms_middle_subap_y = np.sqrt(np.mean(middle_of_subap_y**2))

#print(f'RMS: {rms_whole_mask:.2f} nm')
print(f'RMS in x slice (pixels {xstart_pixel}-{xend_pixel}): {rms_middle_subap_x:.2f} nm')
print(f'RMS in y slice (pixels {ystart_pixel}-{yend_pixel}): {rms_middle_subap_y:.2f} nm')


# ------------------------------ plotting ------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

im0 = axes[0].imshow(tt_centered, cmap='gray')
axes[0].set_title('design of tt mask')
cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.set_label('height (microns)')

im1 = axes[1].imshow(zygo, cmap='gray')
axes[1].set_title('printed zygo mask')
cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_label('height (microns)')

im2 = axes[2].imshow(subtracted, cmap='bwr')
axes[2].set_title('difference')
cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
cbar2.set_label('design - print (microns)')

plt.tight_layout()
plt.show()
'''
plt.figure(figsize=(10,5))
plt.plot(x_slice_design, label='Design', color='mediumpurple')
plt.plot(x_slice, label='Zygo Scan', color='orange')
plt.plot(x_slice_sub, label='subtracted',color='black', alpha=0.3)
plt.legend()
#plt.xlim(480,700)
plt.ylim(-40,40)
plt.grid(True)
plt.title('Central X Slice Comparison')
plt.xlabel('Pixel')
plt.ylabel('Height (microns)')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(y_slice_design, label='Design', color='mediumpurple')
plt.plot(y_slice, label='Zygo Scan', color='orange')
plt.plot(y_slice_sub, label='subtracted',color='black', alpha=0.3)
plt.legend()
#plt.xlim(640,850)
#plt.ylim(-40,40)
plt.grid(True)
plt.title('Central Y Slice Comparison')
plt.xlabel('Pixel')
plt.ylabel('Height (microns)')
plt.show()
'''
'''
#get ticks of axes to line up with physical units of new image size 
img_width = block_size * subap_num
micron_width = img_width * pixel_size_um
d = micron_width / (1024 * pixel_size_um)

x_microns = (np.arange(len(x_slice_design)) * pixel_size_um)*d
y_microns = (np.arange(len(y_slice_design)) * pixel_size_um)*d

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
im0 = axes[0].plot(x_microns, x_slice_design, label='Design', color='mediumblue')
im0 = axes[0].plot(x_microns, x_slice, label='Zygo Scan', color='lightcoral')
im0 = axes[0].plot(x_microns, x_slice_sub, label='Subtracted',color='black', alpha=0.3)
axes[0].set_title('X Slice Comparison Design vs Scan')
axes[0].set_xlabel('Microns')
axes[0].set_ylabel('Height (microns)')
axes[0].set_ylim(-40,40)
axes[0].legend()
axes[0].grid(True)

im1 = axes[1].plot(y_microns, y_slice_design, label='Design', color='cornflowerblue')
im1 = axes[1].plot(y_microns,y_slice, label='Zygo Scan', color='lightcoral')
im1 = axes[1].plot(y_microns,y_slice_sub, label='Subtracted',color='black', alpha=0.3)
axes[1].set_title('Y Slice Comparison Design vs Scan')
axes[1].set_xlabel('Microns')
axes[1].set_ylabel('Height (microns)')
axes[1].set_ylim(-140,140)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()'''
from matplotlib.patches import Rectangle
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Crosshair for slice locations
for ax in axes:
    # vertical line for x slice at center_x
    ax.axvline(center_x, color='lime', linestyle='--', linewidth=1)
    # horizontal line for y slice at center_y
    ax.axhline(center_y, color='cyan', linestyle='--', linewidth=1)

# Rectangles showing RMS regions:
#  - X slice: vertical band around center_x, from y = ystart_pixel to y = yend_pixel
#  - Y slice: horizontal band around center_y, from x = xstart_pixel to x = xend_pixel

rect_kwargs_x = dict(edgecolor='yellow', facecolor='none', linewidth=1.5)
rect_kwargs_y = dict(edgecolor='magenta', facecolor='none', linewidth=1.5)

# thickness of the band around the slice direction when showing as a rectangle
half_band = 5

for ax in axes:
    # Region for RMS along X slice (varies in y, fixed at center_x)
    ax.add_patch(Rectangle(
        (center_x - half_band, ystart_pixel),     # (x, y)
        2 * half_band,                            # width in x
        yend_pixel - ystart_pixel,                # height in y
        **rect_kwargs_x
    ))

    # Region for RMS along Y slice (varies in x, fixed at center_y)
    ax.add_patch(Rectangle(
        (xstart_pixel, center_y - half_band),     # (x, y)
        xend_pixel - xstart_pixel,                # width in x
        2 * half_band,                            # height in y
        **rect_kwargs_y
    ))

plt.tight_layout()
plt.show()

# ------------------------------ CROSS SECTIONS WITH MARKERS ------------------------------

# get ticks of axes to line up with physical units of new image size 
img_width   = block_size * subap_num
micron_width = img_width * pixel_size_um
d = micron_width / (1024 * pixel_size_um)

x_microns = (np.arange(len(x_slice_design)) * pixel_size_um) * d
y_microns = (np.arange(len(y_slice_design)) * pixel_size_um) * d

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# X slice
axes[0].plot(x_microns, x_slice_design, label='Design', color='mediumblue')
axes[0].plot(x_microns, x_slice,        label='Zygo Scan', color='lightcoral')
axes[0].plot(x_microns, x_slice_sub,    label='Subtracted', color='black', alpha=0.3)

# Mark RMS region on X slice
axes[0].axvline(x_microns[xstart_pixel], color='red', linestyle='--', linewidth=1)
axes[0].axvline(x_microns[xend_pixel],   color='red', linestyle='--', linewidth=1)
axes[0].axvspan(x_microns[xstart_pixel], x_microns[xend_pixel],
                color='red', alpha=0.1, label='RMS region')

axes[0].set_title('X Slice Comparison Design vs Scan')
axes[0].set_xlabel('Microns')
axes[0].set_ylabel('Height (microns)')
axes[0].set_ylim(-140, 140)
axes[0].legend()
axes[0].grid(True)

# Y slice
axes[1].plot(y_microns, y_slice_design, label='Design', color='cornflowerblue')
axes[1].plot(y_microns, y_slice,        label='Zygo Scan', color='lightcoral')
axes[1].plot(y_microns, y_slice_sub,    label='Subtracted', color='black', alpha=0.3)

# Mark RMS region on Y slice
axes[1].axvline(y_microns[ystart_pixel], color='red', linestyle='--', linewidth=1)
axes[1].axvline(y_microns[yend_pixel],   color='red', linestyle='--', linewidth=1)
axes[1].axvspan(y_microns[ystart_pixel], y_microns[yend_pixel],
                color='red', alpha=0.1, label='RMS region')

axes[1].set_title('Y Slice Comparison Design vs Scan')
axes[1].set_xlabel('Microns')
axes[1].set_ylabel('Height (microns)')
axes[1].set_ylim(-140, 140)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()



px = np.arange(len(x_slice_sub))
py = np.arange(len(y_slice_sub))

plt.figure()
plt.plot(px, x_slice_sub)
plt.axvspan(xstart_pixel, xend_pixel, color='red', alpha=0.1)
plt.title('X slice in pixel coordinates')
plt.xlabel('Pixel index')
plt.ylabel('Height (microns)')
plt.show()

plt.figure()
plt.plot(py, y_slice_sub)
plt.axvspan(ystart_pixel, yend_pixel, color='red', alpha=0.1)
plt.title('Y slice in pixel coordinates')
plt.xlabel('Pixel index')
plt.ylabel('Height (microns)')
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter  # needs ffmpeg installed

# ------------------ assume zygo_centered already computed above ------------------

# Option: restrict to a subset of rows to speed up / focus on region
y_start = 0
y_end   = zygo_centered.shape[0] - 1   # last index
x_start = 0
x_end   = zygo_centered.shape[1] - 1   # last index
num_frames = zygo_centered.shape[1]    # one frame per row

# Micron axis for the cross section plot (reuse your scaling)
img_width   = block_size * subap_num
micron_width = img_width * pixel_size_um
d = micron_width / (1024 * pixel_size_um)
x_microns = (np.arange(zygo_centered.shape[1]) * pixel_size_um) * d

# ------------------ setup figure ------------------
fig, (ax_img, ax_cs) = plt.subplots(1, 2, figsize=(12, 5))

# 2D Zygo image
im = ax_img.imshow(zygo_centered, cmap='gray')
ax_img.set_title('Zygo scan with moving cross section')
ax_img.set_xlabel('X pixel')
ax_img.set_ylabel('Y pixel')
cbar = plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
cbar.set_label('Height (microns)')

# initial horizontal line
y0 = y_start
line_y = ax_img.axhline(y0, color='cyan', linewidth=1)
x0 = x_start
line_x = ax_img.axvline(x0, color='lime', linewidth=1)

# initial cross section
cs, = ax_cs.plot(x_microns, zygo_centered[y0, :], color='black')
ax_cs.set_title('Cross section through Zygo scan')
ax_cs.set_xlabel('Position (microns)')
ax_cs.set_ylabel('Height (microns)')
# optional: fix y-limits to see relative changes more clearly
# ax_cs.set_ylim(zygo_centered.min(), zygo_centered.max())
ax_cs.grid(True)

plt.tight_layout()

# ------------------ animation function ------------------

def update(frame):
    y_idx = y_start + frame
    if y_idx > y_end:
        y_idx = y_end

    # update horizontal line position
    line_y.set_ydata([y_idx, y_idx])

    # update cross section data
    cs.set_ydata(zygo_centered[y_idx, :])
    ax_cs.set_title(f'Cross section at row y = {y_idx}')

    return im, line_y, cs

'''
def update(frame):
    x_idx = x_start + frame
    if x_idx > x_end:
        x_idx = x_end

    # update horizontal line position
    line_x.set_xdata([x_idx, x_idx])

    # update cross section data
    cs.set_xdata(zygo_centered[x_idx, :])
    ax_cs.set_title(f'Cross section at row x = {x_idx}')
    return im, line_x, cs
'''
# ------------------ create animation ------------------
anim = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=50,   # ms between frames, adjust playback speed
    blit=False
)

plt.show()
'''
# ------------------ save to file ------------------
# Requires ffmpeg installed on your system
output_filename = 'zygo_cross_section_y_sweep.mp4'
writer = FFMpegWriter(fps=20, bitrate=1800)
anim.save(output_filename, writer=writer)

plt.close(fig)
print(f"Saved animation to {output_filename}")'''