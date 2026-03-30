import numpy as np
import scipy
import scipy.io as sio
from os.path import dirname, join as pjoin
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider
from matplotlib.patches import Circle
import matplotlib.cm as cm
from matplotlib.lines import Line2D


#load mat
script_dir = dirname(__file__)
mat_fname = pjoin(script_dir, 'tiptilt_only.mat')
mat_contents = sio.loadmat(mat_fname, spmatrix=False)

mask_bit = mat_contents['difftiptilt_SHWFS_mask']
height_um = mat_contents['difftiptilt_SHWFS_max_height_in_microns'][0, 0]
step_um = mat_contents['difftiptilt_SHWFS_step_size_in_microns'][0, 0]
#pixel_pitch_um = mat_contents['pixel_pitch_in_microns'][0, 0]

pixel_size_um = 300/59 # d_1 = 300 microns over block width = 59 pixels


def convert_bits_microns(mask_bit, height_um):
    mask = (mask_bit / np.max(mask_bit)) * height_um
    return mask
mask = 2*(convert_bits_microns(mask_bit, height_um))
'''
plt.figure(figsize=(8,6))
plt.imshow(mask, cmap='gray')
plt.colorbar(label='height (microns)')
plt.title('tip tilt mask in microns')
plt.show()
'''
def make_blocks(mask_array, block_size, pixel_size_um):
    height, width = mask_array.shape
    blocks = []
    block_id = 1

    num_rows = height // block_size
    num_cols = width // block_size

    for row in range(num_rows):
        for col in range(num_cols):
            y_idx = row * block_size
            x_idx = col * block_size

            block = mask_array[y_idx:y_idx+block_size, x_idx:x_idx+block_size] 

            #coords
            x1 = x_idx * pixel_size_um - 0.5 * pixel_size_um
            x2 = x1 + block_size * pixel_size_um
            y1 = y_idx * pixel_size_um - 0.5 * pixel_size_um
            y2 = y1 + block_size * pixel_size_um

            blocks.append({
                "id": block_id,
                "grid_x": col,
                "grid_y": row,
                "pixel_x": x_idx,
                "pixel_y": y_idx,
                "micron_coords": {
                    "top_left": (x1, y1),
                    "bottom_right": (x2, y2)
                },
                "block": block,
                "min": np.min(block),
                "max": np.max(block)
            })

            block_id += 1

    return blocks

blocks = make_blocks(mask, block_size=59, pixel_size_um=pixel_size_um)

'''
for b in blocks[0:528]:
    #print(f"Block {b['id']}: Grid=({b['grid_x']}, {b['grid_y']}), Min={b['min']:.2f}, Max={b['max']:.2f}")
    print(f"Block {b['id']}: Min={b['min']:.2f}, Max={b['max']:.2f}")

'''
def plot_grid(mask_array, blocks, show_block_ids=True):
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(mask_array, cmap='viridis', origin='upper')

    for b in blocks:
        x = b['pixel_x'] - 0.5
        y = b['pixel_y'] - 0.5
        rect = patches.Rectangle(
            (x, y),
            width=59,
            height=59,
            linewidth=0.5,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        if show_block_ids:
            ax.text(
                x + 29.5, y + 29.5,
                str(b['id']),
                color='white',
                fontsize=6,
                ha='center', va='center'
            )
    '''
    ax.set_title("grid over tip tilt mask")
    ax.set_xlabel("x pix")
    ax.set_ylabel("y pix")
    fig.colorbar(im, label = 'height (microns)')
    plt.grid(False)
    plt.show()
plot_grid(mask, blocks)
'''

#block numbers to remove ** NEW MIDDLE BLOCK is 213 
remove_block_numbers = [
    1, 2, 3, 4, 5, 6, 7, 8,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    66, 67, 68, 69, 70, 71, 72,
    90, 91, 92, 93, 94,
    114, 115, 116, 117,
    138, 139,
    161,162, 
    346,
    369,
    391, 392, 393,
    414, 415, 416,
    436, 437, 438, 439, 440,
    458, 459, 460, 461, 462, 463, 464,
    480, 481, 482, 483, 484, 485, 486, 487, 488,
    502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513,
    523, 524, 525, 526, 527, 528, 529
    ]
remove_indices = set([num - 1 for num in remove_block_numbers])
# update list 
filtered_blocks = [block for i, block in enumerate(blocks) if i not in remove_indices]
blocks = filtered_blocks

def calc_ref_angle(d_1, blocks):
    reflection_angles = []

    for b in blocks:
        d_2 = b['max'] - b['min']  # microns
        theta_rad = np.arctan(d_2 / d_1)  
        theta_deg = np.degrees(theta_rad)

        reflection_angles.append({
            "block_id": b["id"],
            "theta_rad": theta_rad,
            "theta_deg": theta_deg
        })

    return reflection_angles

angles = calc_ref_angle(d_1=59*pixel_size_um, blocks=blocks)

'''
for angle in angles[:529]:
    print(f"Block {angle['block_id']}: θ = {angle['theta_deg']:.4f} degrees")
print(max(angles, key=lambda a: a['theta_deg']))
'''

# cross sec plots
center_y = mask.shape[0] // 2
center_x = mask.shape[1] // 2

y_slice = mask[center_y, :]    
x_slice = mask[:, center_x]   
'''
plt.figure(figsize=(10,5))
plt.plot(x_slice, label='central x slice (column)', color='orange')
plt.title('central x cross section')
plt.grid(True)
plt.show()
plt.figure(figsize=(10,5))
plt.plot(y_slice, label='central y slice (row)')
plt.title('central y cross section')
plt.grid(True)
plt.show()
'''

def center_to_block(blocks, middle_block_id=213, pixel_size_um=pixel_size_um):
    centers_pix = []
    centers_um = []
    for b in blocks:
        center_x_pix = b['pixel_x'] - 0.5 + 59/2
        center_y_pix = b['pixel_y'] - 0.5 + 59/2
        centers_pix.append([center_x_pix, center_y_pix])
        centers_um.append([center_x_pix * pixel_size_um, center_y_pix * pixel_size_um])

    centers_pix = np.array(centers_pix)
    centers_um = np.array(centers_um)
    center_pos_um = centers_um[middle_block_id - 1]

    c = centers_um - center_pos_um
    c_mag = np.linalg.norm(c, axis=1)
    return centers_um, centers_pix, c, c_mag

centers_um, centers_pix, c, c_mag = center_to_block(blocks, middle_block_id=213, pixel_size_um=pixel_size_um)
#print('vector to block 213:', c[212])

def fit_plane(block, x_grid, y_grid):
    """
    Fit a plane z = ax + by + c to a 2D array (block), using global x_grid, y_grid.
    Returns gradient vector (a, b, 1) in physical units.
    """
    nrows, ncols = block.shape
    X = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.ones(nrows * ncols)))
    Z = block.ravel()
    coeffs, _, _, _ = np.linalg.lstsq(X, Z, rcond=None)
    a, b, c = coeffs
    grad = np.array([-a, -b, 1])
    grad_unit = grad / np.linalg.norm(grad)
    return grad, grad_unit

n = []
n_unit = []

block_size = 59

for i, b in enumerate(blocks):
    x0 = b['pixel_x']-0.5
    y0 = b['pixel_y']-0.5

    x_indices = np.arange(x0, x0 + block_size)
    y_indices = np.arange(y0, y0 + block_size)
    x_grid_indices, y_grid_indices = np.meshgrid(x_indices, y_indices)

    x_grid = x_grid_indices * pixel_size_um
    y_grid = y_grid_indices * pixel_size_um

    grad, grad_unit = fit_plane(b['block'], x_grid, y_grid)
    n.append(grad)
    n_unit.append(grad_unit)

'''
for i, grad in enumerate(n):
    print(f"Block {i+1}: grad = {grad}, unit = {n_unit[i]}")

'''

ray_spots = []
z_detector = 23000 # microns
z_lens = 3500  # microns
f_lens = 15.4  # mm
f_lens_um = f_lens * 1000
z_focus = z_lens + f_lens_um # microns

def get_beam_radius(z, r_0=150, z_lens=z_lens, z_focus=z_focus):
    if z < z_lens:
        return r_0
    else:
        # contract starting from the lens
        return r_0 * abs(1 - (z - z_lens) / (z_focus - z_lens))

def propagate_ray(c, n, block_idx, z=z_detector):
    v_x = c[block_idx][0] + n[block_idx][0] * z
    v_y = c[block_idx][1] + n[block_idx][1] * z
    v_z = z
    v = [v_x/1000, v_y/1000, v_z/1000]  # convert to mm
    return v

for block_idx in range(len(blocks)):
    v = propagate_ray(c, n, block_idx, z=z_detector)
    #print(f"Block {block_idx +1}: v = {np.float64(v)}")
    ray_spots.append({
        'x_det': v[0],
        'y_det': v[1],
        'block_id': block_idx + 1
    })

def outermost_rays(c, n, blocks, z, r_0=150, z_focus=z_focus):
    x_centers = []
    for block_idx in blocks:
        v = propagate_ray(c, n, block_idx, z)
        x_centers.append(v[0])
    r_z = get_beam_radius(z, r_0, z_lens, z_focus) / 1000  # convert microns to mm
    min_edge = min(x - r_z for x in x_centers)
    max_edge = max(x + r_z for x in x_centers)
    width = max_edge - min_edge
    return width, min_edge, max_edge

blocks = list(range(len(c)))

def mask_tilt(c, n, blocks, z_range, z_focus=z_focus, r_0=150, half_mask_width=3.45):
    results = []
    for z in z_range:
        width, min_edge, max_edge = outermost_rays(c, n, blocks, z, r_0, z_focus)
        x = width - ((width/2) - half_mask_width)
        alpha_rad = np.arctan((z/1000)/x)
        alpha = np.degrees(alpha_rad)
        center_x = (min_edge + max_edge) / 2
        center_z = z / 1000
        perp_dx = np.cos(alpha_rad)
        perp_dz = -np.sin(alpha_rad)
        x0 = center_x - (width/2) * perp_dx
        z0 = center_z - (width/2) * perp_dz
        theta_rad = np.arctan(z0/(np.abs(x0)+3.45))
        theta = np.degrees(theta_rad)
        beta = 90 - theta

        results.append({
            'z': z,
            'reflection_width_mm': width,
            'min_edge_mm': min_edge,
            'max_edge_mm': max_edge,
            'alpha': alpha,
            'beta': beta,
            'theta': theta,
            'x0': x0,
            'z0': z0
        })
    return results
z_range = np.linspace(0, 65000, 201)
results = mask_tilt(c, n, blocks, z_range)

'''
print(f"{'z (μm)':>12} | {'beam width (mm)':>18} | {'min ray (mm)':>18} | {'max ray (mm)':>18} | {'alpha (deg)':>18} | {'theta (deg)':>18} | {'beta (deg)':>18}")
print('-' * 150)
for row in results:
    print(f"{row['z']:8.0f} μm | {row['reflection_width_mm']:16.4f} mm | {row['min_edge_mm']:15.4f} mm| {row['max_edge_mm']:15.4f} mm | {row['alpha']:16.4f} deg | {row['theta']:16.4f} deg | {row['beta']:16.4f} deg")
'''

def get_initial_ray(c, n, block_idx):
    # returns initial ray position (microns) and direction
    x0 = c[block_idx][0]
    y0 = c[block_idx][1]
    theta_x = n[block_idx][0]
    theta_y = n[block_idx][1]
    return x0, y0, theta_x, theta_y

def propagate_ray_with_lens(c, n, block_idx, z, z_lens, f_lens):
    x0, y0, theta_x, theta_y = get_initial_ray(c, n, block_idx)
    if z < z_lens:
        x = x0 + theta_x * z
        y = y0 + theta_y * z
        return (x, y)
    else:
        # propagate to lens
        x_lens = x0 + theta_x * z_lens
        y_lens = y0 + theta_y * z_lens
        # apply lens transformation to angles
        theta_x_new = theta_x - x_lens / f_lens
        theta_y_new = theta_y - y_lens / f_lens
        # propagate from lens to current z
        dz = z - z_lens
        x = x_lens + theta_x_new * dz
        y = y_lens + theta_y_new * dz
        return (x, y)

# --------------------------------- PLOTTING & MOV -----------------------------------
# f number vs z
'''
z_vals = [row['z'] for row in results]  # z in microns
width_vals = [row['reflection_width_mm'] for row in results]  # D in mm

z_over_D = [((z/1000)) / D if D != 0 else float('nan') for z, D in zip(z_vals, width_vals)] # convert z to mm and subtract 2.7 mm for distance from spot plane
z_mm = [z/1000 for z in z_vals]
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(z_mm, z_over_D, color='mediumpurple', label='tip/tilt mask F#(z)')
ax.set_xlabel('z (mm)')
ax.set_ylabel('F# = f/D')
ax.axhline(y=1, color='orangered', linestyle='--', label='F# = 1')

plt.title('Tip/Tilt Mask F number vs z (distance from mask)')
ax.grid(True)
ax.legend(loc='lower right')
ax.set_ylim(-0.5,1.3)

plt.tight_layout()
plt.show()
'''
# lens prop movie
'''
z_start = 0
z_end = 33000 # microns
num_frames = 300 
z_vals = np.linspace(z_start, z_end, num_frames)

num_blocks = len(blocks)
block_ids = np.arange(1, num_blocks + 1)
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=1, vmax=num_blocks)

fig, ax = plt.subplots(figsize=(14, 8))
scat = ax.scatter([], [], c=[], s=10, cmap=cmap, norm=norm)

#  circles (one per ray spot)
beam_circles = []
for _ in range(num_blocks):
    circle = Circle((0, 0), 0, fill=False, edgecolor='indigo', linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    beam_circles.append(circle)

def init(): 
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True)
    ax.set_aspect('equal', 'box')	
    ax.set_xlim(-10, 10)  
    ax.set_ylim(-5, 5)

    # reset circles
    for circle in beam_circles:
        circle.center = (0, 0)
        circle.set_radius(0)
    return (scat, *beam_circles)

def update(frame):
    z = z_vals[frame]
    ray_spots = []
    for block_idx in range(num_blocks):
        x, y = propagate_ray_with_lens(c, n, block_idx, z, z_lens, f_lens_um)
        ray_spots.append({
            'x_det': x / 1000,  # convert to mm
            'y_det': y / 1000,
            'block_id': block_idx + 1
        })
    x = [r['x_det'] for r in ray_spots]
    y = [r['y_det'] for r in ray_spots]
    ids = [r['block_id'] for r in ray_spots]
    scat.set_offsets(np.column_stack([x, y]))
    scat.set_array(np.array(ids))
 
    # update circles
    r_z = get_beam_radius(z, r_0=150, z_lens=z_lens, z_focus=z_focus) / 1000  # mm
    for circle, xi, yi in zip(beam_circles, x, y):
        circle.center = (xi, yi)
        circle.set_radius(r_z)
    ax.set_title(f"fastrSHWFS Tip Tilt Mask Spots & Beam Size at z = {z/1000:.0f} mm")
    return (scat, *beam_circles)

ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=50)
cbar = plt.colorbar(scat, ax=ax, orientation='vertical', shrink=0.8)
cbar.set_label('Block ID')

z_lens_mm = z_lens / 1000
legend_label = f"Lens: Focal length = {f_lens} mm at z = {z_lens_mm:.1f} mm"
custom_legend = [Line2D([0], [0], color='black', lw=2)]

ax.legend(custom_legend, [legend_label], loc='upper right')
plt.show()

plt.show()
#ani.save('tiptilt_focus_lens.gif', writer='pillow', fps=20)
'''

# mask tilt angle plot
'''
highlight_z = 15000  # μm
highlight_tol = 20  # mm tolerance for floating point

fig, ax = plt.subplots(figsize=(8, 8))
cmap = cm.viridis
norm = plt.Normalize(min([r['theta'] for r in results]), max([r['theta'] for r in results]))

line_length = 30  # mm

for entry in results:
    theta = entry['theta']
    color = cmap(norm(theta))
    z_um = entry['z']
    z_mm = z_um / 1000

    # if this is the z to highlight
    is_highlight = abs(z_um - highlight_z) < highlight_tol

    # Incoming theta 
    theta_rad = np.deg2rad(theta)
    x_start = 3.45 - line_length * np.cos(theta_rad)
    z_start = 0 + line_length * np.sin(theta_rad)
    if is_highlight:
        ax.plot([x_start, 3.45], [z_start, 0], color='red', linewidth=2, label=f'incoming angle {theta:.2f}°' if f'incoming angle {theta:.2f}°' not in ax.get_legend_handles_labels()[1] else "")
        ax.plot([x_start-6.9, -3.45], [z_start, 0], color='red', linewidth=2)
    else:
        continue       
        ax.plot([x_start, 3.45], [z_start, 0], color=color, linewidth=1)

    # beam width 
    min_edge = entry['min_edge_mm']
    max_edge = entry['max_edge_mm']
    center_x = (min_edge + max_edge) / 2
    center_z = z_mm
    width = entry['reflection_width_mm']
    alpha_rad = np.deg2rad(entry['alpha'])
    perp_dx = np.cos(alpha_rad)
    perp_dz = -np.sin(alpha_rad)
    x0 = center_x - (width/2) * perp_dx
    z0 = center_z - (width/2) * perp_dz
    x1 = center_x + (width/2) * perp_dx
    z1 = center_z + (width/2) * perp_dz
    if is_highlight:
        ax.plot([x0, x1], [z0, z1], color='orange', linewidth=2, label='beam width' if 'beam width' not in ax.get_legend_handles_labels()[1] else "")
    else:
        continue
        ax.plot([x0, x1], [z0, z1], color='cornflowerblue', linewidth=1)

# mask at z=0
ax.plot([-3.45, 3.45], [0, 0], color='k', linewidth=3, label='mask')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('theta (deg)')

ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_ylim(-1, 25)
ax.set_xlim(-15, 15)
ax.set_title('tip tilt beam width and incoming ray angle (mask tilt)')
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.legend()
plt.show()
'''

# lin plot for beta and beam width
'''
z_vals = [row['z'] for row in results]
alpha_vals = [row['alpha'] for row in results]
beta_vals = [row['beta'] for row in results]
width_vals = [row['reflection_width_mm'] for row in results]

fig, ax1 = plt.subplots(figsize=(8,5))

# angle vs z on primary y-axis
ax1.plot(z_vals, beta_vals, color='mediumpurple', label='beta (deg)')
#ax1.plot(z_vals, alpha_vals, color='dodgerblue', label='alpha (deg)')
ax1.set_ylim(0,100)
ax1.set_xlabel('z (μm)')
ax1.set_ylabel('degrees')
ax1.tick_params(axis='y')

# secondary y-axis for width
ax2 = ax1.twinx()
ax2.plot(z_vals, width_vals, color='mediumseagreen', label='width (mm)')
ax2.set_ylabel('beam size (mm)')
ax2.set_ylim(5,35)
ax2.tick_params(axis='y')
ax2.axhline(y=25.4, color='orangered', linestyle='--', linewidth=1.5, label='1 in')
ax2.axhline(y=12.7, color='orange', linestyle='--', linewidth=1.5, label='0.5 in')

plt.title('tip tilt mask angle and beam width vs z')
ax1.grid(True)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
'''

# beams at z slider w lens
'''
def beams_at_z(c, n, blocks, r_0=150, z_focus=z_focus, z_max=20000):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)

    z_init = 0 # can set to a val to get a still at some z
    x = []
    y = []
    for block_idx in blocks:
        #v = propagate_ray(c, n, block_idx, z_init)
        v = propagate_ray_with_lens(c, n, block_idx, z_init, z_lens, f_lens_um)
        x.append(v[0]/1000)
        y.append(v[1]/1000)

    scatter = ax.scatter(x, y, c='royalblue', s=5)

    r_z = get_beam_radius(z_init, r_0, z_lens, z_focus) / 1000  # mm
    circles = [plt.Circle((xi, yi), r_z, fill=False, edgecolor='lightsteelblue', linestyle='--') for xi, yi in zip(x, y)]
    for circle in circles:
        ax.add_patch(circle)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"tip tilt ray spots and beam size at z = {z_init/1000:.3f} mm")
    ax.axis('equal')
    ax.grid(True)
    #ax.legend()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # slider
    ax_zlens = plt.axes([0.15, 0.09, 0.7, 0.03])  # (left, bottom, width, height)
    slider_lens = Slider(ax_zlens, 'Lens z (microns)', 0, z_max, valinit=z_lens, valstep=1)

    ax_z = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_z, 'z (microns)', 0, z_max, valinit=0, valstep=1)

    def update(val):
        z = slider.val
        z_lens_val = slider_lens.val  # get current lens position from slider
        z_focus_val = z_lens_val + f_lens_um
        x = []
        y = []
        for block_idx in blocks:
            #v = propagate_ray(c, n, block_idx, z)
            v = propagate_ray_with_lens(c, n, block_idx, z, z_lens_val, f_lens_um)
            x.append(v[0] / 1000) # mm
            y.append(v[1] / 1000)
        scatter.set_offsets(np.c_[x, y])

        r_z = get_beam_radius(z, r_0, z_lens=z_lens_val, z_focus=z_focus_val) / 1000  # mm
        for circle, xi, yi in zip(circles, x, y):
            circle.center = (xi, yi)
            circle.set_radius(r_z)
        
        ax.set_title(f"tip tilt ray spots and beam size at z = {z/1000:.1f} mm")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    slider_lens.on_changed(update)
    plt.show()

beams_at_z(c, n, blocks, r_0=150, z_focus=z_focus, z_max=45000)
'''
#og movie no lens
'''
z_start = 0
z_end = 10000 #microns
num_frames = 180 
z_vals = np.linspace(z_start, z_end, num_frames)

num_blocks = len(blocks)
block_ids = np.arange(1, num_blocks + 1)
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=1, vmax=num_blocks)

fig, ax = plt.subplots(figsize=(8, 8))
scat = ax.scatter([], [], c=[], s=40, cmap=cmap, norm=norm)

def init():
    range = 10 # mm
    ax.set_xlim(-range, range)  
    ax.set_ylim(-range, range) 
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True)
    ax.axis('equal')
    return scat,

def update(frame):
    z = z_vals[frame]
    ray_spots = []
    for block_idx in range(num_blocks):
        v = propagate_ray(c, n, block_idx, z)
        ray_spots.append({
            'x_det': v[0],
            'y_det': v[1],
            'block_id': block_idx + 1
        })
    x = [r['x_det'] for r in ray_spots]
    y = [r['y_det'] for r in ray_spots]
    ids = [r['block_id'] for r in ray_spots]
    scat.set_offsets(np.column_stack([x, y]))
    scat.set_array(np.array(ids))
    ax.set_title(f"tip tilt ray spots at z = {z/1000:.0f} mm")
    return scat,

ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=50)
cbar = plt.colorbar(scat, ax=ax, orientation='vertical')
cbar.set_label('Block ID')
plt.show()

#ani.save('tiptilt_spots.gif', writer='pillow', fps=20)
'''

# movie w beamsize no lens
'''
z_start = 0
z_end = 15000 # microns
num_frames = 180 
z_vals = np.linspace(z_start, z_end, num_frames)

num_blocks = len(blocks)
block_ids = np.arange(1, num_blocks + 1)
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=1, vmax=num_blocks)

fig, ax = plt.subplots(figsize=(14, 8))
scat = ax.scatter([], [], c=[], s=8, cmap=cmap, norm=norm)

# Prepare circles (one per ray spot)
beam_circles = []
for _ in range(num_blocks):
    circle = Circle((0, 0), 0, fill=False, edgecolor='indigo', linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    beam_circles.append(circle)

def init(): 
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True)
    ax.set_aspect('equal', 'box')	
    ax.set_xlim(-10, 10)  
    ax.set_ylim(-5, 5)

    # reset circles
    for circle in beam_circles:
        circle.center = (0, 0)
        circle.set_radius(0)
    return (scat, *beam_circles)

def update(frame):
    z = z_vals[frame]
    ray_spots = []
    for block_idx in range(num_blocks):
        v = propagate_ray(c, n, block_idx, z)
        ray_spots.append({
            'x_det': v[0],
            'y_det': v[1],
            'block_id': block_idx + 1
        })
    x = [r['x_det'] for r in ray_spots]
    y = [r['y_det'] for r in ray_spots]
    ids = [r['block_id'] for r in ray_spots]
    scat.set_offsets(np.column_stack([x, y]))
    scat.set_array(np.array(ids))
 
    # update circles
    r_z = get_beam_radius(z, r_0=150) / 1000  # mm
    for circle, xi, yi in zip(beam_circles, x, y):
        circle.center = (xi, yi)
        circle.set_radius(r_z)
    ax.set_title(f"tip tilt ray spots at z = {z/1000:.0f} mm")
    return (scat, *beam_circles)

ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=50)
cbar = plt.colorbar(scat, ax=ax, orientation='vertical', shrink=0.8)
cbar.set_label('Block ID')
plt.show()
'''

