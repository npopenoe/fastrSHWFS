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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from tt_angle_calc_copy import mask_tilt
from tt_angle_calc_copy import results as results_tt

# --------------- fastrSHWFS focus mask angle calculations by Natasha Popenoe ---------------
# ---------------  if you have questions about this code, contact popenoe1@llnl -------------
# last updated January 2026

script_dir = dirname(__file__)
mat_fname = pjoin(script_dir, 'focus.mat')
mat_contents = sio.loadmat(mat_fname, spmatrix=False)
#print(mat_contents.keys())
mask_bit = mat_contents['fastrSHWFS_mask']
height_um = mat_contents['fastrSHWFS_max_height_in_microns'][0, 0]
step_um = mat_contents['fastrSHWFS_step_size_in_microns'][0, 0]



# MASK PARAMETERS
subap_size = 300        # microns - subap / mirrorlet etc.
block_size = 59         # pixels
total_blocks = 529      # 23x23 grid
middle_block_id = 265   # center mirrorlet / optical axis
mask_width = 6.9        # mm
z_focus = 2722          # microns - focal plane of mask
r_0 = subap_size / 2    # radius of beam at mask
pixel_size_um = subap_size/block_size



def convert_bits_microns(mask_bit, height_um):
    '''
    Converts mask bits to microns using mat content keys
    '''
    mask = (mask_bit / np.max(mask_bit)) * height_um
    return mask
mask = 2*(convert_bits_microns(mask_bit, height_um)) # *2 property of reflected mask

# plot mask design
'''
pixels = mask.shape[1]
microns = pixels * pixel_size_um
print(microns)
plt.figure(figsize=(8,6))
plt.imshow(mask, cmap='gray')
plt.colorbar(label='height (microns)')
plt.xticks(ticks=np.linspace(0, pixels, 6), labels=[f"{int(x)}" for x in np.linspace(0, microns, 6)])
plt.yticks(ticks=np.linspace(0, pixels, 6), labels=[f"{int(y)}" for y in np.linspace(0, microns, 6)])
plt.xlabel('microns')
plt.ylabel('microns')
plt.title('Focus Mask Design')
plt.show()
'''

def make_blocks(mask_array, block_size, pixel_size_um):
    '''
    Divides the mask array into blocks of given size and calculates their pixel and micron coordinates.

    :param mask_array: mask (type: np.ndarray)
    :param block_size: block size in pixels (type: int)
    :param pixel_size_um: pixel size in microns (type: float)

    :return: focus_blocks: list of dicts with block info
    '''
    height, width = mask_array.shape
    focus_blocks = []
    block_id = 1

    num_rows = height // block_size
    num_cols = width // block_size

    for row in range(num_rows):
        for col in range(num_cols):
            y_idx = row * block_size
            x_idx = col * block_size

            focus_block = mask_array[y_idx:y_idx+block_size, x_idx:x_idx+block_size] #2D array per block

            #coords
            x1 = x_idx * pixel_size_um - 0.5 * pixel_size_um
            x2 = x1 + block_size * pixel_size_um
            y1 = y_idx * pixel_size_um - 0.5 * pixel_size_um
            y2 = y1 + block_size * pixel_size_um

            focus_blocks.append({
                "id": block_id,
                "grid_x": col,
                "grid_y": row,
                "pixel_x": x_idx,
                "pixel_y": y_idx,
                "micron_coords": {
                    "top_left": (x1, y1),
                    "bottom_right": (x2, y2)
                },
                "block": focus_block,
            })

            block_id += 1

    return focus_blocks

focus_blocks = make_blocks(mask, block_size=block_size, pixel_size_um=pixel_size_um)

# print each block's pixel and micron coordinates
'''
for b in focus_blocks:
    print(f"Block {b['id']}: pixel x = {b['pixel_x']}, pixel y = {b['pixel_y']}, micron coords ={b['micron_coords']}")
'''

def subtract_focus(focus_blocks):
    '''
    Subtracts the middle block from all focus blocks to remove focus component from mask
    
    :param focus_blocks: list of dicts with block info (type: list)
    :param middle_block_id: ID of the middle block to subtract (type: int)

    :returns blocks: list of dicts with focus subtracted block max and min (type: list)
    '''
    blocks = []
    for focus_block in focus_blocks:
        subtracted_block = focus_block["block"] - focus_blocks[middle_block_id - 1]["block"] # subtract focus (middle block)
        block = focus_block.copy()
        block["block"] = subtracted_block
        block["min"] = np.min(subtracted_block)
        block["max"] = np.max(subtracted_block)
        blocks.append(block)
    return blocks

blocks = subtract_focus(focus_blocks)

# prints min and max of each block (sanity check center block =0)
'''
for b in blocks:
    print(f"Block {b['id']}: Min={b['min']:.2f}, Max={b['max']:.2f}")
'''

def focus_sub_mask(blocks, block_size, mask_shape):
    '''
    Reconstructs the focus subtracted mask from 'blocks' without focus 
    
    :param blocks: list of dicts with focus subtracted block info (type: list)
    :param block_size: block size in pixels (type: int)
    :param mask_shape: shape of the original mask (type: tuple)

    returns: reconstructed focus subtracted mask (type: np.ndarray)
    '''
    reconstructed = np.zeros(mask_shape)
    for block in blocks:
        y = block["pixel_y"]
        x = block["pixel_x"]
        reconstructed[y:y+block_size, x:x+block_size] = block["block"]
    return reconstructed

focus_sub_mask = focus_sub_mask(blocks, block_size=block_size, mask_shape=mask.shape)

# plot focus subtracted mask (should look the same but is centered around 0)
'''
plt.figure(figsize=(8,6))
plt.imshow(focus_sub_mask, cmap='gray')
plt.colorbar(label='height (microns)')
plt.title('focus subtracted mask')
plt.show()
'''

def plot_grid(mask_array, blocks, show_block_ids=True):
    '''
    Plots the grid over the given mask array with optional block IDs to check it aligns with subaps 
    
    :param mask_array: mask array to plot (type: np.ndarray)
    :param blocks: list of dicts with block info (type: list)
    :param show_block_ids: whether to show block IDs on the plot (type: bool), helps identify blocks 
    '''
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(mask_array, cmap='viridis', origin='upper')

    for b in blocks:
        x = b['pixel_x'] - 0.5
        y = b['pixel_y'] - 0.5
        rect = patches.Rectangle(
            (x, y),
            width=block_size,
            height=block_size,
            linewidth=0.5,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        if show_block_ids:
            ax.text(
                x + (block_size/2), y + (block_size/2),
                str(b['id']),
                color='white',
                fontsize=6,
                ha='center', va='center'
            )
    #plots grid with block ids over mask
    '''
    ax.set_title("grid over removed focus mask")
    ax.set_xlabel("x pix")
    ax.set_ylabel("y pix")
    fig.colorbar(im, label='height (microns)')
    plt.grid(False)
    plt.show()
plot_grid(focus_sub_mask, blocks)
'''

#block numbers to remove 
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
# update list 'blocks'
filtered_blocks = [block for i, block in enumerate(blocks) if i not in remove_indices]
blocks = filtered_blocks

def calc_ref_angle(d_1, blocks):
    '''
    Calculates reflection angles for each block based on d_1 and block height differences.
    
    :param d_1: subaperture size (type: int)
    :param blocks: list of dicts with block info (type: list)

    returns: list of dicts with block ID and reflection angles in degrees (type: list)
    '''
    reflection_angles = []

    for b in blocks:
        d_2 = b['max'] - b['min']  # height subap microns
        theta_rad = np.arctan(d_2 / d_1) # d_1 is subap size microns
        theta_deg = np.degrees(theta_rad) 

        reflection_angles.append({
            "block_id": b["id"],
            "theta_rad": theta_rad,
            "theta_deg": theta_deg
        })

    return reflection_angles

angles = calc_ref_angle(d_1=subap_size, blocks=blocks)

# prints each block's reflection angle for propogation in z (sanity check center block = 0 deg)
'''
for angle in angles:
    print(f"Block {angle['block_id']}: θ = {angle['theta_deg']:.4f} degrees")
print(max(angles, key=lambda a: a['theta_deg']))
'''

middle_block_id = 213
# ** NEW MIDDLE BLOCK has index is 213 ** now after removing bad blocks 

def center_to_block(blocks, middle_block_id, pixel_size_um=pixel_size_um):
    '''
    calculates the vector from the center block to each block in microns and pixels
    ray propogation depends on distance from the middle of the mask
    
    :param blocks: list of dicts with block info (type: list)
    :param middle_block_id: ID of the middle block of the mask (type: int)
    :param pixel_size_um: pixel size in microns (type: int)

    returns: centers_um is the micron position of the block in blocks (type: np.ndarray)
                centers_pix is the pixel position of the block in blocks 
                  c is the vector from the middle block to the block in blocks
                    c_mag is the magnitude of c
    '''
    centers_pix = []
    centers_um = []
    for b in blocks:
        center_x_pix = b['pixel_x'] - 0.5 + (block_size/2)
        center_y_pix = b['pixel_y'] - 0.5 + (block_size/2)
        centers_pix.append([center_x_pix, center_y_pix])
        centers_um.append([center_x_pix * pixel_size_um, center_y_pix * pixel_size_um])

    centers_pix = np.array(centers_pix)
    centers_um = np.array(centers_um)
    center_pos_um = centers_um[middle_block_id - 1]

    c = centers_um - center_pos_um # vector from center block to each block in microns
    c_mag = np.linalg.norm(c, axis=1)
    return centers_um, centers_pix, c, c_mag

centers_um, centers_pix, c, c_mag = center_to_block(blocks, middle_block_id=middle_block_id, pixel_size_um=pixel_size_um)

def fit_plane(block, x_grid, y_grid):
    """
    Fits a plane z = ax + by + c to a 2D array (block), using global x_grid, y_grid.
    returns: gradient vector n = (a, b, 1) in physical units and n_unit (unit vector of gradient)
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

ray_spots = []
def get_beam_radius(z, r_0, z_focus):
    '''
    beam size at z for movie and plotting
    
    :param z: distance from mask (in microns)
    :param r_0: radius of beam (half of subap size)
    :param z_focus: focus distance (in microns)
    '''
    return r_0 * abs(1 - z / z_focus)

def propagate_ray(c, n, block_idx, z=z_focus):
    '''
    Propagates ray from block at distance z using c and n vectors.
    
    :param c: vector from middle of mask to block
    :param n: gradient of block
    :param block_idx: block index
    :param z: distance from mask
    '''
    v_x = c[block_idx][0] + n[block_idx][0] * z
    v_y = c[block_idx][1] + n[block_idx][1] * z
    v_z = z
    v = [v_x/1000, v_y/1000, v_z/1000]  # convert to mm
    return v

for block_idx in range(len(blocks)):
    v = propagate_ray(c, n, block_idx, z=z_focus)
    # prints every blocks propagation vector (sanity check: Block 213 has no x,y comp: v = [0,0,z])
    #print(f"Block {block_idx +1}: v = {np.float64(v)}")
    ray_spots.append({
        'x_det': v[0],
        'y_det': v[1],
        'block_id': block_idx + 1
    })

def outermost_rays(c, n, blocks, z, r_0, z_focus):
    '''
    Calculates the outermost rays at distance z to determine entire reflected beam width.
    returns: width, min_edge, max_edge of the beam at distance z (type: tuple)
    '''
    x_centers = []
    for block_idx in blocks:
        v = propagate_ray(c, n, block_idx, z)
        x_centers.append(v[0])
    r_z = get_beam_radius(z, r_0, z_focus) / 1000  # convert microns to mm
    min_edge = min(x - r_z for x in x_centers)
    max_edge = max(x + r_z for x in x_centers)
    width = max_edge - min_edge
    return width, min_edge, max_edge

blocks = list(range(len(c)))

def mask_tilt(c, n, blocks, z_range, r_0=r_0, z_focus=z_focus, mask_width=mask_width):
    '''
    Calculates mask tilt angle wrt incoming beam (theta: angle from the horizontal), beam reflection angle (alpha), and beam widths over a range of z distances.
    returns: list of dicts with z, reflection_width_mm, min_edge_mm, max_edge_mm, alpha, beta, theta, x0, z0 (type: list)
    '''
    results = []
    for z in z_range:
        width, min_edge, max_edge = outermost_rays(c, n, blocks, z, r_0, z_focus)
        x = width - ((width/2) - (mask_width/2))
        alpha_rad = np.arctan((z/1000)/x)
        alpha = np.degrees(alpha_rad)
        center_x = (min_edge + max_edge) / 2
        center_z = z / 1000
        perp_dx = np.cos(alpha_rad)
        perp_dz = -np.sin(alpha_rad)
        x0 = center_x - (width/2) * perp_dx
        z0 = center_z - (width/2) * perp_dz
        theta_rad = np.arctan(z0/(np.abs(x0)+(mask_width/2)))
        theta = np.degrees(theta_rad)

        results.append({
            'z': z,
            'reflection_width_mm': width,
            'min_edge_mm': min_edge,
            'max_edge_mm': max_edge,
            'alpha': alpha,
            'theta': theta,
            'x0': x0,
            'z0': z0
        })
    return results
z_range = np.linspace(0, 65000, 201) # microns, linspace from 0 to 60 mm in 0.1 mm steps
results_focus = mask_tilt(c, n, blocks, z_range)

# angle explanation: if the incoming laser is at 0 degrees then it'd be parallel to the mask, making theta 0 degrees. if theta is 45 degrees, then the mask is tilted 45 deg wrt the incoming laser
# alpha is the angle of the reflected beam width itself. which is not the same as theta, because apertures are not tilted at symmetrical angles throughout the mask. alpha has no 
# useful meaning other than knowing what exact angle to place your downstream lens to capture the beam at its reflected axis

#prints the distance z, beam width, min and max ray positions, and mask angle wrt incoming laser

print(f"{'z (mm)':>12} | {'beam width (mm)':>18} | {'min ray (mm)':>18} | {'max ray (mm)':>18} | {'mask angle wrt. incoming laser':>18}")
print('-' * 150)
for row in results_focus:
    z_mm = row['z'] / 1000.0
    print(
        f"{z_mm:8.3f} mm | "
        f"{row['reflection_width_mm']:16.4f} mm | "
        f"{row['min_edge_mm']:15.4f} mm | "
        f"{row['max_edge_mm']:15.4f} mm | "
        f"{row['theta']:24.4f} deg"
    )


# --------------------------------- PLOTTING & MOVIES -----------------------------------
# f number and beam width versus z for BOTH masks 
        # uses tip/tilt angle calc .py file by dropping a copy of the tip tilt file into the same folder as the focus file 

z_vals_focus = [row['z'] for row in results_focus]  # z in microns
width_vals_focus = [row['reflection_width_mm'] for row in results_focus]  # D in mm

z_over_D_focus = [((z/1000)-(z_focus/1000)) / D if D != 0 else float('nan') for z, D in zip(z_vals_focus, width_vals_focus)] # subtracts z - z_focus/1000 because f number is calculated from focal plane and focus mask focuses at z=2.7mm
z_mm_focus = [z/1000 for z in z_vals_focus]

z_vals_tt = [row['z'] for row in results_tt]  # z in microns
width_vals_tt = [row['reflection_width_mm'] for row in results_tt]  # D in mm

z_over_D_tt = [((z/1000)) / D if D != 0 else float('nan') for z, D in zip(z_vals_tt, width_vals_tt)]
z_mm_tt = [z/1000 for z in z_vals_tt]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_mm_focus, z_over_D_focus, color='cornflowerblue', label='focus mask f#(z)')
ax.plot(z_mm_tt, z_over_D_tt, color='limegreen', label='tip/tilt mask f#(z)')
ax.set_xlabel('z (mm)')
ax.set_ylabel('f# = F/D')
ax.axhline(y=1, color='orangered', linestyle='--', label='f# = 1')
plt.title('f Number vs z (distance from mask)')
ax.grid(True)
ax.legend(loc='lower right')
ax.set_ylim(-0.5, 1.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(z_mm_focus, width_vals_focus, color='cornflowerblue', label='focus beam width')
ax.plot(z_mm_tt, width_vals_tt, color='limegreen', label='tip/tilt beam width')
ax.axhline(y=12.7, color='crimson', linestyle='--', label='1/2 in')
ax.axhline(y=25.4, color='orangered', linestyle='--', label='1 in')
ax.axhline(y=50.8, color='orange', linestyle='--', label='2 in')

ax.set_xlabel('z (mm)')
ax.set_ylabel('beam width (mm)')
plt.title('Beam width vs z (distance from mask)')
ax.grid(True)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()


# cross section plots
'''
#center_y = focus_sub_mask.shape[0] // 2 # use these for focus subtracted cross sections
#center_x = focus_sub_mask.shape[1] // 2

#y_slice = focus_sub_mask[center_y, :]    
#x_slice = focus_sub_mask[:, center_x] 

center_y = mask.shape[0] // 2 # use these for original mask (focus implemented) cross sections
center_x = mask.shape[1] // 2

y_slice = mask[center_y, :]-1033 # subtract to get them centered around 0    
x_slice = mask[:, center_x]-1033

x_microns = np.arange(mask.shape[0]) * pixel_size_um
y_microns = np.arange(mask.shape[1]) * pixel_size_um

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(x_microns, x_slice, label='x slice', color='slateblue')
axes[0].set_xlabel('Microns')
axes[0].set_ylabel('Microns')
axes[0].grid(True)
axes[0].legend()
axes[0].set_ylim(-300,300)
axes[1].plot(y_microns, y_slice, label='y slice', color='mediumorchid')
axes[1].set_xlabel('Microns')
axes[1].set_ylabel('Microns')
axes[1].set_ylim(-300,300)
axes[1].grid(True)
axes[1].legend()

plt.suptitle('Central Cross Sections of Focus Mask')
plt.tight_layout()
plt.show()
'''

# mask tilt angle plot
    # choose a z you want to see the reflection for in highlight_z and will output the angle of the mask wrt to incoming laser
    # uses z_range from above when commenting out 'continue'
'''
highlight_z = 10000  # μm
highlight_tol = 20  # mm tolerance for floating point

fig, ax = plt.subplots(figsize=(8, 8))
cmap = cm.viridis
norm = plt.Normalize(min([r['theta'] for r in results_focus]), max([r['theta'] for r in results_focus]))
line_length = 30  # mm

for entry in results_focus:
    theta = entry['theta']
    color = cmap(norm(theta))
    z_um = entry['z']
    z_mm = z_um / 1000

    # if this is the z to highlight
    is_highlight = abs(z_um - highlight_z) < highlight_tol

    # Incoming theta 
    theta_rad = np.deg2rad(theta)
    x_start = (mask_width/2) - line_length * np.cos(theta_rad)
    z_start = 0 + line_length * np.sin(theta_rad)
    if is_highlight:
        ax.plot([x_start, (mask_width/2)], [z_start, 0], color='red', linewidth=2, label=f'incoming angle {theta:.2f}°' if f'incoming angle {theta:.2f}°' not in ax.get_legend_handles_labels()[1] else "")
        ax.plot([x_start - mask_width, -(mask_width/2)], [z_start, 0], color='red', linewidth=2)
    else:
        #continue # if you comment out continue, it will plot all the other incoming rays and their respective reflected beams
        ax.plot([x_start, (mask_width/2)], [z_start, 0], color=color, linewidth=1)

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
        ax.plot([x0, x1], [z0, z1], color='orange', linewidth=2, label='reflected beam width' if 'reflected beam width' not in ax.get_legend_handles_labels()[1] else "")
    else:
        #continue # if you comment out continue, it will plot all the other incoming rays and their respective reflected beams
        ax.plot([x0, x1], [z0, z1], color='cornflowerblue', linewidth=1, alpha=0.5)

# mask at z=0
ax.plot([-(mask_width/2), (mask_width/2)], [0, 0], color='k', linewidth=3, label='mask')

#sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#cbar = plt.colorbar(sm, ax=ax)
#cbar.set_label('theta (deg)')

ax.set_xlabel('x (mm)')
ax.set_ylabel('z (mm)')
ax.set_ylim(-1, 25)
ax.set_xlim(-15, 15)
ax.set_title('focus beam width and incoming ray angle (mask tilt)')
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.legend()
plt.show()
'''

# slider for beam spots at z snapshot 
'''
z_max = 20000 # microns - when you want the slider to stop
def beams_at_z(c, n, blocks, r_0, z_focus, z_max):
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.15)

    z_init = 0 # can set to a val to get a still at some z
    x = []
    y = []
    for block_idx in blocks:
        v = propagate_ray(c, n, block_idx, z_init)
        x.append(v[0])
        y.append(v[1])

    scatter = ax.scatter(x, y, c='blueviolet', s=5)

    r_z = get_beam_radius(z_init, r_0, z_focus) / 1000  # mm
    circles = [plt.Circle((xi, yi), r_z, fill=False, edgecolor='plum', linestyle='--') for xi, yi in zip(x, y)]
    for circle in circles:
        ax.add_patch(circle)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"focus ray spots and beam size at z = {z_init/1000:.3f} mm")
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    #plt.show()

    # slider
    ax_z = plt.axes([0.15, 0.05, 0.7, 0.03])
    slider = Slider(ax_z, 'z (microns)', 0, z_max, valinit=0, valstep=1)

    def update(val):
        z = slider.val
        x = []
        y = []
        for block_idx in blocks:
            v = propagate_ray(c, n, block_idx, z)
            x.append(v[0])
            y.append(v[1])
        scatter.set_offsets(np.c_[x, y])

        r_z = get_beam_radius(z, r_0, z_focus) / 1000  # mm
        for circle, xi, yi in zip(circles, x, y):
            circle.center = (xi, yi)
            circle.set_radius(r_z)
        
        ax.set_title(f"focus ray spots and beam size at z = {z/1000:.3f} mm")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

beams_at_z(c, n, range(len(blocks)), r_0=r_0, z_focus=z_focus, z_max=z_max)
'''

# movie (no beam size)
'''
z_start = 0
z_end = 8000 # microns
num_frames = 180 
z_vals = np.linspace(z_start, z_end, num_frames)

num_blocks = len(blocks)
block_ids = np.arange(1, num_blocks + 1)
cmap = plt.get_cmap('plasma')
norm = mcolors.Normalize(vmin=1, vmax=num_blocks)

fig, ax = plt.subplots(figsize=(8, 8))
scat = ax.scatter([], [], c=[], s=40, cmap=cmap, norm=norm)

def init():
    range = 10 # mm
    ax.set_xlim(-range, range)  
    ax.set_ylim(-range, range) 
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (m)")
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
    ax.set_title(f"focus ray spots at z = {z/1000:.0f} mm")
    return scat,

ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=50)
cbar = plt.colorbar(scat, ax=ax, orientation='vertical')
cbar.set_label('Block ID')
plt.show()
'''
#ani.save('focus_spots.gif', writer='pillow', fps=20)


# movie w beamsize
'''
z_start = 0
z_end = 15000 # microns
num_frames = 180 
z_vals = np.linspace(z_start, z_end, num_frames)

num_blocks = len(blocks)
block_ids = np.arange(1, num_blocks + 1)
cmap = plt.get_cmap('plasma')
norm = mcolors.Normalize(vmin=1, vmax=num_blocks)

fig, ax = plt.subplots(figsize=(8, 8))
scat = ax.scatter([], [], c=[], s=7, cmap=cmap, norm=norm)

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
    lim = 18 # mm
    ax.set_xlim(-lim, lim)  
    ax.set_ylim(-lim, lim)
    #ax.legend()

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
    r_z = get_beam_radius(z, r_0, z_focus) / 1000  # mm
    for circle, xi, yi in zip(beam_circles, x, y):
        circle.center = (xi, yi)
        circle.set_radius(r_z)
    ax.set_title(f"fastrSHWFS Focus Mask Spots & Beam Size at z = {z/1000:.0f} mm")
    return (scat, *beam_circles)

ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=50)
cbar = plt.colorbar(scat, ax=ax, orientation='vertical', shrink=0.8)
cbar.set_label('Block ID')
plt.show()
'''
#ani.save('focus_spots_beamsize.gif', writer='pillow', fps=20)


# ---- extra plots (useless) ----


# lin plot for angle vs beam width
'''
z_vals = [row['z'] for row in results_focus]
#alpha_vals = [row['alpha'] for row in results_focus]
theta_vals = [row['theta'] for row in results_focus]
width_vals = [row['reflection_width_mm'] for row in results_focus]

fig, ax1 = plt.subplots(figsize=(8,5))

# angle vs z on primary y-axis
ax1.plot(z_vals, theta_vals, color='mediumpurple', label='theta (deg)')
#ax1.plot(z_vals, alpha_vals, color='dodgerblue', label='alpha (deg)')
ax1.set_ylim(0,100)
ax1.set_xlabel('z (μm)')
ax1.set_ylabel('degrees')
ax1.tick_params(axis='y')

# secondary y-axis for width
ax2 = ax1.twinx()
ax2.plot(z_vals, width_vals, color='mediumseagreen', label='beam width (mm)')
ax2.set_ylabel('beam width (mm)')
ax2.tick_params(axis='y')
ax2.axhline(y=25.4, color='orangered', linestyle='--', linewidth=1.5, label='1 in')
ax2.axhline(y=12.7, color='orange', linestyle='--', linewidth=1.5, label='0.5 in')

plt.title('focus mask angle and beam width vs z')
ax1.grid(True)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()
'''

# f number vs z just focus
'''
z_vals = [row['z'] for row in results_focus]  # z in microns
width_vals = [row['reflection_width_mm'] for row in results_focus]  # D in mm

z_over_D = [((z/1000)-(z_focus/1000)) / D if D != 0 else float('nan') for z, D in zip(z_vals, width_vals)] # convert z to mm and subtract 2.7 mm for distance from spot plane
z_mm = [z/1000 for z in z_vals]
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(z_mm, z_over_D, color='cornflowerblue', label='focus mask F#(z)')
ax.set_xlabel('z (mm)')
ax.set_ylabel('F# = f/D')
ax.axhline(y=1, color='orangered', linestyle='--', label='F# = 1')

plt.title('F Number vs z (distance from mask)')
ax.grid(True)
ax.legend(loc='lower right')
ax.set_ylim(-0.5,1.3)

plt.tight_layout()
plt.show()
'''