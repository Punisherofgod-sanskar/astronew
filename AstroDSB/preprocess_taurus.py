
import numpy as np
from astropy.io import fits
from pathlib import Path
from skimage.transform import resize
import matplotlib.pyplot as plt

# --- CONFIG ---
# Update this path to whichever FITS file you want to use (extdPSW is a good candidate)
fits_path = "workspace/I2SB/AstroDSB/herschel_data/anonymous1774626771/1342202252/level2_5/extdPSW/hspirepsw451_25pxmp_0438_p2602_1342202252_1342202253_1462474758423.fits.gz"
output_path = Path("./data/taurus_L1495_column_density_map_rot_norm_128.npy")

# --- STEP 1: Load FITS data ---
hdul = fits.open(fits_path)
hdul.info()  # prints extensions so you can see what's inside

# Science data is usually in extension 1 (sometimes 0)
data = hdul[1].data if hdul[1].data is not None else hdul[0].data
hdul.close()

# --- STEP 2: Handle NaNs or invalid values ---
data = np.nan_to_num(data, nan=0.0)

# --- STEP 3: Normalize values to [0,1] ---
data_min, data_max = data.min(), data.max()
if data_max > data_min:
    data = (data - data_min) / (data_max - data_min)

# --- STEP 4: Resize to 128x128 ---
data_resized = resize(data, (128, 128), anti_aliasing=True)

# --- STEP 5: Rotate if needed (adjust k=0..3 depending on orientation) ---
data_processed = np.rot90(data_resized, k=1)

# --- STEP 6: Save as .npy ---
output_path.parent.mkdir(parents=True, exist_ok=True)
np.save(output_path, data_processed)

print(f"Saved Taurus map to {output_path}")

# --- Optional: Quick check ---
plt.imshow(data_processed, cmap="inferno")
plt.colorbar()
plt.title("Taurus L1495 Column Density (128x128)")
plt.show()
