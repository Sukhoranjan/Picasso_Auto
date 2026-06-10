import io as python_io
import numpy as np
import concurrent.futures
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Dict, Any
from loguru import logger

import matplotlib.pyplot as plt
from matplotlib import patches, colors as mcolors
from picasso import io, render
from src.config import RenderConfig

def render_settings(ch, locs, info, config: RenderConfig, color):
    locs_ch = locs[locs["channel"] == ch]
    img = render.render(locs_ch, info, config.oversampling, blur_method=config.blur_method, min_blur_width=config.min_blur_width)[1]
    #contrast
    p_low, p_high = np.percentile(img, config.percentile_bounds)
    denom = (p_high - p_low) if (p_high - p_low) != 0 else 1.0
    img = np.clip((img - p_low) / denom, 0, 1) ** config.gamma
    
    rgb_img = np.zeros((*img.shape, 3), dtype=np.float32)
    for j in range(3):
        rgb_img[:, :, j] = img * color[j]
        
    return rgb_img, img[..., np.newaxis]
#
def render_channels(root_dir: Path, config: RenderConfig) -> Dict[str, Any]:
    input_path = root_dir / config.input_file
    if not input_path.exists():
        raise FileNotFoundError(f"Render target not found: {input_path}")

    out_path = input_path.with_name(input_path.name.replace(".hdf5", config.output_suffix))
    color_palette = [mcolors.to_rgb(c) for c in config.colors]

    locs, info = io.load_locs(str(input_path))
    info = info if isinstance(info, list) else [info]
    channels = np.unique(locs["channel"])
    
    logger.info(f"Rendering {len(channels)} channels...")

    combined = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(channels)) as executor:
        futures = [executor.submit(render_settings, ch, locs, info, config, color_palette[i % len(color_palette)]) 
                   for i, ch in enumerate(channels)]
        
        for future in concurrent.futures.as_completed(futures):
            rgb_img, alpha = future.result()
            combined = rgb_img if combined is None else (1 - alpha) * combined + alpha * rgb_img

    if combined is None:
        raise RuntimeError("No channels were rendered.")

    max_val = np.max(combined)
    combined = np.clip(combined / (max_val if max_val != 0 else 1.0), 0, 1)
    
    base_image = Image.fromarray((combined * 255).astype(np.uint8)).convert("RGBA")
    W, H = base_image.size
    logger.info("Applying scalebar and legend...")
    camera_pixel_nm = config.pixelsize  
    pixel_size_nm = camera_pixel_nm / config.oversampling
    physical_width_nm = W * pixel_size_nm
    
    target_bar_nm = physical_width_nm * 0.15
    possible_bars = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    bar_nm = min(possible_bars, key=lambda x: abs(x - target_bar_nm))
    label_text = f"{bar_nm / 1000:g} µm" if bar_nm >= 1000 else f"{bar_nm} nm"

    bar_len_px = int(bar_nm / pixel_size_nm)
    bar_height_px = max(5, int(H * 0.005)) 
    margin = int(W * 0.02)
    sb_x1, sb_y1 = margin, H - margin - bar_height_px
    
    draw = ImageDraw.Draw(base_image)
    draw.rectangle([sb_x1, sb_y1, sb_x1 + bar_len_px, sb_y1 + bar_height_px], fill=(255, 255, 255, 255))
    fig_text, ax_text = plt.subplots(figsize=(2, 0.5), dpi=300)
    ax_text.axis("off")
    ax_text.text(0.5, 0.5, label_text, color='white', ha='center', va='center', fontsize=20, weight='bold')
    buf_text = python_io.BytesIO()
    fig_text.savefig(buf_text, format='png', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_text)
    
    text_img = Image.open(buf_text).convert("RGBA")
    target_text_w = min(bar_len_px, int(W * 0.1))
    text_img = text_img.resize((target_text_w, int(text_img.height * (target_text_w / text_img.width))), Image.Resampling.LANCZOS)
    base_image.alpha_composite(text_img, (sb_x1 + (bar_len_px - text_img.width) // 2, sb_y1 - text_img.height - max(5, int(H * 0.005))))

    # Legend Rendering
    labels = config.channel_labels if config.channel_labels else [str(ch) for ch in channels]
    fig_leg, ax_leg = plt.subplots(figsize=(4, 3), dpi=300)
    ax_leg.axis("off")
    legend_patches = [patches.Patch(color=color_palette[i % len(color_palette)], label=l) for i, l in enumerate(labels)]
    ax_leg.legend(handles=legend_patches, loc="center", frameon=True, framealpha=0.3, facecolor='black', edgecolor='none', labelcolor='white', title="Channels").get_title().set_color('white')
    
    buf_leg = python_io.BytesIO()
    fig_leg.savefig(buf_leg, format='png', transparent=True, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_leg)
    
    legend_img = Image.open(buf_leg).convert("RGBA")
    target_leg_w = max(300, min(int(W * 0.15), W // 3))
    legend_img = legend_img.resize((target_leg_w, int(legend_img.height * (target_leg_w / legend_img.width))), Image.Resampling.LANCZOS)
    base_image.alpha_composite(legend_img, (W - legend_img.width - margin, H - legend_img.height - margin))

    base_image.convert("RGB").save(out_path)
    logger.success(f"Saved final presentation image: {out_path.name}")
    
    return {"image_file": str(out_path), "dimensions": [W, H]}