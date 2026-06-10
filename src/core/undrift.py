import os
import sys
import subprocess
import numpy as np
import concurrent.futures
from pathlib import Path
from typing import Dict, Any
from loguru import logger
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from picasso import io, postprocess, imageprocess, render
from src.config import UndriftConfig

#For Calculating the Ellipticity of the Picked Fiducials
#Ellipticity calculation method inspired from https://doi.org/10.1086/517873
def calc_elptct(image, thr=0.001):
    thr = thr * image.max()
    mask = image > thr
    if mask.sum() < 10:
        return np.nan, None, None, np.nan, np.nan
    yy, xx = np.indices(image.shape)
    weights = image * mask
    sum = weights.sum()
    cx, cy = (xx * weights).sum() / sum, (yy * weights).sum() / sum
    x, y, w = xx[mask] - cx, yy[mask] - cy, weights[mask]
    Cxx, Cyy, Cxy = (w * x * x).sum() / sum, (w * y * y).sum() / sum, (w * x * y).sum() / sum
    cov = np.array([[Cxx, Cxy], [Cxy, Cyy]])
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    lam1, lam2 = vals[order[0]], vals[order[1]]
    if lam2 == 0: 
        return np.inf, (cx, cy), 0, lam1, lam2
    return np.sqrt(lam1 / lam2), (cx, cy), np.arctan2(vecs[1, order[0]], vecs[0, order[0]]), lam1, lam2

#Render the picked fiducials in a viewport,along the box and save the plot for ellipticity calculation
def plot_fiducials(image, ellipticity, center, angle, lam1, lam2, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="magma", origin="upper", interpolation="nearest")
    if center is not None:
        cx, cy = center
        ax.plot(cx, cy, "c+", markersize=5, mew=2)
        width = 4 * np.sqrt(lam1)
        height = 4 * np.sqrt(lam2)
        ellipse = Ellipse((cx, cy), width=width, height=height, angle=np.degrees(angle),
                          edgecolor="red", facecolor="none", linewidth=1)
        ax.add_patch(ellipse)
        ax.text(0.02, 0.95, f"Ellipticity = {ellipticity:.2f}",
                transform=ax.transAxes, color="white", fontsize=10, va="top",
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))          
    ax.axis("off")
    fig.savefig(out_path, dpi=500, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

#Undrift File
def undrift_file(file_path: Path, config: UndriftConfig) -> dict:
    try:
        locs, info = io.load_locs(str(file_path))
        result_dict = {"file": file_path.name, "status": "success"}

        # Step 1: Do RCC undrifting 
        logger.debug(f"RCC Undrifting: {file_path.name}")
        drift_rcc, locs_rcc = postprocess.undrift(locs, info, segmentation=config.rcc_segmentation, display=True)    
        rcc_path = file_path.with_name(file_path.name.replace("_locs.hdf5", "_rcc.hdf5"))
        io.save_locs(str(rcc_path), locs_rcc, info)
        pixelsize = info[0].get("Pixelsize", config.pixelsize)
        fig_drift = postprocess.plot_drift(drift_rcc, pixelsize)
        fig_drift.savefig(str(file_path.with_name(file_path.name.replace("_locs.hdf5", "_rcc_drift.png"))), dpi=300, bbox_inches="tight") #save rcc drift file
        plt.close(fig_drift)
        pixelsize = info[0].get("Pixelsize", config.pixelsize)
        nena_locs = locs_rcc.copy()
        nena_result, nena_px = postprocess.nena(nena_locs, info)
        nena_nm = float(nena_px * pixelsize)
        logger.info(f"NeNA after RCC: {nena_nm:.2f} nm")
        result_dict["nena_rcc_nm"] = nena_nm
        fig_nena = postprocess.plot_nena(nena_result)
        fig_nena.savefig(str(file_path.with_name(file_path.name.replace("_locs.hdf5", "_rcc_nena.png"))), dpi=300, bbox_inches="tight")
        plt.close(fig_nena)

        # Step 2: Refinement
        method = config.refinement_method
        if method == "none":
            if "nena_rcc_nm" in result_dict:
                result_dict["nena_nm"] = result_dict["nena_rcc_nm"]
            result_dict["final_path"] = str(rcc_path)
            return result_dict

        if method == "gold":
            logger.debug(f"Gold Refinement: {file_path.name}")
            locs_gold = locs_rcc.copy()
            picks, default_box = imageprocess.find_fiducials(locs_gold, info)
            recentered_picks = []
            if picks:
                default_radius = default_box / 2.0
                index_blocks = postprocess.get_index_blocks(locs_gold, info, default_radius)
                for (x0, y0) in picks:
                    cx, cy = float(x0), float(y0)
                    for _ in range(3):  # Mean shift iteration
                        pl = postprocess.picked_locs(
                            locs_gold, info, [(cx, cy)], pick_shape="Circle", 
                            pick_size=default_radius, add_group=False, index_blocks=index_blocks
                        )
                        if pl and not pl[0].empty:
                            new_cx = float(pl[0]["x"].mean())
                            new_cy = float(pl[0]["y"].mean())
                            if abs(new_cx - cx) < 1e-3 and abs(new_cy - cy) < 1e-3:
                                cx, cy = new_cx, new_cy
                                break
                            cx, cy = new_cx, new_cy
                        else:
                            break
                    recentered_picks.append((cx, cy))
            picks = recentered_picks
            # Override the default 900nm box with the user's configured ROI size, 900nm toolsize give bad NeNA, make it as tight as possible
            pixelsize = info[0].get("Pixelsize", config.pixelsize)
            box = config.fiducial_roi_size / pixelsize
            scored_picks = []
            if picks:
                for idx, (x0, y0) in enumerate(picks):
                    half = box / 4
                    viewport = ((y0 - half, x0 - half), (y0 + half, x0 + half))
                    roi_img = render.render(locs_gold, info, oversampling=100, viewport=viewport, blur_method='smooth')[1]
                    ellip, center, angle, lam1, lam2 = calc_elptct(roi_img)
                    if not np.isnan(ellip) and ellip <= config.max_fiducial_ellipticity:
                        scored_picks.append((abs(ellip - 1.0), (x0, y0), roi_img, ellip, center, angle, lam1, lam2))
                        status = "accepted"
                    else:
                        status = "rejected"

                    fid_path = file_path.with_name(file_path.name.replace("_locs.hdf5", f"_fiducial_{idx}_{status}.png"))
                    plot_fiducials(roi_img, ellip, center, angle, lam1, lam2, str(fid_path))
            # Sort by ellipticity score, 1 is best and keep the top 10 if less than 10 then all
            scored_picks.sort(key=lambda x: x[0])
            if str(config.max_fiducials).lower() == "all":
                accepted_picks_info = scored_picks
            else:
                max_fid = int(config.max_fiducials)
                accepted_picks_info = scored_picks[:max_fid]
            
            accepted_picks = [p[1] for p in accepted_picks_info]
            
            if accepted_picks:
                # Save top accepted fiducials separately
                for idx, p_info in enumerate(accepted_picks_info):
                    _, _, roi_img, ellip, center, angle, lam1, lam2 = p_info
                    top_fid_path = file_path.with_name(file_path.name.replace("_locs.hdf5", f"_top_fiducial_{idx+1}.png"))
                    plot_fiducials(roi_img, ellip, center, angle, lam1, lam2, str(top_fid_path))

                # Render and plot the fiducials
                viewport = ((0, 0), (info[0]["Height"], info[0]["Width"]))
                _, img = render.render(locs_gold, info, oversampling=1.0, viewport=viewport, blur_method="smooth")
                fig, ax = plt.subplots(figsize=(10, 10))
                vmax = np.percentile(img, 99.5) if img.max() > 0 else 1.0
                ax.imshow(img, cmap="hot", vmin=0, vmax=vmax)
                for px, py in accepted_picks:
                    circ = patches.Circle((px, py), radius=box/2, fill=False, edgecolor="cyan", linewidth=2)
                    ax.add_patch(circ)
                ax.axis("off")
                plot_path = file_path.with_name(file_path.name.replace("_locs.hdf5", "_fiducials.png"))
                fig.savefig(str(plot_path), dpi=300, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                picks_dict = {
                    "Centers": [[float(p[0]), float(p[1])] for p in accepted_picks],
                    "Diameter": float(box),
                    "Shape": "Circle"
                }
                picks_path = file_path.with_name(file_path.name.replace("_locs.hdf5", "_rcc_gold_ACCEPTED_picks.yaml"))
                with open(picks_path, "w") as f:
                    yaml.dump(picks_dict, f)

                #Finally undrift with selected and filtered fids
                locs_gold, info, drift_fid = postprocess.undrift_from_fiducials(
                    locs=locs_gold,
                    info=info,
                    picks=accepted_picks,
                    pick_size=box / 2
                )

                pixelsize = info[0].get("Pixelsize", config.pixelsize)
                fig_drift = postprocess.plot_drift(drift_fid, pixelsize)
                fig_drift.savefig(str(file_path.with_name(file_path.name.replace("_locs.hdf5", "_rcc_gold_drift.png"))), dpi=300, bbox_inches="tight")
                plt.close(fig_drift)

                # Save the accepted fiducials into the metadata so align.py can use them
                info.append({
                    "Generated by": "Picasso_Auto Undrift (Gold)",
                    "Fiducials": [[float(p[0]), float(p[1])] for p in accepted_picks],
                    "Fiducial Box": float(box)
                })

                pixelsize = info[0].get("Pixelsize", config.pixelsize)
                nena_locs = locs_gold.copy()
                nena_result, nena_px = postprocess.nena(nena_locs, info)
                nena_nm = float(nena_px * pixelsize)
                if not np.isnan(nena_nm):
                    logger.info(f"NeNA after Gold Refinement: {nena_nm:.2f} nm")
                    result_dict["nena_nm"] = nena_nm    
                    fig_nena = postprocess.plot_nena(nena_result)
                    fig_nena.savefig(str(file_path.with_name(file_path.name.replace("_locs.hdf5", "_rcc_gold_nena.png"))), dpi=300, bbox_inches="tight")
                    plt.close(fig_nena)
                else:
                    logger.warning("NeNA calculation returned NaN after Gold Refinement.")                    
                gold_path = file_path.with_name(file_path.name.replace("_locs.hdf5", "_rcc_gold.hdf5"))
                io.save_locs(str(gold_path), locs_gold, info)
                result_dict["final_path"] = str(gold_path)
                result_dict["fiducials_used"] = len(accepted_picks)
            else:
                #if 0 picks passes the threshold, log an error instead of using bad data
                logger.error(f"Failed: ZERO fiducials passed the ellipticity threshold of {config.max_fiducial_ellipticity} for {file_path.name}.")
                result_dict["status"] = "failed"
                result_dict["error"] = f"ZERO fiducials passed the ellipticity threshold of {config.max_fiducial_ellipticity}."
                result_dict["fiducials_used"] = 0
                return result_dict

        elif method == "aim":
            logger.debug(f"AIM Refinement: {file_path.name}")
            cmd = [
                sys.executable, "-m", "picasso", "aim", str(rcc_path),
                "-s", str(config.aim_segmentation), "-i", str(config.aim_intersectdist), "-r", str(config.aim_roiradius)
            ]
            
            env = os.environ.copy()
            picasso_path = Path(__file__).resolve().parent.parent.parent / "picasso"
            env["PYTHONPATH"] = str(picasso_path) + os.pathsep + env.get("PYTHONPATH", "")
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if result.returncode != 0:
                raise RuntimeError(f"AIM command failed with return code {result.returncode}.\nStderr:\n{result.stderr.strip()}")
            aim_path = file_path.with_name(file_path.name.replace("_locs.hdf5", "_rcc_aim.hdf5"))
            cli_out = file_path.with_name(rcc_path.name.replace(".hdf5", "_aim.hdf5"))
            if cli_out.exists():
                cli_out.replace(aim_path)
                locs_aim, info_aim = io.load_locs(str(aim_path))
                pixelsize = info_aim[0].get("Pixelsize", config.pixelsize)
                nena_locs = locs_aim.copy()
                nena_result, nena_px = postprocess.nena(nena_locs, info_aim)
                nena_nm = float(nena_px * pixelsize)
                if not np.isnan(nena_nm):
                    logger.info(f"NeNA after AIM Refinement: {nena_nm:.2f} nm")
                    result_dict["nena_nm"] = nena_nm
                    fig_nena = postprocess.plot_nena(nena_result)
                    fig_nena.savefig(str(file_path.with_name(file_path.name.replace("_locs.hdf5", "_rcc_aim_nena.png"))), dpi=300, bbox_inches="tight")
                    plt.close(fig_nena)
                else:
                    logger.warning("NeNA calculation returned NaN after AIM Refinement.")
            result_dict["final_path"] = str(aim_path)
        return result_dict
    except Exception as e:
        return {"file": file_path.name, "status": "failed", "error": str(e)}
    
#Do the entire thing parallely
def run_batch(root_dir: Path, config: UndriftConfig) -> Dict[str, Any]:
    files = list(root_dir.rglob(f"*{config.suffix}"))
    logger.info(f"Found {len(files)} files to undrift ({config.refinement_method.upper()} refinement).")
    results = {"total": len(files), "success": 0, "failed": 0, "files": []}
    if hasattr(config, "drive_type") and config.drive_type.lower() == "hdd":
        # Set max_workers to 1 to force pure sequential reads. Reading two files at once causes mechanical head seeking, plummeting read speeds.
        max_workers = 1
    else:
        # For SSDs (if data stored in PC), we use more workers for faster processing.
        max_workers = max(1, os.cpu_count() - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(undrift_file, f, config) for f in files]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results["files"].append(res)
            if res["status"] == "success":
                results["success"] += 1
                nena_str = f" (NeNA: {res['nena_nm']:.2f} nm)" if "nena_nm" in res else ""
                logger.info(f"Undrifted: {res['file']}{nena_str}")
            else:
                results["failed"] += 1
                logger.error(f"Undrift failed for {res['file']}: {res.get('error')}")
    return results