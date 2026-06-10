import numpy as np
import pandas as pd
from pathlib import Path
from numpy.lib import recfunctions as rfn
from typing import Dict, Any
from loguru import logger
from picasso import io, lib, postprocess
from src.config import AlignConfig, UndriftConfig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_alignment(root_dir: Path, align_config: AlignConfig, undrift_config: UndriftConfig) -> Dict[str, Any]:
    channels = align_config.channel_dirs
    method = undrift_config.refinement_method
    suffix = f"_rcc_{method}.hdf5" if method != "none" else "_rcc.hdf5"
    file_paths = []
    for ch in channels:
        matches = list(root_dir.rglob(f"*{ch}*{suffix}"))
        if not matches:
            raise FileNotFoundError(f"Missing alignment file for channel: {ch}")
        file_paths.append(matches[0])
    locs_list, infos_list = [], []
    for p in file_paths:
        locs, info = io.load_locs(str(p))
        if hasattr(locs, "reset_index"): 
            locs.reset_index(drop=True, inplace=True)
        locs_list.append(locs)
        infos_list.append(info)
    logger.info("Aligning based on RCC and optional Fiducials...")
    current_locs = [l.copy() for l in locs_list]
    
    # Step 1: Image RCC
    logger.info("Performing Image RCC alignment...")
    rcc_aligned_locs = postprocess.align_rcc(current_locs, infos_list)

    def merge_channels(locs_to_merge):
        augmented_lists = []
        for idx, l_copy in enumerate(locs_to_merge):
            ch_num = idx + 1
            if isinstance(l_copy, pd.DataFrame):
                l_aug = l_copy.copy()
                l_aug["channel"] = ch_num
            else:
                if "channel" not in l_copy.dtype.names:
                    l_aug = lib.append_to_rec(l_copy, np.full(len(l_copy), ch_num, dtype=np.int32), "channel")
                else:
                    l_aug = l_copy.copy()
                    l_aug["channel"] = ch_num
            augmented_lists.append(l_aug)

        if isinstance(augmented_lists[0], pd.DataFrame):
            return pd.concat(augmented_lists, ignore_index=True)
        else:
            return rfn.stack_arrays(augmented_lists, asrecarray=True, usemask=False, autoconvert=True)

    merged_rcc = merge_channels(rcc_aligned_locs)
    base_info = infos_list[0] if infos_list else []
    
    rcc_nenas = {}
    for idx, l_rcc in enumerate(rcc_aligned_locs):
        pixelsize = infos_list[idx][0].get("Pixelsize", align_config.pixelsize)
        nena_locs = l_rcc.copy()
        nena_result, nena_px = postprocess.nena(nena_locs, infos_list[idx])
        nena_nm = float(nena_px * pixelsize)
        if not np.isnan(nena_nm):
            logger.info(f"Channel {idx + 1} NeNA after RCC alignment: {nena_nm:.2f} nm")
            rcc_nenas[channels[idx]] = nena_nm
            fig_nena = postprocess.plot_nena(nena_result)
            fig_nena.savefig(str(root_dir / f"{channels[idx]}_align_rcc_nena.png"), dpi=300, bbox_inches="tight")
            plt.close(fig_nena)
        else:
            logger.warning(f"Channel {idx + 1} NeNA calculation returned NaN after RCC alignment.")

    # Step 2: Gold Refinement
    gold_nenas = {}
    if method == "gold":
        accepted_picks, box = None, None
        for info_item in infos_list:
            for inf in reversed(info_item):
                if "Fiducials" in inf and "Fiducial Box" in inf:
                    accepted_picks = [(float(p[0]), float(p[1])) for p in inf["Fiducials"]]
                    box = float(inf["Fiducial Box"])
                    break
            if accepted_picks:
                break
                
        if accepted_picks:
            logger.info("Fiducials found in metadata. Applying Gold structural alignment...")
            final_aligned_locs = postprocess.align_from_picked(
                all_locs=rcc_aligned_locs,
                infos=infos_list,
                picks=accepted_picks,
                pick_shape="Circle",
                pick_size=box
            )
            merged_final = merge_channels(final_aligned_locs)
            
            for idx, l_final in enumerate(final_aligned_locs):
                pixelsize = infos_list[idx][0].get("Pixelsize", align_config.pixelsize)
                nena_locs = l_final.copy()
                nena_result, nena_px = postprocess.nena(nena_locs, infos_list[idx])
                nena_nm = float(nena_px * pixelsize)
                if not np.isnan(nena_nm):
                    logger.info(f"Channel {idx + 1} NeNA after Gold alignment: {nena_nm:.2f} nm")
                    gold_nenas[channels[idx]] = nena_nm
                    fig_nena = postprocess.plot_nena(nena_result)
                    fig_nena.savefig(str(root_dir / f"{channels[idx]}_align_gold_nena.png"), dpi=300, bbox_inches="tight")
                    plt.close(fig_nena)
                      
                else:
                    logger.warning(f"Channel {idx + 1} NeNA calculation returned NaN after Gold alignment.")
        else:
            logger.warning("No valid fiducials found in any channel. Skipping Gold alignment.")
            merged_final = merged_rcc
    else:
        merged_final = merged_rcc

    merged_path = root_dir / align_config.merged_output_name
    
    merge_metadata = {"Generated by": "Picasso Pipeline", "Channels": channels}
    base_info = base_info + [merge_metadata] if isinstance(base_info, list) else [base_info, merge_metadata]

    io.save_locs(str(merged_path), merged_final, base_info)
    logger.info(f"Merged output saved to {merged_path.name}")
    
    return {
        "merged_file": str(merged_path),
        "channels_aligned": len(channels),
        "nena_rcc_nm": rcc_nenas,
        "nena_gold_nm": gold_nenas
    }