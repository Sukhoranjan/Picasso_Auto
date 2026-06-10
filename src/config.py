import toml
from pathlib import Path
from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, model_validator

DriveType = Literal["ssd", "hdd"]
FitMethod = Literal["mle", "lq", "lq-gpu", "lq-3d", "lq-gpu-3d", "avg"]
RefinementMethod = Literal["gold", "aim", "none"]
BlurMethod = Literal["smooth", "gaussian", "convolve", "gaussian_iso", "None"]

class LocalizeConfig(BaseModel):

    fit_method: FitMethod = Field(default="lq-gpu", description="Fitting algorithm.")
    pixelsize: float = Field(default=130.0, description="Camera pixel size in nm.")
    gradient_threshold: int = Field(default=5000, ge=0, description="Minimum net gradient.")
    box_side_length: int = Field(default=7, ge=1, description="Box side length (must be odd).")
    camera_baseline: int = Field(default=100)
    camera_sensitivity: float = Field(default=0.22)
    camera_gain: int = Field(default=1)
    quantum_efficiency: float = Field(default=1.0)
    drift_segmentation: int = Field(default=0, description="0 disables default undrifting.")
    drive_type: DriveType = Field(default="ssd", description="Target drive type ('ssd' or 'hdd') to optimize parallel I/O.")
    roi: Optional[Tuple[int, int, int, int]] = Field(
        default=None, 
        description="Format: (y_min, x_min, y_max, x_max)"
    )

class UndriftConfig(BaseModel):

    pixelsize: float = Field(default=130.0, description="Camera pixel size in nm.")
    suffix: str = Field(default="_locs.hdf5")
    show_fiducials: bool = Field(default=True)
    rcc_segmentation: int = Field(default=1000, gt=0)
    refinement_method: RefinementMethod = Field(default="gold")
    aim_segmentation: int = Field(default=1000)
    aim_intersectdist: float = Field(default=0.1538)
    aim_roiradius: float = Field(default=0.4615)
    max_fiducial_ellipticity: float = Field(default=1.5, description="Max ellipticity for gold.")
    drive_type: DriveType = Field(default="ssd", description="Target drive type ('ssd' or 'hdd') ")
    fiducial_roi_size: float = Field(default=150.0, description="Size of the fiducial pick box in nm.")
    max_fiducials: int | str = Field(default=10, description="Max fiducials to use, or 'all'.")

class AlignConfig(BaseModel):

    pixelsize: float = Field(default=130.0, description="Camera pixel size in nm.")
    channel_dirs: List[str] = Field(default_factory=list, description="List of channel folders.")
    show_alignment: bool = Field(default=True)
    merged_output_name: str = Field(default="merged_multicolor.hdf5")

class RenderConfig(BaseModel):

    pixelsize: float = Field(default=130.0, description="Camera pixel size in nm.")
    input_file: str = Field(default="merged_multicolor.hdf5")
    output_suffix: str = Field(default="_final_render.png")
    oversampling: int = Field(default=10, gt=0)
    blur_method: BlurMethod = Field(default="smooth")
    min_blur_width: float = Field(default=0.1, gt=0)
    gamma: float = Field(default=0.9, gt=0)
    percentile_bounds: Tuple[float, float] = Field(default=(1.0, 99.0))
    colors: List[str] = Field(default_factory=list)
    channel_labels: List[str] = Field(default_factory=list)

class RunFlags(BaseModel):

    localize: bool = Field(default=True)
    undrift: bool = Field(default=True)
    align: bool = Field(default=True)
    render: bool = Field(default=True)

class PipelineConfig(BaseModel):
    root_dir: Path = Field(..., description="Absolute path to data folder.")
    ext: str = Field(default="_1_MMStack_Default.ome.tif")
    run: RunFlags = Field(default_factory=RunFlags)
    localize: LocalizeConfig = Field(default_factory=LocalizeConfig)
    undrift: UndriftConfig = Field(default_factory=UndriftConfig)
    align: AlignConfig = Field(default_factory=AlignConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)

    @model_validator(mode='after')
    def validate_cross_dependencies(self) -> "PipelineConfig":
        """Ensure lists in rendering and alignment match exactly."""
        channels = self.align.channel_dirs
        colors = self.render.colors
        labels = self.render.channel_labels
        if len(channels) > 0:
            if colors and len(colors) != len(channels):
                raise ValueError(f"Number of colors ({len(colors)}) must match number of channels ({len(channels)}).")
            if labels and len(labels) != len(channels):
                raise ValueError(f"Number of labels ({len(labels)}) must match number of channels ({len(channels)}).")
        return self

def load_config(filepath: str | Path) -> PipelineConfig:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config_dict = toml.load(f)
        
    return PipelineConfig(**config_dict)