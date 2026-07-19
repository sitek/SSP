import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from glob import glob
from typing import Dict, List, Optional
from nilearn import datasets, surface, plotting

"""
Project per-ROI group statistics (e.g. t-values) onto the fsaverage cortical
surface and render them as lateral/medial views for both hemispheres.

Usage:
    from roi_surface_plotting import plot_roi_surface_stat

    # region_hemi name (e.g. 'L-HG') -> t-value
    stat_dict = {'L-HG': 2.35, 'R-HG': 3.10, ...}
    # region_hemi name -> path to that ROI's volumetric mask (any subject/group mask
    # in the same space as the statistic, used only to find which surface vertices
    # belong to the ROI)
    mask_path_dict = {'L-HG': '/path/to/L-HG_mask.nii.gz', ...}

    fig = plot_roi_surface_stat(stat_dict, mask_path_dict, title='sound vs. baseline')
    fig.savefig('network-auditory_contrast-sound_surface-tstat.svg')
"""

_HEMI_PREFIX = {'L': 'left', 'R': 'right'}


def _hemisphere_for_region(region_hemi: str) -> str:
    prefix = region_hemi[0].upper()
    if prefix not in _HEMI_PREFIX:
        raise ValueError(
            f"region_hemi name '{region_hemi}' must start with 'L' or 'R' to "
            "indicate hemisphere"
        )
    return _HEMI_PREFIX[prefix]


def build_mask_path_dict(
    region_hemi_list: List[str],
    masks_dir: str,
    network_name: str,
    space_label: str,
) -> Dict[str, str]:
    """Glob one representative ROI mask per region_hemi name (e.g. 'L-HG'),
    mirroring the mask directory naming used by mask_stat_maps() in
    group_level_all_ROI.ipynb."""
    if network_name == 'auditory':
        mask_network_name = 'dseg'
    elif network_name == 'pfc':
        mask_network_name = 'dseg-pfc'
    else:
        mask_network_name = network_name

    mask_path_dict = {}
    for region_hemi in region_hemi_list:
        pattern = os.path.join(
            masks_dir, '*', f'space-{space_label}', f'masks-{mask_network_name}',
            f'*{region_hemi}*.nii.gz',
        )
        matches = glob(pattern)
        if not matches:
            raise FileNotFoundError(f"no mask found for '{region_hemi}' matching {pattern}")
        mask_path_dict[region_hemi] = matches[0]
    return mask_path_dict


def _load_gifti_mesh(gifti_path):
    gifti = nib.load(gifti_path)
    coords, faces = gifti.darrays[0].data, gifti.darrays[1].data
    return coords, faces


def project_roi_stats_to_surface(
    stat_dict: Dict[str, float],
    mask_path_dict: Dict[str, str],
    fsaverage=None,
    interpolation: str = 'linear',
    coverage_thresh: float = 0.1,
):
    """Project volumetric per-ROI statistics onto the fsaverage surface.

    Sampling is done on the *pial* mesh (correct anatomical space for
    `vol_to_surf`), but vertex indices carry over 1:1 to the *inflated* mesh
    used for display, since both share the same fsaverage tessellation.

    Returns a dict with, per hemisphere ('left'/'right'):
        texture      -- (n_vertices,) float array of stat values (0 elsewhere)
        roi_vertices -- {region_hemi: boolean (n_vertices,) array}
        coords_infl, faces_infl -- inflated mesh geometry (for plotting)
        coords_pial          -- pial mesh coordinates (for label placement)
        sulc                 -- path to sulcal-depth background map
    """
    if fsaverage is None:
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage')

    out = {}
    for hemi in ('left', 'right'):
        coords_pial, faces_pial = _load_gifti_mesh(fsaverage[f'pial_{hemi}'])
        coords_infl, faces_infl = _load_gifti_mesh(fsaverage[f'infl_{hemi}'])
        out[hemi] = {
            'texture': np.zeros(coords_pial.shape[0]),
            'roi_vertices': {},
            'coords_infl': coords_infl,
            'faces_infl': faces_infl,
            'coords_pial': coords_pial,
            'faces_pial': faces_pial,
            'sulc': fsaverage[f'sulc_{hemi}'],
        }

    for region_hemi, stat_value in stat_dict.items():
        hemi = _hemisphere_for_region(region_hemi)
        mask_path = mask_path_dict[region_hemi]

        vertex_coverage = surface.vol_to_surf(
            mask_path,
            (out[hemi]['coords_pial'], out[hemi]['faces_pial']),
            interpolation=interpolation,
        )
        roi_vertices = vertex_coverage > coverage_thresh

        out[hemi]['texture'][roi_vertices] = stat_value
        out[hemi]['roi_vertices'][region_hemi] = roi_vertices

    return out


def _dominant_view_for_roi(coords_infl, roi_vertices, hemi):
    """Classify an ROI as 'lateral' or 'medial' from its vertex positions.

    The lateral surface of a hemisphere bulges away from the midline (more
    negative x for the left hemisphere, more positive x for the right), while
    medial regions sit close to the midline. Comparing an ROI's mean x to the
    whole hemisphere's mean x gives a robust, atlas-agnostic classification
    that matches the camera angles nilearn uses for 'lateral'/'medial' views.
    """
    relative_x = coords_infl[roi_vertices, 0].mean() - coords_infl[:, 0].mean()
    if hemi == 'left':
        return 'lateral' if relative_x < 0 else 'medial'
    return 'lateral' if relative_x > 0 else 'medial'


def plot_roi_surface_stat(
    stat_dict: Dict[str, float],
    mask_path_dict: Dict[str, str],
    fsaverage=None,
    cmap: str = 'coolwarm',
    vlim: Optional[float] = None,
    threshold: Optional[float] = None,
    title: Optional[str] = None,
    add_labels: bool = True,
    views=('lateral', 'medial'),
    dpi: int = 300,
):
    """Render per-ROI group statistics on the fsaverage surface.

    Produces a grid with one row per view (lateral/medial, or just lateral
    if the ROI set has no medial regions) and one column per hemisphere,
    with a single shared symmetric colorbar and black ROI boundary contours.
    Each ROI's name is only labeled on the view (lateral/medial) it actually
    faces, per `_dominant_view_for_roi`.
    """
    surf_data = project_roi_stats_to_surface(stat_dict, mask_path_dict, fsaverage=fsaverage)

    if vlim is None:
        vlim = max(abs(v) for v in stat_dict.values())
    vmin, vmax = -vlim, vlim

    nrows = len(views)
    fig = plt.figure(figsize=(10, 4.5 * nrows), dpi=dpi)
    panels = [(hemi, view) for view in views for hemi in ('left', 'right')]

    for i, (hemi, view) in enumerate(panels):
        ax = fig.add_subplot(nrows, 2, i + 1, projection='3d')
        data = surf_data[hemi]

        plotting.plot_surf_stat_map(
            surf_mesh=(data['coords_infl'], data['faces_infl']),
            stat_map=data['texture'],
            hemi=hemi,
            view=view,
            bg_map=data['sulc'],
            bg_on_data=True,
            darkness=0.5,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            threshold=threshold,
            colorbar=False,
            axes=ax,
            figure=fig,
            title=f'{hemi} {view}',
        )

        for region_hemi, roi_vertices in data['roi_vertices'].items():
            plotting.plot_surf_contours(
                surf_mesh=(data['coords_infl'], data['faces_infl']),
                roi_map=roi_vertices.astype(int),
                levels=[1],
                colors=['black'],
                axes=ax,
                figure=fig,
            )

            if (add_labels and roi_vertices.any()
                    and _dominant_view_for_roi(data['coords_infl'], roi_vertices, hemi) == view):
                cx, cy, cz = data['coords_infl'][roi_vertices].mean(axis=0)
                ax.text(cx, cy, cz, region_hemi, fontsize=7, fontweight='bold',
                        ha='center', va='center', color='black')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation='vertical',
                         fraction=0.03, pad=0.02, shrink=0.6)
    cbar.set_label('group t-statistic')

    if title:
        fig.suptitle(title)

    return fig
