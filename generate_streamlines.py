import numpy as np
import nibabel as nib
from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.dti import TensorModel, fractional_anisotropy, color_fa
from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.utils import length
from dipy.tracking.streamline import Streamlines

path = "Sample1_MRI/Data/"

def generate_dti_streamlines(nifti_file="data.nii.gz", bval_file="bvals", bvec_file="bvecs"):
    # ------------------------
    # 1. Load diffusion data
    # ------------------------
    data, affine = load_nifti(nifti_file)
    img = nib.load(nifti_file)
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)  # keyword args
    
    # ------------------------
    # 2. Create brain mask
    # ------------------------
    S0_mask, mask = median_otsu(data[:, :, :, 0], median_radius=4, numpass=4)
    
    # ------------------------
    # 3. Fit DTI model
    # ------------------------
    ten_model = TensorModel(gtab)
    ten_fit = ten_model.fit(data, mask=mask)
    fa = fractional_anisotropy(ten_fit.evals)
    cfa = color_fa(fa, ten_fit.evecs)
    
    # ------------------------
    # 4. Extract directions using peaks_from_model (DTI)
    # ------------------------
    sphere = get_sphere(name='symmetric724')  # keyword arg
    peaks = peaks_from_model(
        model=ten_model,
        data=data,
        sphere=sphere,
        mask=mask,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        return_odf=False
    )
    
    # ------------------------
    # 5. Stopping criterion & seeds
    # ------------------------
    stopping_criterion = ThresholdStoppingCriterion(fa, 0.2)
    #Change the density to increase the number of tracts
    seeds = utils.seeds_from_mask(mask, affine, density=8)
    
    # ------------------------
    # 6. Run tractography
    # ------------------------
    streamline_generator = LocalTracking(
        peaks,
        stopping_criterion,
        seeds,
        affine=affine,
        step_size=0.5
    )
    
    streamlines = Streamlines(streamline_generator)
    streamlines = list(streamlines)
    
    # ------------------------
    # 7. Remove short tracts (<30mm)
    # ------------------------
    streamlines = [sl for sl in streamlines if length(np.array(sl)) > 30]

    # ------------------------
    # 8. Save streamlines in TrackVis format
    # ------------------------
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = img.header.get_zooms()[:3]
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = tuple(int(i) for i in fa.shape)
    
    tensor_streamlines_trk = ((sl, None, None) for sl in streamlines)
    ten_sl_fname = 'dti_streamlines.trk'
    nib.trackvis.write(ten_sl_fname, tensor_streamlines_trk, hdr, points_space='voxel')
    
    print(f"Saved {len(streamlines)} streamlines to {ten_sl_fname}")
    return streamlines

if __name__ == "__main__":
    streamlines = generate_dti_streamlines(
        nifti_file=path + "data.nii.gz",
        bval_file=path + "bvals",
        bvec_file=path + "bvecs"
    )
    print("DTI-only streamlines generation complete!")
