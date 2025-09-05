import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
from dipy.segment.mask import median_otsu
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data, load_nifti
from dipy.direction import peaks
from dipy.reconst.shm import CsaOdfModel
from dipy.reconst.dti import TensorModel, fractional_anisotropy, color_fa
from dipy.data import get_sphere
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.utils import length
from dipy.tracking.streamline import Streamlines

def generate_streamlines():
    # Load data
    data, affine, img = load_nifti("Sample1_MRI/Data/data.nii.gz", return_img=True)
    fbval = 'Sample1_MRI/Data/bvals'
    fbvec = 'Sample1_MRI/Data/bvecs'
    bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
    
    # Create brain mask
    mask, S0_mask = median_otsu(data[:, :, :, 0])
    
    # Create gradient table
    gtab = gradient_table(bvals, bvecs)
    
    # Fit tensor model and calculate FA
    ten_model = TensorModel(gtab)
    ten_fit = ten_model.fit(data, mask)
    fa = fractional_anisotropy(ten_fit.evals)
    cfa = color_fa(fa, ten_fit.evecs)
    
    # Create CSA-ODF model and get peaks
    csamodel = CsaOdfModel(gtab, 6)
    sphere = get_sphere('symmetric724')
    
    pmd = peaks.peaks_from_model(
        model=csamodel,
        data=data,
        sphere=sphere,
        relative_peak_threshold=.5,
        min_separation_angle=25,
        mask=mask,
        return_odf=False
    )
    
    # Run deterministic tractography using LocalTracking
    stopping_criterion = ThresholdStoppingCriterion(fa, 0.01)
    seeds = utils.seeds_from_mask(mask, affine, density=8)  # density adjusted to get similar number of streamlines
    
    streamline_generator = LocalTracking(pmd, stopping_criterion, seeds,
                                       affine=affine, step_size=0.5)
    streamlines = Streamlines(streamline_generator)
    streamlines = list(streamlines)
    
    # Remove tracts shorter than 30mm
    streamlines = [t for t in streamlines if length(t) > 30]
    
    # Save in trackvis format
    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = img.header.get_zooms()[:3]
    hdr['voxel_order'] = 'LAS'
    hdr['dim'] = fa.shape
    
    tensor_streamlines_trk = ((sl, None, None) for sl in streamlines)
    ten_sl_fname = 'tensor_streamlines.trk'
    nib.trackvis.write(ten_sl_fname, tensor_streamlines_trk, hdr, points_space='voxel')
    
    # Optional: Create connectivity matrix if atlas is available
    try:
        atlas = nib.load('Sample1_MRI/Data/atlas_reg.nii.gz')
        labels = atlas.get_data()
        labelsint = labels.astype(int)
        
        M = utils.connectivity_matrix(streamlines, labelsint, affine=affine)
        
        # Remove background and process matrix
        M = M[1:, 1:]
        M = M[:90, :90]  # Adjust based on your atlas
        np.fill_diagonal(M, 0)
        
        # Save connectome
        np.savetxt("connectome.csv", M, delimiter=",")
        
    except Exception as e:
        print(f"Could not create connectivity matrix: {str(e)}")
    
    return Streamlines(streamlines)

if __name__ == "__main__":
    streamlines = generate_streamlines() 