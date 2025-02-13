import os

POPULATE_INTENDED_FOR_OPTS = {
        'matching_parameters': ['ImagingVolume', 'Shims', 'ModalityAcquisitionLabel'],
        'criterion': 'Closest'
         } 

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """
    # mp2rage paths done in BIDS format
    t1w = create_key('sub-{subject}/anat/sub-{subject}_T1w')

    # functional paths done in BIDS format
    task_rest = create_key('sub-{subject}/func/sub-{subject}_task-rest_run-{item:02d}_bold')
    task_rest_sbref = create_key('sub-{subject}/func/sub-{subject}_task-rest_run-{item:02d}_sbref')
    task_badaga = create_key('sub-{subject}/func/sub-{subject}_task-badaga_run-{item:02d}_bold')
    task_badaga_sbref = create_key('sub-{subject}/func/sub-{subject}_task-badaga_run-{item:02d}_sbref')
    task_alice = create_key('sub-{subject}/func/sub-{subject}_task-alice_run-{item:02d}_bold')
    task_alice_sbref = create_key('sub-{subject}/func/sub-{subject}_task-alice_run-{item:02d}_sbref')

    # fieldmap paths done in BIDS format
    fieldmap_se_ap = create_key('sub-{subject}/fmap/sub-{subject}_acq-func_dir-AP_run-{item:02d}_epi')
    fieldmap_se_pa = create_key('sub-{subject}/fmap/sub-{subject}_acq-func_dir-PA_run-{item:02d}_epi')
    
    # diffusion paths done in BIDS format
    dwi = create_key('sub-{subject}/dwi/sub-{subject}_dwi')
    dwi_dist_ap = create_key('sub-{subject}/fmap/sub-{subject}_acq-dwi_dir-AP_run-{item:02d}_ge')
    dwi_dist_pa = create_key('sub-{subject}/fmap/sub-{subject}_acq-dwi_dir-PA_run-{item:02d}_ge')
    
    
    # MPM paths - WORK IN PROGRESS - look up BIDS qMRI BEP
    # https://github.com/bids-standard/bids-examples/tree/b8f7f6999a45f1cb94e05629549d60a6393ddfcd/qmri_mpm/sub-01/anat
    # https://bids-specification.readthedocs.io/en/stable/appendices/file-collections.html#magnetic-resonance-imaging
    mpm_t1w = create_key('sub-{subject}/anat/sub-{subject}_acq-t1w_echo-{item:01d}_MPM')
    mpm_pd  = create_key('sub-{subject}/anat/sub-{subject}_acq-pd_echo-{item:01d}_MPM')
    mpm_mt  = create_key('sub-{subject}/anat/sub-{subject}_acq-MTw_echo-{item:01d}_mt-on_MPM')
    mpm_b1  = create_key('sub-{subject}/fmap/sub-{subject}_acq-b1_echo-{item:01d}_MPM')
    mpm_fmap_mag  = create_key('sub-{subject}/fmap/sub-{subject}_desc-fmap_echo-{item:01d}_part-mag_gre_MPM')
    mpm_fmap_phase  = create_key('sub-{subject}/fmap/sub-{subject}_desc-fmap_echo-{item:01d}_MPM')
    
    
    # create `info` dict
    info = {
            t1w:[],
            fieldmap_se_ap:[], fieldmap_se_pa:[],
            task_rest:[], task_rest_sbref:[],
            task_badaga:[], task_badaga_sbref:[],
            task_alice:[], task_alice_sbref:[],
            dwi:[], dwi_dist_ap:[], dwi_dist_pa:[],
            mpm_t1w:[], mpm_pd:[], mpm_mt:[], 
            mpm_b1:[], mpm_fmap_mag:[], mpm_fmap_phase:[],
            }
    
    for s in seqinfo:
        # MPRAGE anatomy run
        if ('T1w' in s.series_id):
            if ('NORM' in s.image_type):
                info[t1w] = [s.series_id]
                
        # spin echo field maps
        if ('SpinEchoFieldMap' in s.series_description):
            if ('AP' in s.series_description):
                info[fieldmap_se_ap].append(s.series_id)
            elif ('PA' in s.series_description):
                info[fieldmap_se_pa].append(s.series_id)               
        
        # fMRI task: rest
        if ('REST' in s.series_description):
            if ('SBRef' in s.series_description):
                info[task_rest_sbref].append(s.series_id)
            elif (s.dim4 > 100): # a full run should have over 100 volumes
                    info[task_rest].append(s.series_id)

        # fMRI task: badaga
        if ('BADAGA' in s.series_description):
            if ('SBRef' in s.series_description):
                    info[task_badaga_sbref].append(s.series_id)
            elif (s.dim4 > 100): # a full run should have over 100 volumes
                    info[task_badaga].append(s.series_id)

        # fMRI task: alice
        if ('ALICE' in s.series_description):
            if ('SBRef' in s.series_description):
                    info[task_alice_sbref].append(s.series_id)
            elif (s.dim4 > 100): # a full run should have over 100 volumes
                    info[task_alice].append(s.series_id)

        # dMRI
        if ('dMRI' in s.protocol_name):
            if ('DistortionMap_AP' in s.series_description):
                info[dwi_dist_ap].append(s.series_id)
            if ('DistortionMap_PA' in s.series_description):
                info[dwi_dist_pa].append(s.series_id)
            elif (s.dim4 > 100): # a full run should have over 100 volumes
                info[dwi].append(s.series_id)
        
        # MPM for hMRI-Toolbox
        if ('mfc_seste_b1map' in s.protocol_name):
            info[mpm_b1].append(s.series_id)
        
        if ('gre_field_mapping' in s.series_description):
            if 'M' in s.image_type:
                info[mpm_fmap_mag].append(s.series_id)
            elif 'P' in s.image_type:
                info[mpm_fmap_phase].append(s.series_id)
        if ('NORM' in s.image_type):
            if ('3dflash_t1W' in s.series_description):
                info[mpm_t1w].append(s.series_id)
            if ('3dflash_MT' in s.series_description):
                info[mpm_mt].append(s.series_id)
            if ('3dflash_PD' in s.series_description):
                info[mpm_pd].append(s.series_id)
            
    return info

