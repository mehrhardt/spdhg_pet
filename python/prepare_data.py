"""Load PET data and prepare it for reconstruction."""

from mMR import load_data

fdata_amyloid = '/home/me404/store/data/201611_PET_Pawel_amyloid'
fdata_fdg = '/home/me404/store/data/201712_PET_Pawel_fdg'

data, background, factors, image, image_mr, image_ct = load_data(
        fdata_amyloid, time=(3000, 3600))

data, background, factors, image, image_mr, image_ct = load_data(fdata_fdg)