import numpy as np
import xarray as xr
from noise import NoiseGenerator
import data

def create_xarray(gen, data_gen, ens_size=4, noise_channels=4, num_cases=64):
    data_gen_iter = iter(data_gen)
    original_images = []
    generated_images = []

    for kk in range(num_cases):
        print(kk)
        inputs, outputs = next(data_gen_iter)
        cond = inputs['lo_res_inputs']
        const = inputs['hi_res_inputs']
        seq_real = outputs['output']
        seq_real = seq_real.reshape((60, 60))
        seq_real = data.denormalise(seq_real)
        original_images.append(seq_real)
        batch_size = cond.shape[0]
        seq_gen = []
        for ii in range(ens_size):
            noise_shape = cond[0, ..., 0].shape + (noise_channels,)
            noise_gen = NoiseGenerator(noise_shape, batch_size=batch_size)

            gen_image = gen.predict([cond, const, noise_gen()]).reshape((60, 60))
            seq_gen.append(gen_image)

        
        seq_gen = [data.denormalise(seq) for seq in seq_gen]
        generated_images.append(seq_gen)

    # Generate some example data
    # original_images = np.random.rand(720, 60, 60)
    original_images = np.array(original_images)
    print(original_images.shape)

    # generated_images = np.random.rand(720, 4, 60, 60)
    generated_images = np.array(generated_images)
    
    # Define the dimensions and coordinates for the xarray
    dims = ('original_image', 'generated_image', 'x', 'y')
    coords = {'original_image': np.arange(num_cases), 'generated_image': np.arange(ens_size), 'x': np.arange(60), 'y': np.arange(60)}

    # Create the xarray
    xarr_gen = xr.DataArray(
        generated_images,
        dims=dims,
        coords=coords,
    )

    xarr_gen.to_netcdf('../data/samples/normal/gen_last.nc')

    dims = ('original_image', 'x', 'y')
    coords = {'original_image': np.arange(num_cases), 'x': np.arange(60), 'y': np.arange(60)}
    
    xarr_real = xr.DataArray(
        original_images,
        dims=dims,
        coords=coords,
    )
    
    xarr_real.to_netcdf('../data/samples/normal/real_last.nc')
    