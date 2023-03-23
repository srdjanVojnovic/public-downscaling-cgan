import argparse
import numpy as np
from datetime import datetime as dt
import netCDF4 as nc
import xarray
import tensorflow as tf
import os
import shutil
import glob
import sys

def convert_to_datetime(ds):
    conversion = []
    for x in ds['time_bnds'].data:
        print(x[0].day)
        x1 = dt.timestamp(dt.strptime(str(x[0]), '%Y-%m-%d %H:%M:%S'))
        x2 = dt.timestamp(dt.strptime(str(x[1]), '%Y-%m-%d %H:%M:%S'))
        conversion.append(np.array([x1, x2]))
    return np.array(conversion)


# from generator method
def load_nc_dir_with_generator(dir_):
    def gen():
        for file in glob.glob(os.path.join(dir_, "precip_combined.nc")):
            ds = xarray.open_dataset(file, engine='netcdf4')
            dataarray = xarray.DataArray(convert_to_datetime(ds), coords={'time': ds['time'].values},
                                         dims=['time', 'bnds'])
            ds['time_bnds'] = dataarray
            yield {key: tf.convert_to_tensor(val) for key, val in ds.items()}

    sample = next(iter(gen()))

    return tf.data.Dataset.from_generator(
        gen,
        output_signature={
            key: tf.TensorSpec(val.shape, dtype=val.dtype)
            for key, val in sample.items()
        }
    )


def get_nc(dir_):
    ds = xarray.open_dataset(dir_ + "/" + "train.nc", engine='netcdf4')
    # dataarray = xarray.DataArray(convert_to_datetime(ds), coords={'time': ds['time'].values}, dims=['time', 'bnds'])
    # ds['time_bnds'] = dataarray
    return ds


def combine_nc():
    ds = xarray.open_mfdataset("../data/*.nc", combine='nested', concat_dim="time")
    print(ds)
    ds.to_netcdf("../data/precip_combined.nc")
    print(len(ds))

def build_generator_input(ds, i):
    max = 0.0027274378
    generator_input = []
    columns = ['psl', 'temp250', 'temp500', 'temp700', 'temp850', 'temp925', 'vorticity250', 'vorticity500', 'vorticity700', 'vorticity850', 'vorticity925']
    for column in columns:
        y = ds[column][i][2:62, 2:62].coarsen({"grid_latitude": 6, "grid_longitude": 6}).mean().data
        y = (max / (y.max() - y.min())) * (y - y.max()) + max # (y - y.min()) / (y.max() - y.min())
        generator_input.append(y)
    return np.transpose(np.array(generator_input))


def load_nc_dir_cached_to_tfrecord(dir_, tf_path):
    """Save data to tfRecord, open it, and deserialize

    Note that tfrecords are not that complicated! The simply store some
    bytes, and you can serialize data into those bytes however you find
    convenient. In this case, I serialise with `tf.io.serialize_tensor` and
    deserialize with `tf.io.parse_tensor`. No need for `tf.train.Example` or any
    of the other complexities mentioned in the official tutorial.

    """
    # generator_tfds = load_nc_dir_with_generator(dir_)
    writer = tf.io.TFRecordWriter(tf_path + "/" + "train.tfrecords")
    ds = get_nc(dir_)
    # for i in range(len(ds['target_pr'].data)):
    #     record_bytes = tf.train.Example(features=tf.train.Features(feature={
    #         "generator_input": tf.train.Feature(
    #             bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(build_generator_input(ds, i)).numpy()])),
    #         "constants": tf.train.Feature(
    #             bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(np.zeros((60, 60, 1))).numpy()])),
    #         "generator_output": tf.train.Feature(
    #             bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(ds['target_pr'].data[i][2:62, 2:62]).numpy()])
    #         )
    #     }))

    for i in range(len(ds['target_pr'].data)):
        y = ds['target_pr'].data[i][2:62, 2:62].reshape(-1)
        # y = (y - y.min()) / (y.max() - y.min())
        # y = np.log10(1+y)
        record_bytes = tf.train.Example(features=tf.train.Features(feature={
            "generator_input": tf.train.Feature(
                float_list=tf.train.FloatList(value=build_generator_input(ds, i).reshape(-1))),
            "constants": tf.train.Feature(
                float_list=tf.train.FloatList(value=np.zeros((60, 60, 2)).reshape(-1))),
            "generator_output": tf.train.Feature(
                float_list=tf.train.FloatList(value=y)
            )
        }))
        writer.write(record_bytes.SerializeToString())
    # writer.write(generator_tfds.map(lambda x: tf.io.serialize_tensor(x["target_pr"])))

    # return tf.data.TFRecordDataset(tf_path + "/" + "test.tfrecord").map(
    #     lambda x: tf.io.parse_tensor(x, tf.float64))

    return "Testing"


# Read TFRecord file
def _parse_tfr_element(element):
    # features = {
    #     'generator_input': tf.io.FixedLenFeature([], tf.string),
    #     'constants': tf.io.FixedLenFeature([], tf.string),
    #     'generator_output': tf.io.FixedLenFeature([], tf.string)
    # }
    features = {
        'generator_input': tf.io.FixedLenFeature((10, 10, 11), tf.float32),
        'constants': tf.io.FixedLenFeature((60, 60, 2), tf.float32),
        'generator_output': tf.io.FixedLenFeature((60, 60, 1), tf.float32)
    }
    example_message = tf.io.parse_single_example(element, features)
    print(example_message)
    b_feature = example_message['generator_output']
    # feature = tf.io.parse_tensor(b_feature, out_type=tf.float32)  # restore 2D array from byte string
    # print(feature)
    return b_feature


def get_ds(dir_):
    ds = xarray.open_dataset(dir_ + "/" + "train.nc", engine='netcdf4')
    return ds


def check_conversion(dir_):
    ds = get_ds(dir_)
    tfr_dataset = tf.data.TFRecordDataset(dir_ + "/" + "tfrecords" + "/" + "train.tfrecords")
    dataset = tfr_dataset.map(_parse_tfr_element)
    i = 0
    for instance in dataset:
        assert (type(instance.numpy()) == type(ds['target_pr'].data[i]))
        assert (instance.numpy().shape == ds['target_pr'].data[i][2:62, 2:62].reshape((60, 60, 1)).shape)
        assert (np.array_equal(instance.numpy(), ds['target_pr'].data[i][2:62, 2:62].reshape((60, 60, 1))))
        i += 1
    print("Conversion from netcdf to TFRecords complete")


if __name__ == "__main__":
    print("Tensorflow: ", tf.version.VERSION)
    print("Xarray: ", xarray.__version__)
    print("netCDF4: ", nc.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--netpath", help="Path to netcdf folder")
    parser.add_argument('--tfpath', help="Path to tfrecords folder")
    args = parser.parse_args()

    # combine_nc()
    # print(load_nc_dir_with_generator(args.netpath))
    print(load_nc_dir_cached_to_tfrecord(args.netpath, args.tfpath))
    check_conversion(args.netpath)
