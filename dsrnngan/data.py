""" File for handling data loading and saving. """
import os
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

import read_config
from datetime import datetime as dt


data_paths = read_config.get_data_paths()
RADAR_PATH = data_paths["GENERAL"]["RADAR_PATH"]
FCST_PATH = data_paths["GENERAL"]["FORECAST_PATH"]
CONSTANTS_PATH = data_paths["GENERAL"]["CONSTANTS_PATH"]

all_fcst_fields = ['psl', 'temp250', 'temp500', 'temp700', 'temp850', 'temp925', 'vorticity250', 'vorticity500', 'vorticity700', 'vorticity850', 'vorticity925']
# all_fcst_fields = ['tp', 'cp', 'sp', 'tisr', 'cape', 'tclw', 'tcwv', 'u700', 'v700']
fcst_hours = np.array(range(24))


def denormalise(x):
    """
    Undo log-transform of rainfall.  Also cap at 500 (feel free to adjust according to application!)
    """
    return np.minimum(10**x - 1, 500.0)

def convert_dates(ds):
    conversion = []
    for x in ds['time'].data:
        x1 = dt.strptime(str(x), '%Y-%m-%d %H:%M:%S')
        conversion.append(x1)
    return np.array(conversion)

def get_dates(year):
    """
    Return dates where we have radar data
    """
    from glob import glob
    # file_paths = os.path.join(RADAR_PATH, str(year), "*.nc")
    file_paths = "../data/precip_combined.nc"
    files = glob(file_paths)
    dates = []
    for f in files:
        ds = xr.open_dataset(f, engine='netcdf4')
        dates = ds['time'].data
        # dates.append(f[:-3].split('_')[-1])
    return sorted(dates)

# CHANGE
def load_radar(date, hour, log_precip=False, aggregate=1):
    # year = date[:4]
    # data_path = os.path.join(RADAR_PATH, year, f"metoffice-c-band-rain-radar_uk_{date}.nc")
    data = xr.open_dataset("../data/precip_combined.nc")
    assert hour+aggregate < 25
    # y = np.array(data['unknown'][hour:hour+aggregate, :, :]).sum(axis=0)
    y = data["target_pr"][np.where(data["target_pr"]['time'].data == date)[0][0]][2:62, 2:62].data

    data.close()
    # The remapping of the NIMROD radar left a few negative numbers, so remove those
    y[y < 0.0] = 0.0
    if log_precip:
        return np.log10(1+y)
    else:
        return y


def logprec(y, log_precip=True):
    if log_precip:
        return np.log10(1+y)
    else:
        return y

def load_hires_constants(batch_size=1):
    # oro_path = os.path.join(CONSTANTS_PATH, "orography.nc")
    oro_path = "../data/precip_combined.nc"
    df = xr.load_dataset(oro_path)
    # LSM is already 0:1
    lsm = np.array(df['psl']) # NOT RIGHT
    lsm = np.zeros((60, 60))

    # Orography.  Clip below, to remove spectral artifacts, and normalise by max
    z = df['vorticity850'].data # NOT RIGHT
    z[z < 5] = 5
    z = z/z.max()
    z = np.zeros((60, 60))

    df.close()
    # print(z.shape, lsm.shape)

    return np.array([np.repeat(np.stack([z, lsm], -1), batch_size, axis=0)])


def load_fcst_radar_batch(batch_dates, fcst_fields=all_fcst_fields, log_precip=False,
                          constants=False, hour=0, norm=False):
    batch_x = []
    batch_y = []

    if type(hour) == str:
        if hour == 'random':
            hours = fcst_hours[np.random.randint(len(fcst_hours), size=[len(batch_dates)])]
        else:
            assert False, f"Not configured for {hour}"
    elif np.issubdtype(type(hour), np.integer):
        hours = len(batch_dates)*[hour]
    else:
        hours = hour

    for i, date in enumerate(batch_dates):
        h = hours[i]
        batch_x.append(load_fcst_stack(fcst_fields, date, h, log_precip=log_precip, norm=norm))
        batch_y.append(load_radar(date, h, log_precip=log_precip))

    if (not constants):
        return np.array(batch_x), np.array(batch_y)
    else:
        return [np.array(batch_x), load_hires_constants(len(batch_dates))], np.array(batch_y)

# CHANGE
def load_fcst(ifield, date, hour, log_precip=False, norm=False):
    # Get the time required (compensating for IFS forecast saving precip at the end of the timestep)
    # time = datetime(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:8]), hour=hour) + timedelta(hours=1)
    hour = 12
    # Get the correct forecast starttime
    # if time.hour < 6:
    #     tmpdate = time - timedelta(days=1)
    #     loaddate = datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=18)
    #     loadtime = '12'
    # elif 6 <= time.hour < 18:
    #     tmpdate = time
    #     loaddate = datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=6)
    #     loadtime = '00'
    # elif 18 <= time.hour < 24:
    #     tmpdate = time
    #     loaddate = datetime(year=tmpdate.year, month=tmpdate.month, day=tmpdate.day, hour=18)
    #     loadtime = '12'
    # else:
    #     assert False, "Not acceptable time"
    # dt = time - loaddate
    # time_index = int(dt.total_seconds()//3600)
    # assert time_index >= 1, "Cannot use first hour of retrival"
    # loaddata_str = loaddate.strftime("%Y%m%d") + '_' + loadtime

    field = ifield
    if field in ['u700', 'v700']:
        fleheader = 'winds'
        field = field[:1]
    elif field in ['cdir', 'tcrw']:
        fleheader = 'missing'
    else:
        fleheader = 'sfc'

    # ds_path = os.path.join(FCST_PATH, f"{fleheader}_{loaddata_str}.nc")
    ds = xr.open_dataset("../data/precip_combined.nc")
    data = ds[field][np.where(ds[field]['time'].data == date)[0][0]]
    field = ifield
    if field in ['tp', 'cp', 'cdir', 'tisr']:
        # data = data[time_index, :, :] - data[time_index-1, :, :]
        print("Skip")
    else:
        # data = data[time_index, :, :]
        print("Skip")

    y = data[2:62, 2:62].coarsen({"grid_latitude": 6, "grid_longitude": 6}).mean().data
    data.close()
    ds.close()
    return y
    if field in ['tp', 'cp', 'pr', 'prl', 'prc']:
        # print('pass')
        y[y < 0] = 0.
        y = 1000*y
    if log_precip and field in ['tp', 'cp', 'pr', 'prc', 'prl']:
        # precip is measured in metres, so multiply up
        return np.log10(1+y)  # *1000)
    elif norm:
        return (y-fcst_norm[field][0])/fcst_norm[field][1]
    else:
        return y


def load_fcst_stack(fields, date, hour, log_precip=False, norm=False):
    field_arrays = []
    for f in fields:
        field_arrays.append(load_fcst(f, date, hour, log_precip=log_precip, norm=norm))
    return np.stack(field_arrays, -1)

# CHANGE
def get_fcst_stats(field, year=2018):
    import datetime

    # create date objects
    begin_year = datetime.date(year, 1, 1)
    end_year = datetime.date(year, 12, 31)
    one_day = datetime.timedelta(days=1)
    next_day = begin_year

    mi = 0
    mx = 0
    mn = 0
    sd = 0
    nsamples = 0
    for day in range(0, 366):  # includes potential leap year
        if next_day > end_year:
            break
        for hour in fcst_hours:
            try:
                dta = load_fcst(field, next_day.strftime("%Y%m%d"), hour)
                mi = min(mi, dta.min())
                mx = max(mx, dta.max())
                mn += dta.mean()
                sd += dta.std()**2
                nsamples += 1
            except:  # noqa
                print(f"Problem loading {next_day.strftime('%Y%m%d')}, {hour}")
        next_day += one_day
    mn /= nsamples
    sd = (sd / nsamples)**0.5
    return mi, mx, mn, sd

# CHANGE
def gen_fcst_norm(year=2018):

    """
    One-off function, used to generate normalisation constants, which are used to normalise the various input fields for training/inference.

    Depending on the field, we may subtract the mean and divide by the std. dev., or just divide by the max observed value.
    """

    import pickle
    stats_dic = {}
    for f in all_fcst_fields:
        stats = get_fcst_stats(f, year)
        if f == 'sp':
            stats_dic[f] = [stats[2], stats[3]]
        elif f == "u700" or f == "v700":
            stats_dic[f] = [0, max(-stats[0], stats[1])]
        else:
            stats_dic[f] = [0, stats[1]]
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")
    with open(fcstnorm_path, 'wb') as f:
        pickle.dump(stats_dic, f, 0)
    return

# CHANGE
def load_fcst_norm(year=2018):
    import pickle
    fcstnorm_path = os.path.join(CONSTANTS_PATH, f"FCSTNorm{year}.pkl")
    with open(fcstnorm_path, 'rb') as f:
        return pickle.load(f)


try:
    fcst_norm = load_fcst_norm(2018)
except:  # noqa
    fcst_norm = None
