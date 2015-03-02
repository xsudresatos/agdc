#!/usr/bin/env python
from datetime import datetime

import gc
import os
import gdal
from gdalconst import GA_ReadOnly
import logging
import numpy
import sys
from datacube import config
from datacube.api.model import Satellite, DatasetType
from datacube.api.query import list_tiles_as_list
from datacube.config import Config


_log = logging.getLogger()

# IS_MAC = False
#
# INDEX_FILE = "/g/data/u46/sjo/tmp/time_series_stats_optimisation/LS57_123_-025.txt"
#
# MB_FACTOR = 1024


def run():

    stack = list()
    stack_data = list()
    stack_data_pqa = list()
    stack_data_pqa_water = list()

    stack_maskable = list()

    # with open(INDEX_FILE, "r") as index:
    #     for f in index:
    #     # for f in index.readlines()[:10]:
    #         path = f.rstrip("\n")

    config = Config(os.path.expanduser("~/.datacube/config"))
    _log.debug(config.to_str())

    for tile in list_tiles_as_list(x=[120], y=[-20], acq_min=datetime(1980,1,1), acq_max=datetime(2020,12,31),
                                   satellites=[Satellite.LS5, Satellite.LS7],
                                   datasets=[DatasetType.ARG25],
                                   database=config.get_db_database(),
                                   user=config.get_db_username(),
                                   password=config.get_db_password(),
                                   host=config.get_db_host(), port=config.get_db_port()):

        _log.info("Processing [%s]", tile.datasets[DatasetType.FC25].path)

        bare_soil, pqa, water = get_pixel_values(tile)

        # Everything goes into stack
        stack.append(bare_soil)

        maskable_value = bare_soil

        if bare_soil == -999:
            _log.info("Skipping no data BARE SOIL value")
            maskable_value = -999
        else:
            # Everything with data goes into stack_data
            stack_data.append(bare_soil)

            # Everything with data plus CLEAR pixels go into stack_data_pqa
            if pqa != 16383:
                _log.info("Skipping not CLEAR according to PQA pixel [PQA=%d]", pqa)
                maskable_value = -999
            else:
                stack_data_pqa.append(bare_soil)

            # Everything with data plus CLEAR and NOT WATER pixels go into stack_data_pqa_water
            if pqa != 16383 or water == 128:
                _log.info("Skipping not CLEAR according to PQA or WATER according to WOFS pixel [PQA=%d WOFS=%s]", pqa, water)
                maskable_value = -999
            else:
                stack_data_pqa_water.append(bare_soil)

        stack_maskable.append(maskable_value)

    stack = numpy.array(stack)
    stack_data = numpy.array(stack_data)
    stack_data_pqa = numpy.array(stack_data_pqa)
    stack_data_pqa_water = numpy.array(stack_data_pqa_water)
    stack_maskable = numpy.ma.masked_equal(stack_maskable, -999)

    _log.info("stack is [%s] [%s]", numpy.shape(stack), stack)
    _log.info("stack_data is [%s] [%s]", numpy.shape(stack_data), stack_data)
    _log.info("stack_data_pqa is [%s] [%s]", numpy.shape(stack_data_pqa), stack_data_pqa)
    _log.info("stack_data_pqa_water is [%s] [%s]", numpy.shape(stack_data_pqa_water), stack_data_pqa_water)
    _log.info("stack_maskable is [%s] [%s]", numpy.shape(stack_maskable), stack_maskable)

    dump_stats(stack, s="STACK")
    dump_stats(stack_data, s="STACK DATA")
    dump_stats(stack_data_pqa, s="STACK DATA PQA")
    dump_stats(stack_data_pqa_water, s="STACK DATA PQA WATER")

    dump_stats_maskable(stack_maskable, "STACK MASKABLE")

    # for d in numpy.nditer(stack_data_pqa):
    #     print d


def dump_stats(data, s=None):
    count = numpy.shape(data)[0]
    mmin = numpy.min(data)
    mmax = numpy.max(data)
    mean = numpy.mean(data)

    # p25 = numpy.percentile(data, 25, interpolation='lower')
    # p50 = numpy.percentile(data, 50, interpolation='lower')
    # p75 = numpy.percentile(data, 75, interpolation='lower')
    # p90 = numpy.percentile(data, 90, interpolation='lower')
    # p95 = numpy.percentile(data, 95, interpolation='lower')

    p25 = numpy.percentile(data.astype(numpy.float16), 25, interpolation='lower').astype(numpy.int16)
    p50 = numpy.percentile(data.astype(numpy.float16), 50, interpolation='lower').astype(numpy.int16)
    p75 = numpy.percentile(data.astype(numpy.float16), 75, interpolation='lower').astype(numpy.int16)
    p90 = numpy.percentile(data.astype(numpy.float16), 90, interpolation='lower').astype(numpy.int16)
    p95 = numpy.percentile(data.astype(numpy.float16), 95, interpolation='lower').astype(numpy.int16)

    if s and len(s) > 0:
        _log.info("s")
    _log.info("\ncount = [%d]\nmin = [%d]\nmax=[%d]\nmean=[%d]\np25=[%d]\np50=[%d]\np75=[%d]\np90=[%d]\np95=[%d]", count, mmin, mmax, mean, p25, p50, p75, p90, p95)


def dump_stats_maskable(data, s=None):
    count = numpy.ma.count(data)
    mmin = numpy.min(data)
    mmax = numpy.max(data)
    mean = numpy.mean(data)
    masked_stack = numpy.ndarray.astype(data, dtype=numpy.float16).filled(numpy.nan)
    p25 = numpy.ndarray.astype(numpy.ma.masked_invalid(numpy.nanpercentile(masked_stack, 25, interpolation='lower')), dtype=numpy.int16)
    p50 = numpy.ndarray.astype(numpy.ma.masked_invalid(numpy.nanpercentile(masked_stack, 50, interpolation='lower')), dtype=numpy.int16)
    p75 = numpy.ndarray.astype(numpy.ma.masked_invalid(numpy.nanpercentile(masked_stack, 75, interpolation='lower')), dtype=numpy.int16)
    p90 = numpy.ndarray.astype(numpy.ma.masked_invalid(numpy.nanpercentile(masked_stack, 90, interpolation='lower')), dtype=numpy.int16)
    p95 = numpy.ndarray.astype(numpy.ma.masked_invalid(numpy.nanpercentile(masked_stack, 95, interpolation='lower')), dtype=numpy.int16)

    if s and len(s) > 0:
        _log.info("s")
    _log.info("\ncount = [%d]\nmin = [%d]\nmax=[%d]\nmean=[%d]\np25=[%d]\np50=[%d]\np75=[%d]\np90=[%d]\np95=[%d]", count, mmin, mmax, mean, p25, p50, p75, p90, p95)


def convert_path_fc_to_pqa(path):
    return path.replace("_FC_", "_PQA_")


def convert_path_fc_to_wofs(path):
    # /Volumes/Seagate Expansion Drive/data/cube/tiles/EPSG4326_1deg_0.00025pixel/LS5_TM/120_-020/1987/LS5_TM_FC_120_-020_1987-05-24T01-26-07.062063.tif
    # /Volumes/Seagate Expansion Drive/data/cube/tiles/EPSG4326_1deg_0.00025pixel/wofs_f7q/extents/120_-020/LS5_TM_WATER_120_-020_1987-05-24T01-26-07.062063.tif

    import os
    import glob

    filename = os.path.basename(path)
    filename = filename.replace("_FC_", "_WATER_")

    paths = glob.glob(os.path.join("/Volumes/Seagate Expansion Drive/data/cube/tiles/EPSG4326_1deg_0.00025pixel/wofs_f7q/extents/*", filename))

    if paths and len(paths) == 1:
        return paths[0]

    return None


    # TODO!!!!!

    # path = tile.datasets[DatasetType.FC25].path
    #
    # if IS_MAC:
    #     path = path.replace("/g/data1/rs0/tiles/EPSG4326_1deg_0.00025pixel/",
    #                         "/Volumes/Seagate Expansion Drive/data/cube/tiles/EPSG4326_1deg_0.00025pixel/")


def get_pixel_values(tile):
    return get_pixel_value_bare_soil(tile), get_pixel_value_pqa(tile), get_pixel_value_wofs(tile)


def get_pixel_value_bare_soil(tile):
    return get_pixel_value(tile.datasets[DatasetType.FC25].path)


def get_pixel_value_pqa(tile):
    return get_pixel_value(tile.datasets[DatasetType.PQ25].path)


def get_pixel_value_wofs(tile):
    # WOFS dataset is not always present
    if DatasetType.WATER in tile.datasets:
        return get_pixel_value(tile.datasets[DatasetType.WATER].path)

    return -888


def get_pixel_value(path, b=1, x=2020, y=3250, x_size=1, y_size=1):

    import os

    if not path or not os.path.exists(path):
        return -888

    raster = gdal.Open(path, GA_ReadOnly)

    band = raster.GetRasterBand(b)

    data = band.ReadAsArray(x, y, x_size, y_size)[0, 0]
    _log.info("data is [%s]", data)

    del band, raster

    return data


def log_mem(s=None):

    if s and len(s) > 0:
        _log.debug(s)

    import psutil

    _log.debug("Current memory usage is [%s]", psutil.Process().memory_info())
    _log.debug("Current memory usage is [%d] MB", psutil.Process().memory_info().rss / 1024 / 1024)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    global IS_MAC, INDEX_FILE, MB_FACTOR

    if len(sys.argv) > 1 and sys.argv[1] == "--mac":
        _log.info("Running on mac so adjusting stuff")
        IS_MAC = True
        INDEX_FILE = "/Users/simon/tmp/cube/time_series_statistics/LS57_120_-020.txt"
        MB_FACTOR = 1024 * 1024
    else:
        IS_MAC = False
        INDEX_FILE = "/g/data/u46/sjo/tmp/time_series_stats_optimisation/LS57_120_-020.txt"
        MB_FACTOR = 1024

    run()
