#!/usr/bin/env python

import gc
import gdal
from gdalconst import GA_ReadOnly
import logging
import numpy
import sys


_log = logging.getLogger()

# IS_MAC = False
#
# INDEX_FILE = "/g/data/u46/sjo/tmp/time_series_stats_optimisation/LS57_123_-025.txt"
#
# MB_FACTOR = 1024


def run():

    # log_mem("Before allocate array")
    #
    # pixels = numpy.random.rand(4000, 4000, 6)
    # _log.info("pixels takes up [%d] MB", pixels.nbytes / 1024 / 1024)
    # log_mem("After allocated array")
    #
    # del pixels
    #
    # log_mem("After del pixels")
    #
    # return

    log_mem("Starting")

    # data = numpy.zeros((1000, 4000, 4000), dtype=numpy.int16)
    # # data = numpy.random.rand(1000, 4000, 4000)
    # _log.info("data takes up [%d] MB", data.nbytes / 1024 / 1024)
    #
    # log_mem("About to del data")
    #
    # del data

    # data = numpy.empty((4000, 4000), dtype=numpy.int16)
    # _log.info("data takes up [%d] MB", data.nbytes / 1024 / 1024)

    stack = list()
    log_mem("Allocated list")

    projection = None
    transform = None

    with open(INDEX_FILE, "r") as index:
        for f in index:
        # for f in index.readlines()[:100]:
            path = f.rstrip("\n")

            if IS_MAC:
                path = path.replace("/g/data1/rs0/tiles/EPSG4326_1deg_0.00025pixel/",
                                    "/Volumes/Seagate Expansion Drive/data/cube/tiles/EPSG4326_1deg_0.00025pixel/")

            _log.info("Processing [%s]", path)

            raster = gdal.Open(path, GA_ReadOnly)

            band = raster.GetRasterBand(1)

            # data = band.ReadAsArray()

            data = band.ReadAsArray(0, 0, 500, 4000)

            _log.info("data takes up [%d] MB", data.nbytes / 1024 / 1024)

            # data = numpy.ma.masked_equal(data, -999, copy=False)

            log_mem("After read before del")

            # band.FlushCache()
            # raster.FlushCache()

            stack.append(data)

            if not projection:
                projection = raster.GetProjection()
                transform = raster.GetGeoTransform()

            del data, band, raster
            # data = band = raster = None

            # # band.ReadAsArray(0, 0, 4000, 4000, buf_obj=data)
            # # _log.info("data takes up [%d] MB", data.nbytes / 1024 / 1024)
            # #
            # # log_mem("After read before del")
            # #
            # # band.FlushCache()
            # # raster.FlushCache()
            #
            # del band, raster

            # gc.collect()

            log_mem("After del")

    #stack2 = numpy.ma.masked_equal(stack, -999, copy=False)

    #stack = numpy.memmap("x", stack2.dtype, mode="w+", shape=numpy.shape(stack2))

    #stack[:] = stack2[:]

    #del stack2

    stack = numpy.ma.masked_equal(stack, -999, copy=False)

    log_mem("Before calculate statistic")

    # summary = numpy.ma.count(stack, axis=0)
    # summary = numpy.min(stack, axis=0)
    # summary = numpy.max(stack, axis=0)
    # summary = numpy.mean(stack, axis=0)
    # summary = numpy.median(stack, axis=0)
    # summary = numpy.sum(stack, axis=0)

    log_mem("Before convert to float16")
    stack = numpy.ndarray.astype(stack, dtype=numpy.float16, copy=False)
    log_mem("Before fill with NaN")
    stack = stack.filled(numpy.nan)
    log_mem("Before percentile")
    summary = numpy.nanpercentile(stack, 0.95, axis=0, interpolation='lower')
    log_mem("Before convert back to int16")
    summary = numpy.ndarray.astype(summary, dtype=numpy.int16, copy=False)
    log_mem("Done - about to del stack")

#    log_mem("Before sort")
#    #stack = numpy.ma.sort(stack, axis=0)
#    numpy.ndarray.sort(stack, axis=0)
#    log_mem("Before calculate index")
#    index = numpy.ma.floor(numpy.ma.count(stack, axis=0) * 0.95).astype(numpy.int16)
#    log_mem("Before flatten index")
#    index_flat = index.ravel() * index.size + numpy.arange(index.size)
#    log_mem("Before extract using index")
#    summary = stack.ravel()[index_flat].reshape(numpy.shape(index))
#    log_mem("After calculate statistic")

    del stack

    log_mem("After calculate statistic and del")

    driver = gdal.GetDriverByName("GTiff")

    raster = driver.Create("out.tif", 4000, 4000, 1, gdal.GDT_Int16)

    raster.SetProjection(projection)
    raster.SetGeoTransform(transform)
    #raster.SetMetadata({"STATISTIC": "COUNT"})
    raster.SetMetadata({"STATISTIC": "PERCENTILE 95"})

    band = raster.GetRasterBand(1)

    band.SetNoDataValue(-999)
    band.SetMetadataItem("BAND_ID", "COUNT")
    band.WriteArray(summary)
    band.ComputeStatistics(True)

    band.FlushCache()
    raster.FlushCache()

    # del summary, stack
    del summary

    del band, raster

    log_mem("Finished")


# def log_mem(s=None):
#     if IS_MAC:
#         import resource
#
#         if s and len(s) > 0:
#             _log.debug(s)
#
#         _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB_FACTOR)
#     else:
#         import psutil

def log_mem(s=None):

    if s and len(s) > 0:
        _log.debug(s)

    import psutil

    _log.debug("Current memory usage is [%s]", psutil.Process().memory_info())
    _log.debug("Current memory usage is [%d] MB", psutil.Process().memory_info().rss / 1024 / 1024)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    global IS_MAC, INDEX_FILE, MB_FACTOR

    if sys.argv[1] == "--mac":
        _log.info("Running on mac so adjusting stuff")
        IS_MAC = True
        INDEX_FILE = "/Users/simon/tmp/LS57_123_-025.txt"
        MB_FACTOR = 1024 * 1024
    else:
        IS_MAC = False
        INDEX_FILE = "/g/data/u46/sjo/tmp/time_series_stats_optimisation/LS57_123_-025.txt"
        MB_FACTOR = 1024

    run()
