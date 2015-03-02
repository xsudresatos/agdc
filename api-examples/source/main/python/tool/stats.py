#!/usr/bin/env python


import gdal
from gdalconst import GA_ReadOnly
import logging
import numpy
import sys


_log = logging.getLogger()


def run():

    log_mem("Starting")

    stack = list()
    log_mem("Allocated list")

    projection = None
    transform = None

    log_mem("About to create raster")

    driver = gdal.GetDriverByName("GTiff")

    rasterout = driver.Create("out.tif", 4000, 4000, 8, gdal.GDT_Int16)
    assert rasterout

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

            data = band.ReadAsArray(0, 0, 500, 4000)

            _log.info("data takes up [%d] MB", data.nbytes / 1024 / 1024)

            log_mem("After read before del")

            stack.append(data)

            if not projection:
                projection = raster.GetProjection()
                transform = raster.GetGeoTransform()

            del data, band, raster

            log_mem("After del")

    log_mem("About to set the meta data on the raster")

    rasterout.SetProjection(projection)
    rasterout.SetGeoTransform(transform)
    rasterout.SetMetadata({"STATISTIC": "COUNT MIN MAX MEAN MEDIAN PERCENTILE_75 PERCENTILE_90 PERCENTILE_95"})

    log_mem("About to apply mask to the stack")

    stack = numpy.ma.masked_equal(stack, -999, copy=False)

    log_mem("About to calculate COUNT")

    summary = numpy.ma.count(stack, axis=0)
    write_band(rasterout, 1, "COUNT", summary)

    summary = numpy.min(stack, axis=0)
    write_band(rasterout, 2, "MIN", summary)

    summary = numpy.max(stack, axis=0)
    write_band(rasterout, 3, "MAX", summary)

    summary = numpy.mean(stack, axis=0)
    write_band(rasterout, 4, "MEAN", summary)

    summary = numpy.median(stack, axis=0)
    write_band(rasterout, 5, "MEDIAN", summary)

    # summary = numpy.sum(stack, axis=0)
    # write_band(raster, 6, "SUM", summary)

    stack = numpy.ndarray.astype(stack, dtype=numpy.float16, copy=False)
    stack = stack.filled(numpy.nan)

    summary = numpy.nanpercentile(stack, 0.75, axis=0, interpolation='lower')
    summary = numpy.ndarray.astype(summary, dtype=numpy.int16, copy=False)
    write_band(rasterout, 6, "PERCENTILE_75", summary)

    summary = numpy.nanpercentile(stack, 0.90, axis=0, interpolation='lower')
    summary = numpy.ndarray.astype(summary, dtype=numpy.int16, copy=False)
    write_band(rasterout, 7, "PERCENTILE_90", summary)

    summary = numpy.nanpercentile(stack, 0.95, axis=0, interpolation='lower')
    summary = numpy.ndarray.astype(summary, dtype=numpy.int16, copy=False)
    write_band(rasterout, 8, "PERCENTILE_95", summary)

    log_mem("Done - about to del stack")

    del stack, summary, rasterout

    log_mem("Finished")


def write_band(raster, band_number, band_name, summary):

    band = raster.GetRasterBand(band_number)

    band.SetNoDataValue(-999)
    band.SetMetadataItem("BAND_ID", band_name)
    band.WriteArray(summary)
    band.ComputeStatistics(True)

    band.FlushCache()
    raster.FlushCache()


def log_mem(s=None):

    if s and len(s) > 0:
        _log.debug(s)

    import psutil

    _log.debug("Current memory usage is [%s]", psutil.Process().memory_info())
    _log.debug("Current memory usage is [%d] MB", psutil.Process().memory_info().rss / 1024 / 1024)

    import resource

    _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MB_FACTOR)


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
