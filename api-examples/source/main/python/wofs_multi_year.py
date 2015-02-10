#!/usr/bin/env python

# ===============================================================================
# Copyright (c)  2014 Geoscience Australia
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# * Neither Geoscience Australia nor the names of its contributors may be
# used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================


__author__ = "Simon Oldfield"


import gdal
from gdalconst import *
import logging
import luigi
import numpy
import os


_log = logging.getLogger()


class WofsMultiYearSummaryTask(luigi.Task):
    x_min = luigi.IntParameter()
    x_max = luigi.IntParameter()

    y_min = luigi.IntParameter()
    y_max = luigi.IntParameter()

    year_min = luigi.IntParameter()
    year_max = luigi.IntParameter()

    input_directory = luigi.Parameter()
    output_directory = luigi.Parameter()

    def requires(self):
        x_list = range(self.x_min, self.x_max + 1)
        y_list = range(self.y_min, self.y_max + 1)

        import itertools

        return [WofsMultiYearCellTask(x=x, y=y,
                                      year_min=self.year_min, year_max=self.year_max,
                                      input_directory=self.input_directory,
                                      output_directory=self.output_directory)
                for x, y in itertools.product(x_list, y_list)]


class WofsMultiYearCellTask(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter()

    year_min = luigi.IntParameter()
    year_max = luigi.IntParameter()

    input_directory = luigi.Parameter()
    output_directory = luigi.Parameter()

    dummy = False

    def requires(self):
        return [WofsAnnualWaterSummaryTileTask(x=self.x, y=self.y, year=year, input_directory=self.input_directory) for year in range(self.year_min, self.year_max + 1)]

    def output(self):
        return luigi.LocalTarget(os.path.join(self.output_directory, "annualWaterSummary_{year_min:04d}_{year_max:04d}_{x:03d}_{y:04d}.tiff".format(year_min=self.year_min, year_max=self.year_max, x=self.x, y=self.y)))

    def run(self):
        if self.dummy:
            from datacube.api.workflow import dummy
            dummy(self.output().path)
            
        else:
            _log.debug("*** Actually doing it")

            transform = projection = None

            data = list()

            for i in self.input():
                _log.info("*** About to read %s", i.path)
                transform, projection, datum = get_data(i.path)
                data.append(datum)

            # Quick and dirty
            water_counts = numpy.array([item[0] for item in data])
            observation_counts = numpy.array([item[1] for item in data])

            _log.debug("*** shapes are %s and %s", numpy.shape(water_counts), numpy.shape(observation_counts))

            total_water_counts = numpy.sum(water_counts, axis=0)
            total_observation_counts = numpy.sum(observation_counts, axis=0)

            _log.debug("*** shapes are %s and %s", numpy.shape(total_water_counts), numpy.shape(total_observation_counts))

            _log.debug("*** transform = [%s] and projection = [%s]", transform, projection)

            _log.debug("2005-1000-1000 is [%d,%d]\n2006-1000-1000 is [%d,%d]\ntotal-1000-1000 is [%d,%d]\n",
                      data[0][0][1000][1000], data[0][1][1000][1000],
                      data[1][0][1000][1000], data[1][1][1000][1000],
                      total_water_counts[1000][1000], total_observation_counts[1000][1000])

            write_data(self.output().path, [total_water_counts, total_observation_counts], transform, projection, gdal.GDT_UInt16)


class WofsAnnualWaterSummaryTileTask(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    year = luigi.IntParameter()
    input_directory = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.input_directory, "annualWaterSummary_{year:04d}_{x:03d}_{y:04d}.tiff".format(year=self.year, x=self.x, y=self.y)))


class WofsMultiYearWorkflow():

    def __init__(self, application_name):
        self.application_name = application_name

        self.x_min = None
        self.x_max = None

        self.y_min = None
        self.y_max = None

        self.year_min = None
        self.year_max = None

        self.input_directory = None
        self.output_directory = None

        self.local_scheduler = None

    def parse_arguments(self):
        import argparse

        parser = argparse.ArgumentParser(prog=__name__, description=self.application_name)

        group = parser.add_mutually_exclusive_group()

        group.add_argument("--quiet", help="Less output", action="store_const", dest="log_level", const=logging.WARN)
        group.add_argument("--verbose", help="More output", action="store_const", dest="log_level", const=logging.DEBUG)

        parser.set_defaults(log_level=logging.INFO)

        parser.add_argument("--x-min", help="X grid reference", action="store", dest="x_min", type=int,
                            choices=range(110, 155+1), required=True, metavar="[110 - 155]")
        parser.add_argument("--x-max", help="X grid reference", action="store", dest="x_max", type=int,
                            choices=range(110, 155+1), required=True, metavar="[110 - 155]")

        parser.add_argument("--y-min", help="Y grid reference", action="store", dest="y_min", type=int,
                            choices=range(-45, -10+1), required=True, metavar="[-45 - -10]")
        parser.add_argument("--y-max", help="Y grid reference", action="store", dest="y_max", type=int,
                            choices=range(-45, -10+1), required=True, metavar="[-45 - -10]")

        parser.add_argument("--year-min", help="Year", action="store", dest="year_min", type=int, required=True)
        parser.add_argument("--year-max", help="Year", action="store", dest="year_max", type=int, required=True)

        from datacube.api.workflow import readable_dir, writeable_dir

        parser.add_argument("--input-directory", help="Input directory", action="store", dest="input_directory",
                            type=readable_dir, required=True)

        parser.add_argument("--output-directory", help="Output directory", action="store", dest="output_directory",
                            type=writeable_dir, required=True)

        parser.add_argument("--local-scheduler", help="Use local luigi scheduler rather than MPI", action="store_true",
                            dest="local_scheduler", default=False)

        args = parser.parse_args()

        _log.setLevel(args.log_level)

        self.x_min = args.x_min
        self.x_max = args.x_max

        self.y_min = args.y_min
        self.y_max = args.y_max

        self.year_min = args.year_min
        self.year_max = args.year_max

        self.input_directory = args.input_directory
        self.output_directory = args.output_directory

        self.local_scheduler = args.local_scheduler

        _log.info("""
        x = {x_min:03d} to {x_max:03d}
        y = {y_min:04d} to {y_min:04d}
        year = {year_min:04d} to {year_max:04d}
        input directory = {input_directory}
        output directory = {output_directory}
        scheduler = {scheduler}
        """.format(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max,
                   year_min=self.year_min, year_max=self.year_max,
                   input_directory=self.input_directory,
                   output_directory=self.output_directory,
                   scheduler=self.local_scheduler and "LOCAL" or "MPI"))

    def run(self):
        self.parse_arguments()

        tasks = [WofsMultiYearSummaryTask(x_min=self.x_min, x_max=self.x_max,
                                          y_min=self.y_min, y_max=self.y_max,
                                          year_min=self.year_min, year_max=self.year_max,
                                          input_directory=self.input_directory,
                                          output_directory=self.output_directory)]

        if self.local_scheduler:
            # Standard luigi
            import luigi
            luigi.build(tasks, local_scheduler=True)

        else:
            # Luigi MPI extension
            import luigi.contrib.mpi as mpi
            mpi.run(tasks)


def get_data(path):
    raster = gdal.Open(path, GA_ReadOnly)
    assert raster

    transform = raster.GetGeoTransform()
    projection = raster.GetProjection()

    data = list()

    for b in range(1, raster.RasterCount + 1):

        band = raster.GetRasterBand(b)
        assert band

        data.append(band.ReadAsArray())

    return transform, projection, data


def write_data(path, data, transform, projection, data_type, options=["INTERLEAVE=PIXEL"]):

    _log.debug("creating output raster %s", path)
    _log.debug("filename=%s | shape = %s | bands = %d | data type = %s", path,
               (numpy.shape(data[0])[0], numpy.shape(data[0])[1]), len(data), data_type)

    driver = gdal.GetDriverByName("GTiff")
    assert driver

    width = numpy.shape(data[0])[1]
    height = numpy.shape(data[0])[0]

    dataset = driver.Create(path, width, height, len(data), data_type, options)
    assert dataset

    dataset.SetGeoTransform(transform)
    dataset.SetProjection(projection)

    for i in range(0, len(data)):
        _log.debug("Writing band %d", i + 1)
        dataset.GetRasterBand(i + 1).WriteArray(data[i])
        dataset.GetRasterBand(i + 1).ComputeStatistics(True)

    dataset.FlushCache()

    dataset = None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    WofsMultiYearWorkflow("WOFS 2 Year").run()

