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
import gc


__author__ = "Simon Oldfield"

import argparse
import gdal
import numpy
from gdalconst import GA_Update
import logging
import os
import resource
from datetime import datetime, timedelta
from datacube.api.model import DatasetType, Satellite, get_bands, dataset_type_database, dataset_type_derived_nbar
from datacube.api.query import list_tiles_as_list
from datacube.api.utils import PqaMask, get_dataset_metadata, get_dataset_data, get_dataset_data_with_pq, empty_array, \
    WofsMask, union, BYTE_MAX, get_mask_pqa, get_mask_wofs, get_dataset_data_masked
from datacube.api.utils import NDV, UINT16_MAX
from datacube.api.workflow import writeable_dir
from datacube.config import Config
from enum import Enum


_log = logging.getLogger()


def satellite_arg(s):
    if s in Satellite._member_names_:
        return Satellite[s]
    raise argparse.ArgumentTypeError("{0} is not a supported satellite".format(s))


def pqa_mask_arg(s):
    if s in PqaMask._member_names_:
        return PqaMask[s]
    raise argparse.ArgumentTypeError("{0} is not a supported PQA mask".format(s))


def wofs_mask_arg(s):
    if s in WofsMask._member_names_:
        return WofsMask[s]
    raise argparse.ArgumentTypeError("{0} is not a supported WOFS mask".format(s))


def dataset_type_arg(s):
    if s in DatasetType._member_names_:
        return DatasetType[s]
    raise argparse.ArgumentTypeError("{0} is not a supported dataset type".format(s))


def statistic_arg(s):
    if s in Statistic._member_names_:
        return Statistic[s]
    raise argparse.ArgumentTypeError("{0} is not a supported statistic".format(s))


class Statistic(Enum):
    __order__ = "COUNT MIN MAX MEAN MEDIAN SUM STANDARD_DEVIATION VARIANCE PERCENTILE_25 PERCENTILE_50 PERCENTILE_75 PERCENTILE_90 PERCENTILE_95"

    COUNT = "COUNT"
    MIN = "MIN"
    MAX = "MAX"
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    SUM = "SUM"
    STANDARD_DEVIATION = "STANDARD_DEVIATION"
    VARIANCE = "VARIANCE"
    PERCENTILE_25 = "PERCENTILE_25"
    PERCENTILE_50 = "PERCENTILE_50"
    PERCENTILE_75 = "PERCENTILE_75"
    PERCENTILE_90 = "PERCENTILE_90"
    PERCENTILE_95 = "PERCENTILE_95"


class SummariseDatasetTimeSeriesStatistics():
    application_name = None

    x = None
    y = None

    acq_min = None
    acq_max = None

    process_min = None
    process_max = None

    ingest_min = None
    ingest_max = None

    satellites = None

    mask_pqa_apply = None
    mask_pqa_mask = None

    mask_wofs_apply = None
    mask_wofs_mask = None

    dataset_type = None

    bands = None

    output_directory = None
    overwrite = None
    list_only = None

    statistics = None
    percentiles = None

    chunk_size_x = None
    chunk_size_y = None

    def __init__(self, application_name):
        self.application_name = application_name

    def parse_arguments(self):
        parser = argparse.ArgumentParser(prog=__name__, description=self.application_name)

        group = parser.add_mutually_exclusive_group()

        group.add_argument("--quiet", help="Less output", action="store_const", dest="log_level", const=logging.WARN)
        group.add_argument("--verbose", help="More output", action="store_const", dest="log_level", const=logging.DEBUG)

        parser.set_defaults(log_level=logging.INFO)

        parser.add_argument("--x", help="X grid reference", action="store", dest="x",
                            type=int, choices=range(110, 155 + 1), required=True, metavar="[110 - 155]")
        parser.add_argument("--y", help="Y grid reference", action="store", dest="y",
                            type=int, choices=range(-45, -10 + 1), required=True, metavar="[-45 - -10]")

        parser.add_argument("--acq-min", help="Acquisition Date (YYYY or YYYY-MM or YYYY-MM-DD)", action="store",
                            dest="acq_min", type=str, default="1980")
        parser.add_argument("--acq-max", help="Acquisition Date (YYYY or YYYY-MM or YYYY-MM-DD)", action="store",
                            dest="acq_max", type=str, default="2020")

        # parser.add_argument("--process-min", help="Process Date", action="store", dest="process_min", type=str)
        # parser.add_argument("--process-max", help="Process Date", action="store", dest="process_max", type=str)
        #
        # parser.add_argument("--ingest-min", help="Ingest Date", action="store", dest="ingest_min", type=str)
        # parser.add_argument("--ingest-max", help="Ingest Date", action="store", dest="ingest_max", type=str)

        parser.add_argument("--satellite", help="The satellite(s) to include", action="store", dest="satellite",
                            type=satellite_arg, nargs="+", choices=Satellite, default=[Satellite.LS5, Satellite.LS7],
                            metavar=" ".join([s.name for s in Satellite]))

        parser.add_argument("--mask-pqa-apply", help="Apply PQA mask", action="store_true", dest="mask_pqa_apply",
                            default=False)
        parser.add_argument("--mask-pqa-mask", help="The PQA mask to apply", action="store", dest="mask_pqa_mask",
                            type=pqa_mask_arg, nargs="+", choices=PqaMask, default=[PqaMask.PQ_MASK_CLEAR],
                            metavar=" ".join([s.name for s in PqaMask]))

        parser.add_argument("--mask-wofs-apply", help="Apply WOFS mask", action="store_true", dest="mask_wofs_apply",
                            default=False)
        parser.add_argument("--mask-wofs-mask", help="The WOFS mask to apply", action="store", dest="mask_wofs_mask",
                            type=wofs_mask_arg, nargs="+", choices=WofsMask, default=[WofsMask.WET],
                            metavar=" ".join([s.name for s in WofsMask]))

        supported_dataset_types = union(dataset_type_database, dataset_type_derived_nbar)

        parser.add_argument("--dataset-type", help="The type of dataset to retrieve", action="store",
                            dest="dataset_type",
                            type=dataset_type_arg,
                            choices=supported_dataset_types, default=DatasetType.ARG25,
                            metavar=" ".join([s.name for s in supported_dataset_types]))

        # TODO this is a quick and dirty
        parser.add_argument("--band", help="The band(s) for which to calculate statistics", action="store",
                            dest="bands", type=int, nargs="+")

        parser.add_argument("--output-directory", help="Output directory", action="store", dest="output_directory",
                            type=writeable_dir, required=True)

        parser.add_argument("--overwrite", help="Over write existing output file", action="store_true",
                            dest="overwrite", default=False)

        parser.add_argument("--list-only", help="List the datasets that would be retrieved rather than retrieving them",
                            action="store_true", dest="list_only", default=False)

        supported_statistics = [
            Statistic.COUNT,
            Statistic.MIN,
            Statistic.MAX,
            Statistic.MEAN,
            Statistic.MEDIAN,
            Statistic.SUM,
            Statistic.STANDARD_DEVIATION,
            Statistic.VARIANCE,
            Statistic.PERCENTILE_25,
            Statistic.PERCENTILE_50,
            Statistic.PERCENTILE_75,
            Statistic.PERCENTILE_90,
            Statistic.PERCENTILE_95]

        parser.add_argument("--statistic", help="The statistic(s) to calculate", action="store",
                            dest="statistics",
                            type=statistic_arg,
                            nargs="+",
                            choices=supported_statistics,
                            default=supported_statistics,
                            metavar=" ".join([s.name for s in supported_statistics]))

        # parser.add_argument("--percentile", help="The percentile value(s) calculate", action="store",
        #                     dest="percentile",
        #                     type=int,
        #                     nargs="+",
        #                     choices=range(0, 100 + 1), required=False, default=[25, 50, 75, 90, 95],
        #                     metavar="[0 - 100]")

        parser.add_argument("--chunk-size-x", help="Number of X pixels to process at once", action="store",
                            dest="chunk_size_x", type=int, choices=range(0, 4000 + 1), default=4000,
                            metavar="[1 - 4000]")
        parser.add_argument("--chunk-size-y", help="Number of Y pixels to process at once", action="store",
                            dest="chunk_size_y", type=int, choices=range(0, 4000 + 1), default=4000,
                            metavar="[1 - 4000]")

        args = parser.parse_args()

        _log.setLevel(args.log_level)

        self.x = args.x
        self.y = args.y

        def parse_date_min(s):
            from datetime import datetime

            if s:
                if len(s) == len("YYYY"):
                    return datetime.strptime(s, "%Y").date()

                elif len(s) == len("YYYY-MM"):
                    return datetime.strptime(s, "%Y-%m").date()

                elif len(s) == len("YYYY-MM-DD"):
                    return datetime.strptime(s, "%Y-%m-%d").date()

            return None

        def parse_date_max(s):
            from datetime import datetime
            import calendar

            if s:
                if len(s) == len("YYYY"):
                    d = datetime.strptime(s, "%Y").date()
                    d = d.replace(month=12, day=31)
                    return d

                elif len(s) == len("YYYY-MM"):
                    d = datetime.strptime(s, "%Y-%m").date()

                    first, last = calendar.monthrange(d.year, d.month)
                    d = d.replace(day=last)
                    return d

                elif len(s) == len("YYYY-MM-DD"):
                    d = datetime.strptime(s, "%Y-%m-%d").date()
                    return d

            return None

        self.acq_min = parse_date_min(args.acq_min)
        self.acq_max = parse_date_max(args.acq_max)

        # self.process_min = parse_date_min(args.process_min)
        # self.process_max = parse_date_max(args.process_max)
        #
        # self.ingest_min = parse_date_min(args.ingest_min)
        # self.ingest_max = parse_date_max(args.ingest_max)

        self.satellites = args.satellite

        self.mask_pqa_apply = args.mask_pqa_apply
        self.mask_pqa_mask = args.mask_pqa_mask

        self.mask_wofs_apply = args.mask_wofs_apply
        self.mask_wofs_mask = args.mask_wofs_mask

        self.dataset_type = args.dataset_type

        self.bands = args.bands

        self.output_directory = args.output_directory
        self.overwrite = args.overwrite
        self.list_only = args.list_only

        self.statistics = args.statistics

        self.chunk_size_x = args.chunk_size_x
        self.chunk_size_y = args.chunk_size_y

        _log.info("""
        x = {x:03d}
        y = {y:04d}
        acq = {acq_min} to {acq_max}
        process = {process_min} to {process_max}
        ingest = {ingest_min} to {ingest_max}
        satellites = {satellites}
        PQA mask = {pqa_mask}
        WOFS mask = {wofs_mask}
        dataset to retrieve = {dataset_type}
        bands = {bands}
        output directory = {output}
        over write existing = {overwrite}
        list only = {list_only}
        statistics = {statistics}
        chunk size = {chunk_size_x:4d} x {chunk_size_y:4d} pixels
        """.format(x=self.x, y=self.y,
                   acq_min=self.acq_min, acq_max=self.acq_max,
                   process_min=self.process_min, process_max=self.process_max,
                   ingest_min=self.ingest_min, ingest_max=self.ingest_max,
                   satellites=" ".join([satellite.name for satellite in self.satellites]),
                   pqa_mask=self.mask_pqa_apply and " ".join([mask.name for mask in self.mask_pqa_mask]) or "",
                   wofs_mask=self.mask_wofs_apply and " ".join([mask.name for mask in self.mask_wofs_mask]) or "",
                   dataset_type=decode_dataset_type(self.dataset_type),
                   bands=self.bands,
                   output=self.output_directory,
                   overwrite=self.overwrite,
                   list_only=self.list_only,
                   statistics=" ".join([statistic.name for statistic in self.statistics]),
                   chunk_size_x=self.chunk_size_x,
                   chunk_size_y=self.chunk_size_y))

    # The basic logic here is:
    #
    # for each chunk:
    # for each band:
    #       for each tile
    #           read data for chunk of tile and append to time series array
    #
    #       for each statistic:
    #           calculate statistic over time series array
    #           write statistic to output file

    def run(self):
        self.parse_arguments()

        config = Config(os.path.expanduser("~/.datacube/config"))
        _log.debug(config.to_str())

        # TODO
        dataset_bands = get_bands(self.dataset_type, self.satellites[0])

        # Verify that all the requested satellites have the same band combinations
        for satellite in self.satellites:
            if dataset_bands != get_bands(self.dataset_type, satellite):
                _log.error("Satellites [%s] have differing bands", " ".join([satellite.name for satellite in self.satellites]))
                raise Exception("Satellites with different band combinations selected")

        # Which bands are we going to process - default is all unless they have specified a subset

        bands = []

        dataset_bands_list = list(dataset_bands)

        if not self.bands:
            bands = dataset_bands_list

        else:
            for b in self.bands:
                bands.append(dataset_bands_list[b - 1])

        _log.info("Using bands [%s]", " ".join(band.name for band in bands))

        paths = dict()

        # Check output files
        for band in bands:
            paths[band] = self.get_output_filename(band)
            _log.info("Output file for band [%s] is [%s]", band.name, paths[band])

            if os.path.exists(paths[band]):
                if self.overwrite:
                    _log.info("Removing existing output file [%s]", paths[band])
                    os.remove(paths[band])
                else:
                    _log.error("Output file [%s] exists", paths[band])
                    raise Exception("Output file [%s] already exists" % paths[band])

        tiles = list_tiles_as_list(x=[self.x], y=[self.y], acq_min=self.acq_min, acq_max=self.acq_max,
                                   satellites=[satellite for satellite in self.satellites],
                                   datasets=[self.dataset_type],
                                   database=config.get_db_database(),
                                   user=config.get_db_username(),
                                   password=config.get_db_password(),
                                   host=config.get_db_host(), port=config.get_db_port())

        raster = None
        metadata = None

        # TODO - PQ is UNIT16 and WOFS is BYTE (others are INT16) and so -999 NDV doesn't work
        ndv = NDV

        if self.dataset_type == DatasetType.PQ25:
            ndv = UINT16_MAX

        elif self.dataset_type == DatasetType.WATER:
            ndv = BYTE_MAX

        _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

        import itertools

        for band in bands:

            for x, y in itertools.product(range(0, 4000, self.chunk_size_x), range(0, 4000, self.chunk_size_y)):

                _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

                stack = list()

                for tile in tiles:

                    if self.list_only:
                        _log.info("Would summarise dataset [%s]", tile.datasets[self.dataset_type].path)
                        continue

                    # Apply PQA if specified

                    pqa = None

                    if self.mask_pqa_apply and DatasetType.PQ25 in tile.datasets:
                        pqa = tile.datasets[DatasetType.PQ25]

                    # Apply WOFS if specified

                    wofs = None

                    if self.mask_wofs_apply and DatasetType.WATER in tile.datasets:
                        wofs = tile.datasets[DatasetType.WATER]

                    _log.info("About to read data chunk ({xmin:4d},{ymin:4d}) to ({xmax:4d},{ymax:4d}) for band {band} from file {file}".format(xmin=x, ymin=y,
                                                                                                               xmax=x + self.chunk_size_x - 1,
                                                                                                               ymax=y + self.chunk_size_y - 1,
                                                                                                               band=band.name,  file=tile.datasets[self.dataset_type].path))

                    if not metadata:
                        metadata = get_dataset_metadata(tile.datasets[self.dataset_type])

                    mask = None

                    if pqa:
                        mask = get_mask_pqa(pqa, self.mask_pqa_mask, x=x, y=y, x_size=self.chunk_size_x, y_size=self.chunk_size_y)

                    if wofs:
                        mask = get_mask_wofs(wofs, self.mask_wofs_mask, mask=mask, x=x, y=y, x_size=self.chunk_size_x, y_size=self.chunk_size_y)

                    data = get_dataset_data_masked(tile.datasets[self.dataset_type], bands=[band], x=x, y=y, x_size=self.chunk_size_x, y_size=self.chunk_size_y, mask=mask, ndv=ndv)

                    stack.append(data[band])

                    _log.debug("data[%s] has shape [%s] and MB [%s]", band.name, numpy.shape(data[band]), data[band].nbytes / 1000 / 1000)
                    _log.debug("stack[%s] has [%s] elements", band.name, len(stack))

                    del data

                # Apply summary method

                _log.info(
                    "Finished reading {count} datasets for chunk ({xmin:4d},{ymin:4d}) to ({xmax:4d},{ymax:4d}) - about to summarise them".format(
                        count=len(tiles), xmin=x, ymin=y, xmax=x + self.chunk_size_x - 1, ymax=y + self.chunk_size_y - 1))
                _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

                masked_stack = numpy.ma.masked_equal(stack, ndv)

                _log.debug("masked_stack[%s] is %s", band.name, masked_stack)
                _log.debug("masked stack[%s] has shape [%s] and MB [%s]", band.name, numpy.shape(masked_stack), masked_stack.nbytes / 1000 / 1000)
                _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

                if stack is not None:
                    del stack
                    _log.debug("Just NONE-ed the stack")
                    _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

                masked_sorted = None

                for statistic in self.statistics:
                    _log.info("Calculating statistic [%s]", statistic.name)

                    if statistic == Statistic.COUNT:
                        # TODO Need to artificially create masked array here since it is being expected/filled below!!!
                        masked_summary = numpy.ma.masked_equal(masked_stack.count(axis=0), ndv)

                    elif statistic == Statistic.MIN:
                        masked_summary = numpy.min(masked_stack, axis=0)

                    elif statistic == Statistic.MAX:
                        masked_summary = numpy.max(masked_stack, axis=0)

                    elif statistic == Statistic.MEAN:
                        masked_summary = numpy.mean(masked_stack, axis=0)

                    elif statistic == Statistic.MEDIAN:
                        masked_summary = numpy.median(masked_stack, axis=0)

                    elif statistic == Statistic.SUM:
                        masked_summary = numpy.sum(masked_stack, axis=0)

                    elif statistic == Statistic.STANDARD_DEVIATION:
                        masked_summary = numpy.std(masked_stack, axis=0)

                    elif statistic == Statistic.VARIANCE:
                        masked_summary = numpy.var(masked_stack, axis=0)

                    elif statistic == Statistic.PERCENTILE_25:
                        percent = 0.25
                        if masked_sorted is None:
                            numpy.ndarray.sort(masked_stack, axis=0)
                        _log.info("masked_sorted [%s] [%s]", numpy.shape(masked_sorted), masked_sorted)
                        masked_percentile_index = numpy.ma.floor(numpy.ma.count(masked_sorted, axis=0) * percent).astype(numpy.int16)
                        _log.info("masked_percentile_index [%s] [%s]", numpy.shape(masked_percentile_index), masked_percentile_index)
                        index_flat = masked_percentile_index.ravel() * masked_percentile_index.size + numpy.arange(masked_percentile_index.size)
                        masked_summary = masked_sorted.ravel()[index_flat].reshape(numpy.shape(masked_percentile_index))
                        _log.info("masked_summary [%s] [%s]", numpy.shape(masked_summary), masked_summary)
                        del masked_percentile_index

                    elif statistic == Statistic.PERCENTILE_50:
                        percent = 0.50
                        if masked_sorted is None:
                            numpy.ndarray.sort(masked_stack, axis=0)
                        _log.info("masked_sorted [%s] [%s]", numpy.shape(masked_sorted), masked_sorted)
                        masked_percentile_index = numpy.ma.floor(numpy.ma.count(masked_sorted, axis=0) * percent).astype(numpy.int16)
                        _log.info("masked_percentile_index [%s] [%s]", numpy.shape(masked_percentile_index), masked_percentile_index)
                        index_flat = masked_percentile_index.ravel() * masked_percentile_index.size + numpy.arange(masked_percentile_index.size)
                        masked_summary = masked_sorted.ravel()[index_flat].reshape(numpy.shape(masked_percentile_index))
                        _log.info("masked_summary [%s] [%s]", numpy.shape(masked_summary), masked_summary)
                        del masked_percentile_index

                    elif statistic == Statistic.PERCENTILE_75:
                        percent = 0.75
                        if masked_sorted is None:
                            numpy.ndarray.sort(masked_stack, axis=0)
                        _log.info("masked_sorted [%s] [%s]", numpy.shape(masked_sorted), masked_sorted)
                        masked_percentile_index = numpy.ma.floor(numpy.ma.count(masked_sorted, axis=0) * percent).astype(numpy.int16)
                        _log.info("masked_percentile_index [%s] [%s]", numpy.shape(masked_percentile_index), masked_percentile_index)
                        index_flat = masked_percentile_index.ravel() * masked_percentile_index.size + numpy.arange(masked_percentile_index.size)
                        masked_summary = masked_sorted.ravel()[index_flat].reshape(numpy.shape(masked_percentile_index))
                        _log.info("masked_summary [%s] [%s]", numpy.shape(masked_summary), masked_summary)
                        del masked_percentile_index

                    elif statistic == Statistic.PERCENTILE_90:
                        percent = 0.90
                        if masked_sorted is None:
                            numpy.ndarray.sort(masked_stack, axis=0)
                        _log.info("masked_sorted [%s] [%s]", numpy.shape(masked_sorted), masked_sorted)
                        masked_percentile_index = numpy.ma.floor(numpy.ma.count(masked_sorted, axis=0) * percent).astype(numpy.int16)
                        _log.info("masked_percentile_index [%s] [%s]", numpy.shape(masked_percentile_index), masked_percentile_index)
                        index_flat = masked_percentile_index.ravel() * masked_percentile_index.size + numpy.arange(masked_percentile_index.size)
                        masked_summary = masked_sorted.ravel()[index_flat].reshape(numpy.shape(masked_percentile_index))
                        _log.info("masked_summary [%s] [%s]", numpy.shape(masked_summary), masked_summary)
                        del masked_percentile_index

                    elif statistic == Statistic.PERCENTILE_95:
                        percent = 0.95
                        if masked_sorted is None:
                            numpy.ndarray.sort(masked_stack, axis=0)
                        _log.info("masked_sorted [%s] [%s]", numpy.shape(masked_sorted), masked_sorted)
                        masked_percentile_index = numpy.ma.floor(numpy.ma.count(masked_sorted, axis=0) * percent).astype(numpy.int16)
                        _log.info("masked_percentile_index [%s] [%s]", numpy.shape(masked_percentile_index), masked_percentile_index)
                        index_flat = masked_percentile_index.ravel() * masked_percentile_index.size + numpy.arange(masked_percentile_index.size)
                        masked_summary = masked_sorted.ravel()[index_flat].reshape(numpy.shape(masked_percentile_index))
                        _log.info("masked_summary [%s] [%s]", numpy.shape(masked_summary), masked_summary)
                        del masked_percentile_index

                    _log.debug("masked summary is [%s]", masked_summary)
                    _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

                    # Create the output file

                    if not os.path.exists(paths[band]):
                        _log.info("Creating raster [%s]", paths[band])

                        driver = gdal.GetDriverByName("GTiff")
                        assert driver

                        raster = driver.Create(paths[band], metadata.shape[0], metadata.shape[1], len(self.statistics), gdal.GDT_Int16)
                        assert raster

                        raster.SetGeoTransform(metadata.transform)
                        raster.SetProjection(metadata.projection)

                        raster.SetMetadata(self.generate_raster_metadata())

                    _log.info("Setting no data value for band [%s] which has index [%d]", statistic.name, self.statistics.index(statistic) + 1)
                    raster.GetRasterBand(self.statistics.index(statistic) + 1).SetNoDataValue(ndv)

                    raster.GetRasterBand(self.statistics.index(statistic) + 1).SetMetadataItem("BAND_ID", statistic.name)

                    _log.info("Writing band [%s] data to raster [%s]", band.name, paths[band])
                    _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

                    _log.info("masked_summary has shape [%s]", numpy.shape(masked_summary))
                    _log.info("masked_summary has data\n[%s]", masked_summary)

                    raster.GetRasterBand(self.statistics.index(statistic) + 1).WriteArray(masked_summary.filled(ndv), xoff=x, yoff=y)
                    raster.GetRasterBand(self.statistics.index(statistic) + 1).ComputeStatistics(True)

                    raster.FlushCache()

                    if masked_summary is not None:
                        del masked_summary
                        _log.debug("NONE-ing the masked summary")
                        _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

                if masked_stack is not None:
                    del masked_stack
                    _log.debug("NONE-ing masked stack[%s]", band.name)
                    _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

                if masked_sorted is not None:
                    del masked_sorted
                    _log.debug("NONE-ing masked sorted[%s]", band.name)
                    _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

            if raster:
                del raster
                _log.debug("Just NONE'd the raster")
                _log.debug("Current MAX RSS  usage is [%d] MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)

        _log.info("Memory usage was [%d MB]", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        _log.info("CPU time used [%s]", timedelta(seconds=int(resource.getrusage(resource.RUSAGE_SELF).ru_utime)))

    def generate_raster_metadata(self):
        return {
            "X_INDEX": "{x:03d}".format(x=self.x),
            "Y_INDEX": "{y:04d}".format(y=self.y),
            "ACQUISITION_DATE": "{acq_min} to {acq_max}".format(acq_min=self.acq_min, acq_max=self.acq_max),
            "SATELLITES": " ".join([satellite.name for satellite in self.satellites]),
            "PIXEL_QUALITY_FILTER": self.mask_pqa_apply and " ".join([mask.name for mask in self.mask_pqa_mask]) or "",
            "WATER_FILTER": self.mask_wofs_apply and " ".join([mask.name for mask in self.mask_wofs_mask]) or "",
            "STATISTICS": " ".join([statistic.name for statistic in self.statistics])
        }

    def get_output_filename(self, band):

        satellite_str = ""

        if Satellite.LS5 in self.satellites or Satellite.LS7 in self.satellites or Satellite.LS8 in self.satellites:
            satellite_str += "LS"

        if Satellite.LS5 in self.satellites:
            satellite_str += "5"

        if Satellite.LS7 in self.satellites:
            satellite_str += "7"

        if Satellite.LS8 in self.satellites:
            satellite_str += "8"

        dataset_type_string = {
            DatasetType.ARG25: "NBAR",
            DatasetType.PQ25: "PQA",
            DatasetType.FC25: "FC",
            DatasetType.WATER: "WATER",
            DatasetType.NDVI: "NDVI",
            DatasetType.EVI: "EVI",
            DatasetType.NBR: "NBR"
        }[self.dataset_type]

        if self.mask_pqa_apply and self.mask_wofs_apply:
            dataset_type_string += "_WITH_PQA_WATER"

        elif self.mask_pqa_apply:
            dataset_type_string += "_WITH_PQA"

        elif self.mask_wofs_apply:
            dataset_type_string += "_WITH_WATER"

        return os.path.join(self.output_directory,
                            "{satellite}_{dataset}_STATISTICS_{x:03d}_{y:04d}_{acq_min}_{acq_max}_{band}.tif".format(
                                satellite=satellite_str, dataset=dataset_type_string, x=self.x, y=self.y,
                                acq_min=self.acq_min, acq_max=self.acq_max, band=band.name))


def decode_dataset_type(dataset_type):
    return {DatasetType.ARG25: "Surface Reflectance",
            DatasetType.PQ25: "Pixel Quality",
            DatasetType.FC25: "Fractional Cover",
            DatasetType.WATER: "WOFS Woffle",
            DatasetType.NDVI: "NDVI",
            DatasetType.EVI: "EVI",
            DatasetType.NBR: "Normalised Burn Ratio"}[dataset_type]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    SummariseDatasetTimeSeriesStatistics("Summarise Dataset Time Series - Statistics").run()
