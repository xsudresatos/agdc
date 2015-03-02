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
import gdal


__author__ = "Simon Oldfield"


import argparse
import csv
import glob
import logging
import os
import sys
from datacube.api.model import DatasetType, DatasetTile, Wofs25Bands, Satellite, dataset_type_database, \
    dataset_type_filesystem, dataset_type_derived_nbar
from datacube.api.query import list_tiles
from datacube.api.utils import latlon_to_cell, latlon_to_xy, PqaMask, UINT16_MAX, WofsMask, BYTE_MAX, get_mask_pqa, \
    get_mask_wofs, get_dataset_data_masked, intersection, calculate_ndvi, calculate_evi, calculate_nbr, union
from datacube.api.utils import get_dataset_data, get_dataset_data_with_pq, get_dataset_metadata
from datacube.api.utils import extract_fields_from_filename, NDV
from datacube.api.workflow import writeable_dir
from datacube.config import Config


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


class TimeSeriesRetrievalWorkflow():

    application_name = None

    latitude = None
    longitude = None

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

    output_no_data = None

    dataset_type = None

    delimiter = None
    output_directory = None
    overwrite = None

    def __init__(self, application_name):
        self.application_name = application_name

    def parse_arguments(self):
        parser = argparse.ArgumentParser(prog=__name__, description=self.application_name)

        group = parser.add_mutually_exclusive_group()

        group.add_argument("--quiet", help="Less output", action="store_const", dest="log_level", const=logging.WARN)
        group.add_argument("--verbose", help="More output", action="store_const", dest="log_level", const=logging.DEBUG)

        parser.set_defaults(log_level=logging.INFO)

        parser.add_argument("--lat", help="Latitude value of pixel", action="store", dest="latitude", type=float, required=True)
        parser.add_argument("--lon", help="Longitude value of pixel", action="store", dest="longitude", type=float, required=True)

        parser.add_argument("--acq-min", help="Acquisition Date", action="store", dest="acq_min", type=str)
        parser.add_argument("--acq-max", help="Acquisition Date", action="store", dest="acq_max", type=str)

        # parser.add_argument("--process-min", help="Process Date", action="store", dest="process_min", type=str)
        # parser.add_argument("--process-max", help="Process Date", action="store", dest="process_max", type=str)
        #
        # parser.add_argument("--ingest-min", help="Ingest Date", action="store", dest="ingest_min", type=str)
        # parser.add_argument("--ingest-max", help="Ingest Date", action="store", dest="ingest_max", type=str)

        parser.add_argument("--satellite", help="The satellite(s) to include", action="store", dest="satellite",
                            type=satellite_arg, nargs="+", choices=Satellite, default=[Satellite.LS5, Satellite.LS7], metavar=" ".join([s.name for s in Satellite]))

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

        parser.add_argument("--hide-no-data", help="Don't output records that are completely no data value(s)", action="store_false", dest="output_no_data", default=True)

        supported_dataset_types = dataset_type_database + dataset_type_filesystem + dataset_type_derived_nbar

        # For now only only one type of dataset per customer
        parser.add_argument("--dataset-type", help="The type of dataset from which values will be retrieved", action="store",
                            dest="dataset_type",
                            type=dataset_type_arg,
                            #nargs="+",
                            choices=supported_dataset_types, default=DatasetType.ARG25, required=True, metavar=" ".join([s.name for s in supported_dataset_types]))

        parser.add_argument("--delimiter", help="Field delimiter in output file", action="store", dest="delimiter", type=str, default=",")

        parser.add_argument("--output-directory", help="Output directory", action="store", dest="output_directory",
                            type=writeable_dir)

        parser.add_argument("--overwrite", help="Over write existing output file", action="store_true", dest="overwrite", default=False)

        args = parser.parse_args()

        _log.setLevel(args.log_level)

        self.latitude = args.latitude
        self.longitude = args.longitude

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

        self.output_no_data = args.output_no_data

        self.dataset_type = args.dataset_type

        self.delimiter = args.delimiter
        self.output_directory = args.output_directory
        self.overwrite = args.overwrite

        _log.info("""
        longitude = {longitude:f}
        latitude = {latitude:f}
        acq = {acq_min} to {acq_max}
        process = {process_min} to {process_max}
        ingest = {ingest_min} to {ingest_max}
        satellites = {satellites}
        PQA mask = {pqa_mask}
        WOFS mask = {wofs_mask}
        datasets to retrieve = {dataset_type}
        output no data values = {output_no_data}
        output = {output}
        over write = {overwrite}
        delimiter = {delimiter}
        """.format(longitude=self.longitude, latitude=self.latitude,
                   acq_min=self.acq_min, acq_max=self.acq_max,
                   process_min=self.process_min, process_max=self.process_max,
                   ingest_min=self.ingest_min, ingest_max=self.ingest_max,
                   satellites=" ".join([satellite.name for satellite in self.satellites]),
                   pqa_mask=self.mask_pqa_apply and " ".join([mask.name for mask in self.mask_pqa_mask]) or "",
                   wofs_mask=self.mask_wofs_apply and " ".join([mask.name for mask in self.mask_wofs_mask]) or "",
                   dataset_type=decode_dataset_type(self.dataset_type),
                   output_no_data=self.output_no_data,
                   output=self.output_directory and self.output_directory or "STDOUT",
                   overwrite=self.overwrite,
                   delimiter=self.delimiter))

    def run(self):
        self.parse_arguments()

        config = Config(os.path.expanduser("~/.datacube/config"))
        _log.debug(config.to_str())

        cell_x, cell_y = latlon_to_cell(self.latitude, self.longitude)

        # TODO once WOFS is in the cube

        if self.dataset_type in union(dataset_type_database, dataset_type_derived_nbar):

            # TODO - PQ is UNIT16 and WOFS is BYTE (others are INT16) and so -999 NDV doesn't work
            ndv = NDV

            if self.dataset_type == DatasetType.PQ25:
                ndv = UINT16_MAX

            elif self.dataset_type == DatasetType.WATER:
                ndv = BYTE_MAX

            headered = False

            with self.get_output_file(self.dataset_type, self.overwrite) as csv_file:

                csv_writer = csv.writer(csv_file, delimiter=self.delimiter)

                for tile in list_tiles(x=[cell_x], y=[cell_y], acq_min=self.acq_min, acq_max=self.acq_max,
                                       satellites=[satellite for satellite in self.satellites],
                                       datasets=[self.dataset_type],
                                       database=config.get_db_database(),
                                       user=config.get_db_username(),
                                       password=config.get_db_password(),
                                       host=config.get_db_host(), port=config.get_db_port()):

                    # Apply PQA if specified

                    pqa = None

                    if self.mask_pqa_apply and DatasetType.PQ25 in tile.datasets:
                        pqa = tile.datasets[DatasetType.PQ25]

                    # Apply WOFS if specified

                    wofs = None

                    if self.mask_wofs_apply and DatasetType.WATER in tile.datasets:
                        wofs = tile.datasets[DatasetType.WATER]

                    # Output a HEADER
                    if not headered:
                        if self.dataset_type in dataset_type_database:
                            header_fields = ["SATELLITE", "ACQUISITION DATE"] + [b.name for b in tile.datasets[self.dataset_type].bands]
                        elif self.dataset_type in dataset_type_derived_nbar:
                            header_fields = ["SATELLITE", "ACQUISITION DATE"] + [self.dataset_type.name]
                        csv_writer.writerow(header_fields)
                        headered = True

                    if self.dataset_type in dataset_type_database:
                        data = retrieve_pixel_value(tile.datasets[self.dataset_type], pqa, self.mask_pqa_mask,
                                                    wofs, self.mask_wofs_mask, self.latitude, self.longitude, ndv=ndv)

                        bands = tile.datasets[self.dataset_type].bands
                        satellite = tile.datasets[self.dataset_type].satellite.value

                    elif self.dataset_type in dataset_type_derived_nbar:
                        data = retrieve_pixel_value_derived_nbar(self.dataset_type, tile.datasets[DatasetType.ARG25],
                                                                 pqa, self.mask_pqa_mask,
                                                                 wofs, self.mask_wofs_mask,
                                                                 self.latitude, self.longitude, ndv=ndv)
                        bands = [self.dataset_type.name]
                        satellite = tile.datasets[DatasetType.ARG25].satellite.value

                    _log.debug("data is [%s]", data)
                    if has_data(bands, data, no_data_value=ndv) or self.output_no_data:
                        csv_writer.writerow([satellite, str(tile.end_datetime)] + decode_data(self.dataset_type, bands, data))

    def get_output_file(self, dataset_type, overwrite=False):

        if not self.output_directory:
            _log.info("Writing output to standard output")
            return sys.stdout

        filename = self.get_output_filename(dataset_type)

        _log.info("Writing output to %s", filename)

        if os.path.exists(filename) and not overwrite:
            _log.error("Output file [%s] exists", filename)
            raise Exception("Output file [%s] already exists" % filename)

        return open(self.get_output_filename(dataset_type), "wb")

    def get_output_filename(self, dataset_type):

        if dataset_type == DatasetType.WATER:
            return os.path.join(self.output_directory,"LS_WOFS_{longitude:03.5f}_{latitude:03.5f}_{acq_min}_{acq_max}.csv".format(latitude=self.latitude,
                                                                                              longitude=self.longitude,
                                                                                              acq_min=self.acq_min,
                                                                                              acq_max=self.acq_max))
        satellite_str = ""

        if Satellite.LS5 in self.satellites or Satellite.LS7 in self.satellites or Satellite.LS8 in self.satellites:
            satellite_str += "LS"

        if Satellite.LS5 in self.satellites:
            satellite_str += "5"

        if Satellite.LS7 in self.satellites:
            satellite_str += "7"

        if Satellite.LS8 in self.satellites:
            satellite_str += "8"

        dataset_str = ""

        if dataset_type == DatasetType.ARG25:
            dataset_str += "NBAR"

        elif dataset_type == DatasetType.PQ25:
            dataset_str += "PQA"

        elif dataset_type == DatasetType.FC25:
            dataset_str += "FC"

        elif dataset_type == DatasetType.WATER:
            dataset_str += "WOFS"

        if self.mask_pqa_apply:
            dataset_str += "_WITH_PQA"

        return os.path.join(self.output_directory,
                            "{satellite}_{dataset}_{longitude:03.5f}_{latitude:03.5f}_{acq_min}_{acq_max}.csv".format(satellite=satellite_str, dataset=dataset_str, latitude=self.latitude,
                                                                                          longitude=self.longitude,
                                                                                          acq_min=self.acq_min,
                                                                                          acq_max=self.acq_max))


def decode_dataset_type(dataset_type):
    return {DatasetType.ARG25: "Surface Reflectance",
              DatasetType.PQ25: "Pixel Quality",
              DatasetType.FC25: "Fractional Cover",
              DatasetType.WATER: "WOFS Woffle",
              DatasetType.NDVI: "NDVI",
              DatasetType.EVI: "EVI",
              DatasetType.NBR: "NBR"}[dataset_type]


def has_data(bands, data, no_data_value=NDV):
    for value in [data[band][0][0] for band in bands]:
        if value != no_data_value:
            return True

    return False


def decode_data(dataset_type, bands, data):

    if dataset_type == DatasetType.WATER:
        return [decode_wofs_water_value(data[Wofs25Bands.WATER][0][0]), str(data[Wofs25Bands.WATER][0][0])]

    return [str(data[band][0][0]) for band in bands]


def retrieve_pixel_value(dataset, pqa, pqa_masks, wofs, wofs_masks, latitude, longitude, ndv=NDV):
    _log.debug(
        "Retrieving pixel value(s) at lat=[%f] lon=[%f] from [%s] with pqa [%s] and paq mask [%s] and wofs [%s] and wofs mask [%s]",
        latitude, longitude, dataset.path, pqa and pqa.path or "", pqa and pqa_masks or "",
        wofs and wofs.path or "", wofs and wofs_masks or "")

    metadata = get_dataset_metadata(dataset)
    x, y = latlon_to_xy(latitude, longitude, metadata.transform)

    _log.debug("Retrieving value at x=[%d] y=[%d]", x, y)

    mask = None

    if pqa:
        mask = get_mask_pqa(pqa, pqa_masks, x=x, y=y, x_size=1, y_size=1)

    if wofs:
        mask = get_mask_wofs(wofs, wofs_masks, x=x, y=y, x_size=1, y_size=1, mask=mask)

    data = get_dataset_data_masked(dataset, x=x, y=y, x_size=1, y_size=1, mask=mask, ndv=ndv)

    _log.debug("data is [%s]", data)

    return data


def retrieve_pixel_value_derived_nbar(dataset_type, nbar, pqa, pqa_masks, wofs, wofs_masks, latitude, longitude, ndv=NDV):
    _log.debug(
        "Retrieving pixel value(s) at lat=[%f] lon=[%f] from [%s] derived from [%s] with pqa [%s] and paq mask [%s] and wofs [%s] and wofs mask [%s]",
        latitude, longitude, dataset_type.name, nbar.path, pqa and pqa.path or "", pqa and pqa_masks or "",
        wofs and wofs.path or "", wofs and wofs_masks or "")

    metadata = get_dataset_metadata(nbar)

    x, y = latlon_to_xy(latitude, longitude, metadata.transform)

    _log.debug("Retrieving value at x=[%d] y=[%d]", x, y)

    mask = None

    if pqa:
        mask = get_mask_pqa(pqa, pqa_masks, x=x, y=y, x_size=1, y_size=1)

    if wofs:
        mask = get_mask_wofs(wofs, wofs_masks, x=x, y=y, x_size=1, y_size=1, mask=mask)

    data = get_dataset_data_masked(nbar, x=x, y=y, x_size=1, y_size=1, mask=mask, ndv=ndv)

    if dataset_type == DatasetType.NDVI:
        data = calculate_ndvi(data[nbar.bands.RED], data[nbar.bands.NEAR_INFRARED])

    elif dataset_type == DatasetType.EVI:
        data = calculate_evi(data[nbar.bands.RED], data[nbar.bands.BLUE], data[nbar.bands.NEAR_INFRARED])

    elif dataset_type == DatasetType.NBR:
        data = calculate_nbr(data[nbar.bands.NEAR_INFRARED], data[nbar.bands.SHORT_WAVE_INFRARED_2])

    _log.debug("data is [%s]", data)

    return {dataset_type.name: data}


# A WaterTile stores 1 data layer encoded as unsigned BYTE values as described in the WaterConstants.py file.
#
# Note - legal (decimal) values are:
#
#        0:  no water in pixel
#        1:  no data (one or more bands) in source NBAR image
#    2-127:  pixel masked for some reason (refer to MASKED bits)
#      128:  water in pixel
#
# Values 129-255 are illegal (i.e. if bit 7 set, all others must be unset)
#
#
# WATER_PRESENT             (dec 128) bit 7: 1=water present, 0=no water if all other bits zero
# MASKED_CLOUD              (dec 64)  bit 6: 1=pixel masked out due to cloud, 0=unmasked
# MASKED_CLOUD_SHADOW       (dec 32)  bit 5: 1=pixel masked out due to cloud shadow, 0=unmasked
# MASKED_HIGH_SLOPE         (dec 16)  bit 4: 1=pixel masked out due to high slope, 0=unmasked
# MASKED_TERRAIN_SHADOW     (dec 8)   bit 3: 1=pixel masked out due to terrain shadow or low incident angle, 0=unmasked
# MASKED_SEA_WATER          (dec 4)   bit 2: 1=pixel masked out due to being over sea, 0=unmasked
# MASKED_NO_CONTIGUITY      (dec 2)   bit 1: 1=pixel masked out due to lack of data contiguity, 0=unmasked
# NO_DATA                   (dec 1)   bit 0: 1=pixel masked out due to NO_DATA in NBAR source, 0=valid data in NBAR
# WATER_NOT_PRESENT         (dec 0)          All bits zero indicated valid observation, no water present


def decode_wofs_water_value(value):

    # values = {
    #     0: "Dry|7",
    #     1: "No Data|0",
    #     2: "Saturation/Contiguity|1",
    #     4: "Sea Water|2",
    #     8: "Terrain Shadow|3",
    #     16: "High Slope|4",
    #     32: "Cloud Shadow|5",
    #     64: "Cloud|6",
    #     128: "Wet|8"
    #     }

    values = {
        0: "Dry",
        1: "No Data",
        2: "Saturation/Contiguity",
        4: "Sea Water",
        8: "Terrain Shadow",
        16: "High Slope",
        32: "Cloud Shadow",
        64: "Cloud",
        128: "Wet",
        BYTE_MAX: "--"
        }

    return values[value]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    TimeSeriesRetrievalWorkflow("Time Series Retrieval").run()