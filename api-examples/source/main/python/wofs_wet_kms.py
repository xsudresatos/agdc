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


import logging


_log = logging.getLogger()


class WofsWetKmsWorkflow():

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

        x_list = range(self.x_min, self.x_max + 1)
        y_list = range(self.y_min, self.y_max + 1)
        year_list = range(self.year_min, self.year_max + 1)
        month_list = range(3, 12 + 1, 3)

        summary = dict()

        import itertools

        for x, y, year, month in itertools.product(x_list, y_list, year_list, month_list):
            _log.info("Doing {year:04d}-{month:02d}_{x:03d}_{y:04d}".format(year=year, month=month, x=x, y=y))

            import os
            import glob

            files = glob.glob(os.path.join(self.input_directory, "{x:03d}_{y:04d}/qtrInYearWaterSummary_{year:04d}_{month:02d}_{x:03d}_{y:04d}.tiff".format(x=x, y=y, year=year, month=month)))
            _log.debug("candidate files = [%s]", files)

            for f in files:
                import gdal

                raster = gdal.Open(f, gdal.GA_ReadOnly)
                assert raster

                metadata = raster.GetMetadata()
                _log.debug("metadata for file [%s] is [%s]", f, metadata)

                qtr = str(metadata["qtrId"])
                observed_pixels = int(metadata["observed_pixels"])
                wet_pixels = int(metadata["wet_pixels"])
                pixel_size = float(metadata["pixel_scale_metres"])

                _log.debug("qtr=[%s] observed=[%d] wet=[%d] pixel size=[%f]", qtr, observed_pixels, wet_pixels, pixel_size)

                if qtr not in summary:
                    summary[qtr] = {"observed_kms": float(0), "wet_kms": float(0)}

                record = summary[qtr]

                record["observed_kms"] += (observed_pixels * pixel_size * pixel_size / 1000 / 1000)
                record["wet_kms"] += (wet_pixels * pixel_size * pixel_size / 1000 / 1000)

                raster = None

            _log.debug("summary contains [%s]", summary)
            _log.debug("summary sorted contains [%s]", sorted(summary))

        import csv
        with open(os.path.join(self.output_directory, "wofs_wet_kms_{x_min:03d}_{x_max:03d}_{y_min:04d}_{y_max:04d}_{year_min:04d}_{year_max:04d}.csv".format(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max, year_min=self.year_min, year_max=self.year_max)), "wb") as f:
            w = csv.writer(f)

            w.writerow(["Quarter", "Wet KMs", "Observed KMs"])

            for k in sorted(summary):
                w.writerow([k, summary[k]["wet_kms"], summary[k]["observed_kms"]])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    WofsWetKmsWorkflow("WOFS Wet Kms").run()
