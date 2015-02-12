import os
import unittest
from datacube.api.model import DatasetType, Tile, Cell, Satellite
from datacube.config import Config, DbCredentials
from datetime import datetime
from datacube.api.query import list_tiles_as_list, TimeInterval, DatacubeQueryContext
import logging

logging.basicConfig(level=logging.INFO, \
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')

class TestQueryWrapper(unittest.TestCase):

    def test_get_all_tiles_for_WOfS_processing(self):
        """
        This test demonstrates how to get all tiles relevant to WOfS processing
        for the entire continent over all time
        """

        logging.info("get_all_tiles started")

        # get a CubeQueryContext (this is a wrapper around the API)

        config = Config(os.path.expanduser("~/.datacube/config"))
        cube = DatacubeQueryContext(config.get_DbCredentials())

        # we are interested in all time

        time_interval = TimeInterval(datetime(1950,1,1), datetime(2050,1,1))

        # all landsat satellites

        satellite_list = [Satellite.LS7, Satellite.LS5, Satellite.LS8]

        # NBAR, PQ and DSM

        dataset_list = [DatasetType.ARG25, DatasetType.PQ25]

        # now get the tile list and write it to a working file

        wofs_tiles = "./wofs_tiles.csv"
        tiles = cube.tiles_to_file(range(111, 154), range(-46, 4), satellite_list, time_interval, \
            dataset_list, wofs_tiles)

        # all done

        self.assertTrue(os.path.exists(wofs_tiles))
        logging.info("get_all_tiles finished")


if __name__ == '__main__':
    unittest.main()
