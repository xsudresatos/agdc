import os
import unittest
from datacube.api.model import DatasetType, Tile, Cell, Satellite
from datacube.config import Config, DbCredentials
from datetime import datetime
from datacube.api.query import list_tiles_as_list, TimeInterval, DatacubeQueryContext
import logging

logging.basicConfig(level=logging.INFO)

class TestQueryWrapper(unittest.TestCase):

    def test_get_tiles_base_api(self):
        config = Config(os.path.expanduser("~/.datacube/config"))
        start = datetime(1950,1,1)
        end = datetime(2050,1,1)

        tiles = list_tiles_as_list(x=[123], y=[-25], acq_min=start, acq_max=end, \
                       satellites=[Satellite.LS7],
                       datasets=[DatasetType.ARG25, DatasetType.PQ25, DatasetType.FC25], \
                       database=config.get_db_database(), user=config.get_db_username(), \
                       password=config.get_db_password(), \
                       host=config.get_db_host(), port=config.get_db_port())
        self.assertEqual(len(tiles), 523)

    def test_get_tiles_with_cube_context(self):
        config = Config(os.path.expanduser("~/.datacube/config"))
        time_interval = TimeInterval(datetime(1950,1,1), datetime(2050,1,1))
        print "type of time_interval = %s" % type(time_interval)
        satellite_list = [Satellite.LS7]
        dataset_list = [DatasetType.ARG25, DatasetType.PQ25, DatasetType.FC25]

        cube = DatacubeQueryContext(config.get_DbCredentials())
        tiles = cube.tile_list([123], [-25], satellite_list, time_interval, dataset_list)

        self.assertEqual(len(tiles), 523)


    def test_get_tiles_to_file_with_cube_context(self):
        config = Config(os.path.expanduser("~/.datacube/config"))
        time_interval = TimeInterval(datetime(1950,1,1), datetime(2050,1,1))
        print "type of time_interval = %s" % type(time_interval)
        satellite_list = [Satellite.LS7]
        dataset_list = [DatasetType.ARG25, DatasetType.PQ25, DatasetType.FC25]

        cube = DatacubeQueryContext(config.get_DbCredentials())
        tiles = cube.tiles_to_file([123], [-25], satellite_list, time_interval, dataset_list, \
                 "tmp.txt")

#        self.assertEqual(len(tiles), 523)

    def test_get_all_tiles_with_cube_context(self):
        logging.info("get_all_tiles started")
        config = Config(os.path.expanduser("~/.datacube/config"))
        time_interval = TimeInterval(datetime(1950,1,1), datetime(2050,1,1))
        satellite_list = [Satellite.LS7, Satellite.LS5, Satellite.LS8]
        dataset_list = [DatasetType.ARG25, DatasetType.PQ25]

        cube = DatacubeQueryContext(config.get_DbCredentials())
        tiles = cube.tiles_to_file(range(111, 154), range(-46, 4), satellite_list, time_interval, \
            dataset_list, "tmp.txt")

        logging.info("get_all_tiles finished")

#        self.assertEqual(len(tiles), 523)


if __name__ == '__main__':
    unittest.main()
