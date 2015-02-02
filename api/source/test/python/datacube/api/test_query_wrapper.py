import os
import unittest
from datacube.config import Config, DbCredentials
from datetime import datetime
from datacube.api.query import list_tiles, TimeInterval, DatacubeQueryContext
from datacube.api.model import DatasetType, Tile, Cell

class TestQueryWrapper(unittest.TestCase):


    def test_get_DbCredentials(self):

        config = Config(os.path.expanduser("~/.datacube/config"))
        start = datetime(1950,1,1)
        end = datetime(2050,1,1)

        tiles = list_tiles(x=[123], y=[-25], acq_min=start, acq_max=end, satellites=["LS7"],
                       datasets=[DatasetType.ARG25, DatasetType.PQ25, DatasetType.FC25],
                       database=config.get_db_database(), user=config.get_db_username(),
                       password=config.get_db_password(),
                       host=config.get_db_host(), port=config.get_db_port())

        print len(tiles)

if __name__ == '__main__':
    unittest.main()
