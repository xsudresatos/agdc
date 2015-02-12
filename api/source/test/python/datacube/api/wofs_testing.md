==========================
WOFS Testing
==========================

Getting all tiles
-----------------

This test looks at the simple use case of getting all NBAR, PQ and DSM tiles from the datacube. 

See [the code](https://github.com/GeoscienceAustralia/agdc/blob/stevenring/api/api/source/test/python/datacube/api/wofs_use_case_tests.py)
```
[smr547@raijin4 api]$ python wofs_use_case_tests.py
02-12 16:17 root         INFO     get_all_tiles started
02-12 16:19 root         INFO     get_all_tiles finished
.
----------------------------------------------------------------------
Ran 1 test in 97.736s
  
OK
[smr547@raijin4 api]$ cat wofs_tiles.csv | wc -l
997374
```
