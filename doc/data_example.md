This is an example raw data for AI's refernce. 

Note that: 
- measurement is added in the operation history simply for ease of data recording, it will not be an input feature
- the transition between the measurement and the operation right before the measurement always has vacuum_break=1 so we don't need this as an input feature
- all wafer in the same group will share the same history (they are repeated experiments) but possibly different measurement results


| group | wafer_id | step | operation | parameter | value |
|-------|----------|------|-----------|-----------|-------|
| 1     | ABC000001 | 1 | deposition | material | TiN |
| 1     | ABC000001 | 1 | deposition | temperature | 1000 |
| 1     | ABC000001 | 1 | deposition | recipe | TiN_version1 |
| 1     | ABC000001 | 1 | transition | vacuum_break | 1 |
| 1     | ABC000001 | 2 | treatment | recipe | anneal_veresion1 |
| 1     | ABC000001 | 2 | treatment | time | 120 |
| 1     | ABC000001 | 2 | treatment | temperature | 2000 |
| 1     | ABC000001 | 2 | transition | vacuum_break | 1 |
| 1     | ABC000001 | 3 | measurement | stress | 100 |
| 1     | ABC000002 | 1 | deposition | material | TiN |
| 1     | ABC000002 | 1 | deposition | temperature | 1000 |
| 1     | ABC000002 | 1 | deposition | recipe | TiN_version1 |
| 1     | ABC000002 | 1 | transition | vacuum_break | 1 |
| 1     | ABC000002 | 2 | treatment | recipe | anneal_veresion1 |
| 1     | ABC000002 | 2 | treatment | time | 120 |
| 1     | ABC000002 | 2 | treatment | temperature | 2000 |
| 1     | ABC000002 | 2 | transition | vacuum_break | 1 |
| 1     | ABC000002 | 3 | measurement | stress | 120 |
| 2     | ABC000003 | 1 | deposition | material | W |
| 2     | ABC000003 | 1 | deposition | temperature | 2000 |
| 2     | ABC000003 | 1 | deposition | recipe | W_version3 |
| 2     | ABC000003 | 1 | deposition | thickness | 900 |
| 2     | ABC000003 | 1 | transition | vacuum_break | 0 |
| 2     | ABC000003 | 2 | treatment | recipe | oxidation_version2 |
| 2     | ABC000003 | 2 | treatment | time | 500 |
| 2     | ABC000003 | 2 | treatment | temperature | 60 |
| 2     | ABC000003 | 2 | transition | vacuum_break | 1 |
| 2     | ABC000003 | 3 | measurement | stress | 1000 |