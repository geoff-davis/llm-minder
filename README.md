# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/llm-minder/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/llm\_minder/\_\_init\_\_.py                        |       45 |        2 |        2 |        0 |     95.74% |   360-362 |
| src/llm\_minder/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/admission.py                |      421 |       16 |      118 |       14 |     94.43% |106, 114, 131, 133, 135, 155, 171, 173, 267, 299-300, 389, 403, 408-409, 523-\>521, 605 |
| src/llm\_minder/\_internal/artifact\_codec.py          |      193 |       14 |       58 |        7 |     91.63% |69, 73, 77, 124, 153, 189, 300-\>299, 376, 402, 410, 455, 471-474 |
| src/llm\_minder/\_internal/backoff.py                  |       14 |        2 |        8 |        0 |     90.91% |     19-20 |
| src/llm\_minder/\_internal/budget.py                   |       74 |        0 |       24 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/cache\_groups.py            |       40 |        0 |        6 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/capacity.py                 |      131 |       11 |       32 |        2 |     90.80% |43-44, 84-93, 159-160, 187-193 |
| src/llm\_minder/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/cleanup.py                  |      262 |        8 |       94 |        7 |     94.66% |144-\>exit, 150, 177, 297-\>299, 303-\>exit, 384, 417-\>430, 471-475 |
| src/llm\_minder/\_internal/error\_logging.py           |      107 |       16 |       40 |        8 |     82.31% |42-\>45, 93, 107-\>117, 115-116, 118, 121-132, 149-\>170, 151, 153-156, 167-168 |
| src/llm\_minder/\_internal/event\_dispatcher.py        |       90 |        4 |       30 |        1 |     95.83% |82-83, 122-\>124, 147, 163 |
| src/llm\_minder/\_internal/execution\_state.py         |       59 |        1 |        6 |        1 |     96.92% |       125 |
| src/llm\_minder/\_internal/executor\_host.py           |       69 |        0 |        6 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/guardrails.py               |      145 |        4 |       50 |        6 |     94.87% |79, 89, 121, 160-\>exit, 178, 192-\>194 |
| src/llm\_minder/\_internal/input\_validation.py        |       36 |        1 |       10 |        1 |     95.65% |        59 |
| src/llm\_minder/\_internal/item\_executor.py           |      790 |       23 |      242 |       33 |     94.57% |203, 246-\>exit, 248-\>exit, 252-\>exit, 261-\>exit, 269-\>exit, 303-\>exit, 320, 336, 416, 476, 536-\>exit, 545-\>547, 548-\>550, 564-567, 681, 707-\>715, 754-\>758, 772-\>exit, 816, 818, 972, 1047, 1122-1129, 1169, 1393-1394, 1397-\>1402, 1415, 1504-\>exit, 1566, 1582-\>1586, 1603-\>1607, 1658-\>1662, 1667-\>1671, 1707-\>1712, 1766-\>1777, 1866-\>1873, 1922, 1924-\>1926 |
| src/llm\_minder/\_internal/logical\_item.py            |       17 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/rate\_limit\_coordinator.py |      168 |        5 |       42 |        1 |     97.14% |122, 293, 370-376 |
| src/llm\_minder/\_internal/responses\_translation.py   |      163 |       13 |      102 |       15 |     89.43% |50, 69, 97, 106, 119, 121-\>123, 139, 144, 157, 168, 170-\>172, 189, 208, 212, 222 |
| src/llm\_minder/\_internal/strategy\_lifecycle.py      |      172 |        5 |       42 |        5 |     95.33% |91, 143-\>141, 218, 245, 251-\>254, 257, 330 |
| src/llm\_minder/artifacts.py                           |      539 |       68 |      176 |       33 |     85.03% |131, 143-\>145, 160, 162-164, 199-200, 261-262, 278-279, 286, 290-291, 307-308, 323, 328, 341, 345, 350, 364-365, 369-\>373, 388, 390-393, 395, 418-419, 428, 434-438, 443, 449, 459, 465, 482-483, 487, 543-544, 551, 588, 606-607, 654-655, 680, 760-761, 781, 787-\>786, 814, 817, 855-857, 861-862, 879, 894, 911-\>923, 916-917, 945, 954-\>956, 978-979, 1013-\>1015 |
| src/llm\_minder/base.py                                |      965 |       42 |      266 |       21 |     94.23% |129, 357, 366, 686-\>exit, 689-\>exit, 743, 780-\>782, 833-\>838, 1017-\>1020, 1138, 1140, 1517, 1559-1560, 1726, 1822-1827, 1852, 1867-1870, 1891-\>exit, 1928-1929, 1959, 1963, 2006-\>exit, 2017, 2045-2046, 2068-2076, 2092-\>2099, 2172, 2221, 2240, 2244-\>2250, 2246-2249, 2280-2281, 2296-2298 |
| src/llm\_minder/budget.py                              |       13 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/cache\_groups.py                       |      221 |        3 |       88 |        3 |     98.06% |305, 420, 462 |
| src/llm\_minder/call\_pool.py                          |      130 |        3 |       32 |        2 |     96.91% |262-\>264, 277, 281-282 |
| src/llm\_minder/callable\_strategy.py                  |      141 |        4 |       50 |        6 |     94.76% |56-\>exit, 69-\>exit, 88, 131, 172, 246 |
| src/llm\_minder/categories.py                          |       96 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/classifiers/\_\_init\_\_.py            |        5 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/classifiers/gemini.py                  |       92 |        7 |       50 |        5 |     91.55% |27-\>30, 41, 55-\>52, 60-61, 115-116, 223, 231 |
| src/llm\_minder/classifiers/openai.py                  |       71 |        5 |       40 |        1 |     94.59% |114, 146-147, 181-182 |
| src/llm\_minder/classifiers/openrouter.py              |       37 |        2 |       16 |        0 |     96.23% |     85-86 |
| src/llm\_minder/classifiers/pydantic\_ai.py            |       16 |        2 |        6 |        0 |     90.91% |     16-17 |
| src/llm\_minder/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/core/config.py                         |      231 |        4 |      136 |        4 |     97.82% |149, 151, 276, 543 |
| src/llm\_minder/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/factory.py                             |       66 |        3 |       32 |        3 |     91.84% |106-\>exit, 117-\>exit, 219-224 |
| src/llm\_minder/gateway.py                             |        8 |        0 |        2 |        0 |    100.00% |           |
| src/llm\_minder/llm\_strategies.py                     |      200 |        8 |       54 |        8 |     93.70% |34-35, 69-\>67, 71, 324-\>exit, 336-\>exit, 423, 441, 454-\>457, 644-\>646, 649, 787-789 |
| src/llm\_minder/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/models.py                              |     1250 |       90 |      564 |       77 |     90.02% |57-59, 62-63, 113-129, 492-\>495, 570-\>exit, 578, 581, 619, 626, 652, 831-\>834, 836, 842-\>exit, 867, 948, 951, 1003, 1033, 1044, 1070-1071, 1087-\>1121, 1098, 1100, 1136, 1152-\>1170, 1191-\>1196, 1493-\>1542, 1496-\>1499, 1500-\>1536, 1516-1517, 1522, 1543-\>1547, 1545-1546, 1557, 1611-\>1621, 1614-\>1621, 1616-1617, 1637, 1640, 1643-\>1651, 1898, 1907-1912, 1949, 2019, 2032, 2049-\>2051, 2052-\>2054, 2055-\>2057, 2069-2072, 2074-2078, 2086, 2088, 2153-2154, 2348, 2351-\>2353, 2354-\>2356, 2363, 2368-\>2373, 2380-\>2382, 2407-\>2405, 2411-\>2409, 2415-\>2402, 2425-\>2402, 2430-\>2426, 2432-\>2426, 2435-\>2433, 2452, 2503-2504, 2594, 2617, 2627-2628, 2631, 2643-\>2652, 2647-2651, 2658-2659, 2672, 2748, 2757, 2761-2762, 2796, 2799, 2802-2805, 2933, 2953-\>2958, 2999-\>3001 |
| src/llm\_minder/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/llm\_minder/observers/metrics.py                   |      143 |        5 |       72 |       10 |     92.09% |56, 73-\>80, 83-\>89, 90-\>exit, 115-\>109, 118-\>exit, 129, 135-136, 153, 176-\>exit |
| src/llm\_minder/parallel.py                            |      437 |       22 |      108 |        8 |     94.50% |114-115, 118-119, 305, 316, 320, 328, 332, 336, 340, 344, 356, 389-\>391, 479-\>481, 488, 502, 510, 543-544, 587-\>589, 655, 660, 755, 938-\>943, 990 |
| src/llm\_minder/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/llm\_minder/provider\_calls.py                     |       22 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/provider\_output.py                    |      102 |        0 |       28 |        0 |    100.00% |           |
| src/llm\_minder/serialization.py                       |      250 |       39 |      102 |       21 |     82.39% |119, 133, 142-143, 225-226, 230, 255, 261, 271, 273, 353-354, 369, 379-380, 389, 391, 405, 476, 483, 490, 495, 524, 541, 577, 608-611, 663-664, 675-676, 682, 685-686, 698, 709 |
| src/llm\_minder/single.py                              |       35 |        3 |        4 |        1 |     89.74% | 54-55, 69 |
| src/llm\_minder/sqlite\_artifacts.py                   |      721 |       89 |      208 |       39 |     84.93% |88-89, 126-\>exit, 134, 139, 218, 226-227, 276-277, 284-286, 291-296, 319, 361-362, 421-422, 427, 431, 461, 528-531, 551, 583, 602-605, 608-609, 611-\>618, 614-615, 626, 655, 663, 677, 683, 689-690, 712, 714-\>exit, 728-\>730, 731, 737-\>739, 748, 756-762, 799-800, 811-\>810, 826-\>825, 840-841, 853, 863-865, 881-\>896, 910, 930, 947-950, 954-956, 1049-1051, 1067, 1134, 1141, 1144, 1148, 1159, 1190, 1286, 1352-1353, 1399-1400, 1408-1409, 1412-1413, 1417 |
| src/llm\_minder/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/strategies/errors.py                   |      259 |       19 |       94 |        4 |     93.48% |95-\>99, 116, 162-163, 168-169, 175-186, 397, 647-648, 685 |
| src/llm\_minder/strategies/rate\_limit.py              |       32 |        0 |        2 |        0 |    100.00% |           |
| src/llm\_minder/streaming.py                           |      213 |        1 |       78 |        4 |     98.28% |81-\>exit, 219-\>221, 275-\>278, 362, 368-\>exit |
| src/llm\_minder/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/llm\_minder/token\_extractor.py                    |      136 |        3 |       56 |        5 |     95.83% |88-\>98, 101-\>107, 103-\>107, 146-147, 154, 206-\>208 |
| **TOTAL**                                              | **9612** |  **548** | **3202** |  **356** | **92.52%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/geoff-davis/llm-minder/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/geoff-davis/llm-minder/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/geoff-davis/llm-minder/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/geoff-davis/llm-minder/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fgeoff-davis%2Fllm-minder%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/geoff-davis/llm-minder/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.