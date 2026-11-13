# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/llm-minder/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/llm\_minder/\_\_init\_\_.py                        |       46 |        2 |        2 |        0 |     95.83% |   361-363 |
| src/llm\_minder/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/admission.py                |      421 |       16 |      118 |       14 |     94.43% |106, 114, 131, 133, 135, 155, 171, 173, 267, 299-300, 389, 403, 408-409, 523-\>521, 605 |
| src/llm\_minder/\_internal/artifact\_codec.py          |      193 |       14 |       58 |        7 |     91.63% |69, 73, 77, 124, 153, 189, 300-\>299, 376, 402, 410, 455, 471-474 |
| src/llm\_minder/\_internal/backoff.py                  |       14 |        2 |        8 |        0 |     90.91% |     19-20 |
| src/llm\_minder/\_internal/budget.py                   |       74 |        0 |       24 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/cache\_groups.py            |       40 |        0 |        6 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/capacity.py                 |      140 |        9 |       32 |        2 |     92.44% |43-44, 85-94, 202-208 |
| src/llm\_minder/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/cleanup.py                  |      291 |       17 |      106 |        8 |     91.69% |144-\>exit, 150, 157-158, 163-166, 207-209, 213, 242, 362-\>364, 368-\>exit, 449, 482-\>495, 536-540 |
| src/llm\_minder/\_internal/error\_logging.py           |      107 |       16 |       40 |        8 |     82.31% |42-\>45, 93, 107-\>117, 115-116, 118, 121-132, 149-\>170, 151, 153-156, 167-168 |
| src/llm\_minder/\_internal/event\_dispatcher.py        |       90 |        4 |       30 |        1 |     95.83% |82-83, 122-\>124, 147, 163 |
| src/llm\_minder/\_internal/execution\_state.py         |       59 |        1 |        6 |        1 |     96.92% |       125 |
| src/llm\_minder/\_internal/executor\_host.py           |       69 |        0 |        6 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/guardrails.py               |      143 |        4 |       50 |        6 |     94.82% |80, 91, 123, 162-\>exit, 180, 194-\>196 |
| src/llm\_minder/\_internal/input\_validation.py        |       43 |        1 |       12 |        1 |     96.36% |        84 |
| src/llm\_minder/\_internal/item\_executor.py           |      793 |       23 |      244 |       33 |     94.60% |209, 252-\>exit, 254-\>exit, 258-\>exit, 267-\>exit, 275-\>exit, 309-\>exit, 326, 342, 422, 482, 542-\>exit, 551-\>553, 554-\>556, 570-573, 687, 713-\>721, 760-\>764, 778-\>exit, 822, 824, 978, 1053, 1128-1135, 1175, 1399-1400, 1403-\>1408, 1421, 1510-\>exit, 1572, 1588-\>1592, 1609-\>1613, 1664-\>1668, 1673-\>1677, 1713-\>1718, 1772-\>1783, 1872-\>1879, 1928, 1930-\>1932 |
| src/llm\_minder/\_internal/logical\_item.py            |       17 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/rate\_limit\_coordinator.py |      168 |        5 |       42 |        1 |     97.14% |122, 293, 370-376 |
| src/llm\_minder/\_internal/removed\_names.py           |       14 |        2 |        2 |        0 |     87.50% |     32-33 |
| src/llm\_minder/\_internal/responses\_translation.py   |      188 |       13 |      116 |       16 |     90.46% |113-\>115, 124, 143, 171, 180, 193, 195-\>197, 213, 218, 231, 242, 244-\>246, 263, 282, 286, 296 |
| src/llm\_minder/\_internal/strategy\_lifecycle.py      |      172 |        5 |       42 |        5 |     95.33% |91, 143-\>141, 218, 245, 251-\>254, 257, 330 |
| src/llm\_minder/artifacts.py                           |      552 |       67 |      184 |       30 |     86.01% |165, 207-209, 244-245, 306-307, 323-324, 331, 335-336, 352-353, 368, 373, 386, 390, 395, 409-410, 414-\>418, 433, 435-438, 440, 463-464, 473, 479-483, 488, 494, 504, 510, 527-528, 532, 588-589, 596, 633, 651-652, 699-700, 725, 805-806, 826, 859, 862, 900-902, 906-907, 924, 939, 956-\>968, 961-962, 990, 999-\>1001, 1023-1024, 1058-\>1060 |
| src/llm\_minder/base.py                                |      963 |       42 |      266 |       21 |     94.22% |130, 358, 367, 687-\>exit, 690-\>exit, 744, 781-\>783, 834-\>839, 1018-\>1021, 1139, 1141, 1518, 1560-1561, 1727, 1823-1828, 1853, 1868-1871, 1892-\>exit, 1929-1930, 1960, 1964, 2007-\>exit, 2018, 2046-2047, 2067-2075, 2091-\>2098, 2171, 2220, 2239, 2243-\>2249, 2245-2248, 2279-2280, 2295-2297 |
| src/llm\_minder/budget.py                              |       13 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/cache\_groups.py                       |      221 |        3 |       88 |        3 |     98.06% |305, 420, 462 |
| src/llm\_minder/call\_pool.py                          |      133 |        4 |       34 |        2 |     96.41% |278, 281, 284-285 |
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
| src/llm\_minder/factory.py                             |       73 |        1 |       32 |        3 |     96.19% |113-\>exit, 124-\>exit, 227 |
| src/llm\_minder/gateway.py                             |        9 |        0 |        2 |        0 |    100.00% |           |
| src/llm\_minder/llm\_strategies.py                     |      203 |        8 |       56 |        8 |     93.82% |34-35, 69-\>67, 71, 324-\>exit, 336-\>exit, 423, 441, 454-\>457, 654-\>656, 659, 797-799 |
| src/llm\_minder/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/models.py                              |     1283 |       91 |      570 |       75 |     90.29% |62-64, 67-68, 112-\>exit, 114-115, 122-123, 152-168, 531-\>534, 609-\>exit, 617, 620, 658, 665, 691, 870-\>873, 875, 881-\>exit, 906, 987, 990, 1042, 1072, 1083, 1109-1110, 1126-\>1160, 1137, 1139, 1175, 1191-\>1209, 1230-\>1235, 1532-\>1581, 1535-\>1538, 1539-\>1575, 1555-1556, 1561, 1582-\>1586, 1584-1585, 1596, 1650-\>1660, 1653-\>1660, 1655-1656, 1676, 1679, 1682-\>1690, 1975-1980, 2019, 2089, 2102, 2119-\>2121, 2122-\>2124, 2125-\>2127, 2139-2142, 2144-2148, 2156, 2158, 2224-2225, 2421, 2424-\>2426, 2427-\>2429, 2436, 2441-\>2446, 2453-\>2455, 2480-\>2478, 2484-\>2482, 2488-\>2475, 2498-\>2475, 2503-\>2499, 2505-\>2499, 2508-\>2506, 2525, 2576-2577, 2667, 2690, 2700-2701, 2704, 2716-\>2725, 2720-2724, 2731-2732, 2830, 2834-2835, 2869, 2872, 2875-2878, 3006, 3026-\>3031, 3072-\>3074 |
| src/llm\_minder/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/llm\_minder/observers/metrics.py                   |      143 |        5 |       72 |       10 |     92.09% |56, 73-\>80, 83-\>89, 90-\>exit, 115-\>109, 118-\>exit, 129, 135-136, 153, 176-\>exit |
| src/llm\_minder/parallel.py                            |      435 |       22 |      108 |        8 |     94.48% |113-114, 117-118, 307, 318, 322, 330, 334, 338, 342, 346, 358, 391-\>393, 479-\>481, 488, 502, 510, 543-544, 587-\>589, 655, 660, 755, 945-\>950, 997 |
| src/llm\_minder/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/llm\_minder/provider\_calls.py                     |       22 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/provider\_output.py                    |      102 |        0 |       28 |        0 |    100.00% |           |
| src/llm\_minder/serialization.py                       |      250 |       39 |      102 |       21 |     82.39% |119, 133, 142-143, 225-226, 230, 255, 261, 271, 273, 353-354, 369, 379-380, 389, 391, 405, 476, 483, 490, 495, 524, 541, 577, 608-611, 663-664, 675-676, 682, 685-686, 698, 709 |
| src/llm\_minder/single.py                              |       35 |        3 |        4 |        1 |     89.74% | 54-55, 69 |
| src/llm\_minder/sqlite\_artifacts.py                   |      721 |       89 |      208 |       39 |     84.93% |88-89, 126-\>exit, 134, 139, 218, 226-227, 276-277, 284-286, 291-296, 319, 361-362, 421-422, 427, 431, 461, 528-531, 551, 583, 602-605, 608-609, 611-\>618, 614-615, 626, 655, 663, 677, 683, 689-690, 712, 714-\>exit, 728-\>730, 731, 737-\>739, 748, 756-762, 799-800, 811-\>810, 826-\>825, 840-841, 853, 863-865, 881-\>896, 910, 930, 947-950, 954-956, 1049-1051, 1067, 1134, 1141, 1144, 1148, 1159, 1190, 1286, 1352-1353, 1399-1400, 1408-1409, 1412-1413, 1417 |
| src/llm\_minder/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/strategies/errors.py                   |      264 |       19 |       96 |        4 |     93.61% |95-\>99, 116, 162-163, 168-169, 175-186, 397, 668-669, 706 |
| src/llm\_minder/strategies/rate\_limit.py              |       32 |        0 |        2 |        0 |    100.00% |           |
| src/llm\_minder/streaming.py                           |      211 |        1 |       76 |        4 |     98.26% |84-\>exit, 222-\>224, 278-\>281, 373, 379-\>exit |
| src/llm\_minder/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/llm\_minder/token\_extractor.py                    |      136 |        3 |       56 |        5 |     95.83% |88-\>98, 101-\>107, 103-\>107, 146-147, 154, 206-\>208 |
| **TOTAL**                                              | **9757** |  **556** | **3252** |  **353** | **92.58%** |           |


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