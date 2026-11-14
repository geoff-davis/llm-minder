# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/llm-minder/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/llm\_minder/\_\_init\_\_.py                        |       46 |        2 |        2 |        0 |     95.83% |   361-363 |
| src/llm\_minder/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/admission.py                |      421 |       16 |      118 |       14 |     94.43% |106, 114, 131, 133, 135, 155, 171, 173, 267, 299-300, 389, 403, 408-409, 523-\>521, 605 |
| src/llm\_minder/\_internal/artifact\_codec.py          |      193 |       14 |       58 |        7 |     91.63% |69, 73, 77, 124, 153, 189, 300-\>299, 376, 402, 410, 455, 471-474 |
| src/llm\_minder/\_internal/backoff.py                  |       21 |        2 |       12 |        0 |     93.94% |     20-21 |
| src/llm\_minder/\_internal/budget.py                   |       74 |        0 |       24 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/cache\_groups.py            |       40 |        0 |        6 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/capacity.py                 |      140 |        9 |       32 |        2 |     92.44% |43-44, 85-94, 202-208 |
| src/llm\_minder/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/cleanup.py                  |      291 |       17 |      106 |        8 |     91.69% |144-\>exit, 150, 157-158, 163-166, 207-209, 213, 242, 362-\>364, 368-\>exit, 449, 482-\>495, 536-540 |
| src/llm\_minder/\_internal/error\_logging.py           |      107 |       16 |       40 |        8 |     82.31% |42-\>45, 93, 107-\>117, 115-116, 118, 121-132, 149-\>170, 151, 153-156, 167-168 |
| src/llm\_minder/\_internal/event\_dispatcher.py        |       90 |        4 |       30 |        1 |     95.83% |82-83, 122-\>124, 147, 163 |
| src/llm\_minder/\_internal/execution\_state.py         |       59 |        1 |        6 |        1 |     96.92% |       125 |
| src/llm\_minder/\_internal/executor\_host.py           |       71 |        0 |        8 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/guardrails.py               |      165 |        4 |       56 |        6 |     95.48% |111-\>113, 136, 168, 207-\>exit, 225-226, 240-\>243 |
| src/llm\_minder/\_internal/input\_validation.py        |       43 |        1 |       12 |        1 |     96.36% |        84 |
| src/llm\_minder/\_internal/item\_executor.py           |      793 |       20 |      244 |       34 |     94.79% |209, 252-\>exit, 254-\>exit, 258-\>exit, 267-\>exit, 275-\>exit, 309-\>exit, 326, 342, 432, 492, 552-\>exit, 561-\>563, 564-\>566, 580-583, 697, 723-\>731, 770-\>774, 788-\>exit, 832, 834, 988, 1063, 1185, 1405-1406, 1409-\>1414, 1427, 1516-\>exit, 1578, 1594-\>1598, 1615-\>1619, 1670-\>1674, 1676-\>1678, 1681-\>1685, 1721-\>1726, 1780-\>1791, 1880-\>1887, 1936, 1938-\>1940 |
| src/llm\_minder/\_internal/logical\_item.py            |       17 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/rate\_limit\_coordinator.py |      168 |        5 |       42 |        1 |     97.14% |122, 293, 370-376 |
| src/llm\_minder/\_internal/removed\_names.py           |       14 |        2 |        2 |        0 |     87.50% |     32-33 |
| src/llm\_minder/\_internal/responses\_translation.py   |      191 |       13 |      118 |       16 |     90.61% |113-\>115, 124, 153, 181, 190, 203, 205-\>207, 223, 228, 241, 252, 254-\>256, 273, 292, 296, 306 |
| src/llm\_minder/\_internal/strategy\_lifecycle.py      |      172 |        5 |       42 |        5 |     95.33% |91, 143-\>141, 218, 245, 251-\>254, 257, 330 |
| src/llm\_minder/artifacts.py                           |      551 |       60 |      174 |       24 |     87.86% |165, 207-209, 244-245, 306-307, 368-369, 410-411, 429-430, 459-\>461, 469, 482, 485, 504-505, 512, 516-517, 536, 551, 575-576, 599, 601-604, 606, 629-630, 639, 645-649, 672-673, 677, 733-734, 741, 837, 917, 923-\>922, 950, 953, 991-993, 997-998, 1015, 1030, 1047-\>1059, 1052-1053, 1081, 1105-1106 |
| src/llm\_minder/base.py                                |      964 |       42 |      266 |       21 |     94.23% |131, 359, 368, 688-\>exit, 691-\>exit, 745, 782-\>784, 835-\>840, 1019-\>1022, 1140, 1142, 1524, 1566-1567, 1733, 1829-1834, 1859, 1874-1877, 1898-\>exit, 1935-1936, 1966, 1970, 2013-\>exit, 2024, 2052-2053, 2073-2081, 2097-\>2104, 2177, 2226, 2245, 2249-\>2255, 2251-2254, 2285-2286, 2301-2303 |
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
| src/llm\_minder/core/config.py                         |      238 |        4 |      142 |        4 |     97.89% |159, 161, 303, 579 |
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
| src/llm\_minder/parallel.py                            |      446 |       22 |      114 |       10 |     94.29% |113-114, 117-118, 310, 321, 325, 333, 337, 341, 345, 349, 361, 394-\>396, 453-\>455, 515-\>517, 517-\>519, 526, 540, 548, 581-582, 625-\>627, 693, 698, 793, 983-\>988, 1035 |
| src/llm\_minder/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/llm\_minder/provider\_calls.py                     |       22 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/provider\_output.py                    |      102 |        0 |       28 |        0 |    100.00% |           |
| src/llm\_minder/serialization.py                       |      250 |       39 |      102 |       21 |     82.39% |119, 133, 142-143, 225-226, 230, 255, 261, 271, 273, 353-354, 369, 379-380, 389, 391, 405, 476, 483, 490, 495, 524, 541, 577, 608-611, 663-664, 675-676, 682, 685-686, 698, 709 |
| src/llm\_minder/single.py                              |       35 |        3 |        4 |        1 |     89.74% | 54-55, 69 |
| src/llm\_minder/sqlite\_artifacts.py                   |      708 |       83 |      216 |       39 |     85.50% |119-\>exit, 127, 132, 211, 219-220, 265-267, 272-277, 369, 373, 403, 465-468, 488, 520, 534-537, 540-541, 543-\>550, 546-547, 558, 587, 595, 609, 615, 621-622, 644, 646-\>exit, 660-\>662, 663, 669-\>671, 680, 688-694, 731-732, 743-\>742, 758-\>757, 772-773, 785, 795-797, 813-\>828, 842, 867, 884-887, 891-893, 932-934, 1012-1014, 1035, 1102, 1109, 1112, 1116, 1127, 1158, 1254, 1320-1321, 1367-1368, 1376-1377, 1380-1381, 1385 |
| src/llm\_minder/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/strategies/errors.py                   |      264 |       19 |       96 |        4 |     93.61% |95-\>99, 116, 162-163, 168-169, 175-186, 397, 668-669, 706 |
| src/llm\_minder/strategies/rate\_limit.py              |       32 |        0 |        2 |        0 |    100.00% |           |
| src/llm\_minder/streaming.py                           |      211 |        1 |       76 |        4 |     98.26% |84-\>exit, 222-\>224, 278-\>281, 373, 379-\>exit |
| src/llm\_minder/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/llm\_minder/token\_extractor.py                    |      136 |        3 |       56 |        5 |     95.83% |88-\>98, 101-\>107, 103-\>107, 146-147, 154, 206-\>208 |
| **TOTAL**                                              | **9796** |  **540** | **3276** |  **350** | **92.78%** |           |


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