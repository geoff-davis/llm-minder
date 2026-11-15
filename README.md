# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/llm-minder/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/llm\_minder/\_\_init\_\_.py                        |       47 |        2 |        2 |        0 |     95.92% |   378-380 |
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
| src/llm\_minder/\_internal/guardrails.py               |      165 |        2 |       56 |        5 |     96.83% |111-\>113, 136, 168, 207-\>exit, 240-\>243 |
| src/llm\_minder/\_internal/input\_validation.py        |       43 |        1 |       12 |        1 |     96.36% |        84 |
| src/llm\_minder/\_internal/item\_executor.py           |      797 |       20 |      244 |       33 |     94.91% |209, 252-\>exit, 254-\>exit, 258-\>exit, 267-\>exit, 275-\>exit, 309-\>exit, 326, 342, 432, 492, 552-\>exit, 561-\>563, 564-\>566, 580-583, 697, 723-\>731, 801-\>exit, 845, 847, 1001, 1076, 1198, 1418-1419, 1422-\>1427, 1440, 1529-\>exit, 1591, 1607-\>1611, 1628-\>1632, 1683-\>1687, 1689-\>1691, 1694-\>1698, 1734-\>1739, 1793-\>1804, 1893-\>1900, 1949, 1951-\>1953 |
| src/llm\_minder/\_internal/logical\_item.py            |       17 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/rate\_limit\_coordinator.py |      168 |        5 |       42 |        1 |     97.14% |122, 293, 370-376 |
| src/llm\_minder/\_internal/removed\_names.py           |       14 |        2 |        2 |        0 |     87.50% |     32-33 |
| src/llm\_minder/\_internal/responses\_translation.py   |      191 |       13 |      118 |       16 |     90.61% |113-\>115, 124, 153, 181, 190, 203, 205-\>207, 223, 228, 241, 252, 254-\>256, 273, 292, 296, 306 |
| src/llm\_minder/\_internal/strategy\_lifecycle.py      |      172 |        5 |       42 |        5 |     95.33% |91, 143-\>141, 218, 245, 251-\>254, 257, 330 |
| src/llm\_minder/artifacts.py                           |      555 |       55 |      178 |       20 |     89.22% |165, 207-209, 244-245, 306-307, 368-369, 410-411, 429-430, 459-\>461, 469, 482, 485, 504-505, 536, 551, 575-576, 599, 601-604, 606, 629-630, 639, 645-649, 672-673, 677, 733-734, 741, 837, 917, 923-\>922, 953, 991-993, 997-998, 1015, 1030, 1052-1053, 1114-1115 |
| src/llm\_minder/base.py                                |      967 |       42 |      266 |       21 |     94.24% |132, 360, 369, 689-\>exit, 692-\>exit, 746, 783-\>785, 836-\>841, 1031-\>1034, 1152, 1154, 1536, 1578-1579, 1745, 1841-1846, 1871, 1886-1889, 1910-\>exit, 1947-1948, 1978, 1982, 2025-\>exit, 2036, 2064-2065, 2085-2093, 2109-\>2116, 2189, 2238, 2257, 2261-\>2267, 2263-2266, 2297-2298, 2313-2315 |
| src/llm\_minder/budget.py                              |       13 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/cache\_groups.py                       |      221 |        3 |       88 |        3 |     98.06% |305, 420, 462 |
| src/llm\_minder/call\_pool.py                          |      138 |        4 |       34 |        2 |     96.51% |282, 285, 288-289 |
| src/llm\_minder/callable\_strategy.py                  |      141 |        4 |       50 |        6 |     94.76% |56-\>exit, 69-\>exit, 88, 131, 172, 246 |
| src/llm\_minder/categories.py                          |       96 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/classifiers/\_\_init\_\_.py            |        5 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/classifiers/gemini.py                  |       92 |        7 |       50 |        5 |     91.55% |27-\>30, 41, 55-\>52, 60-61, 115-116, 223, 231 |
| src/llm\_minder/classifiers/openai.py                  |       71 |        5 |       40 |        1 |     94.59% |114, 146-147, 181-182 |
| src/llm\_minder/classifiers/openrouter.py              |       37 |        2 |       16 |        0 |     96.23% |     85-86 |
| src/llm\_minder/classifiers/pydantic\_ai.py            |       16 |        2 |        6 |        0 |     90.91% |     16-17 |
| src/llm\_minder/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/core/config.py                         |      243 |        4 |      142 |        4 |     97.92% |159, 161, 303, 590 |
| src/llm\_minder/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/factory.py                             |       72 |        1 |       32 |        3 |     96.15% |113-\>exit, 124-\>exit, 227 |
| src/llm\_minder/gateway.py                             |        9 |        0 |        2 |        0 |    100.00% |           |
| src/llm\_minder/limits.py                              |      149 |        0 |       52 |        0 |    100.00% |           |
| src/llm\_minder/llm\_strategies.py                     |      203 |        7 |       56 |        7 |     94.59% |34-35, 69-\>67, 71, 324-\>exit, 336-\>exit, 423, 454-\>457, 654-\>656, 659, 797-799 |
| src/llm\_minder/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/models/\_\_init\_\_.py                 |        6 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/models/\_deepseek.py                   |      155 |       20 |       66 |       11 |     83.26% |47, 70, 80-81, 84, 96-\>105, 100-104, 187, 191-192, 226, 229, 232-235, 363, 383-\>388 |
| src/llm\_minder/models/\_gemini.py                     |      301 |       17 |      114 |       21 |     90.84% |110-\>exit, 118, 121, 159, 166, 192, 371-\>374, 376, 382-\>exit, 407, 488, 491, 543, 573, 584, 610-611, 627-\>661, 638, 640, 676, 692-\>710, 731-\>736 |
| src/llm\_minder/models/\_openai.py                     |      159 |        5 |       86 |       15 |     91.84% |94-95, 291, 294-\>296, 297-\>299, 306, 311-\>316, 323-\>325, 350-\>348, 354-\>352, 358-\>345, 368-\>345, 373-\>369, 375-\>369, 378-\>376, 395 |
| src/llm\_minder/models/\_openai\_compatible.py         |      255 |       10 |      108 |       11 |     94.21% |321-\>370, 324-\>327, 328-\>364, 344-345, 350, 371-\>375, 373-374, 385, 444-\>454, 447-\>454, 449-450, 470, 473, 476-\>484 |
| src/llm\_minder/models/\_openrouter.py                 |       34 |        2 |       10 |        1 |     93.18% |     64-65 |
| src/llm\_minder/models/\_responses.py                  |      159 |       18 |       70 |       12 |     84.28% |122-127, 166, 236, 249, 266-\>268, 269-\>271, 272-\>274, 286-289, 291-295, 303, 305, 312-313 |
| src/llm\_minder/models/\_sdk.py                        |       39 |        9 |        2 |        1 |     75.61% |20-22, 25-26, 67-\>exit, 69-70, 77-78 |
| src/llm\_minder/models/\_shared.py                     |      241 |       10 |      114 |        3 |     95.77% |60-76, 439-\>442, 483-\>485 |
| src/llm\_minder/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/llm\_minder/observers/metrics.py                   |      145 |        5 |       72 |        9 |     92.63% |66, 83-\>90, 100-\>exit, 125-\>119, 128-\>exit, 139, 145-146, 163, 186-\>exit |
| src/llm\_minder/parallel.py                            |      477 |       22 |      124 |       10 |     94.68% |117-118, 121-122, 315, 326, 330, 338, 342, 346, 350, 354, 366, 399-\>401, 458-\>460, 564-\>566, 566-\>568, 575, 589, 622, 655-656, 706-\>708, 774, 779, 874, 1064-\>1069, 1116 |
| src/llm\_minder/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/llm\_minder/provider\_calls.py                     |       22 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/provider\_output.py                    |      102 |        0 |       28 |        0 |    100.00% |           |
| src/llm\_minder/serialization.py                       |      250 |       39 |      102 |       21 |     82.39% |119, 133, 142-143, 225-226, 230, 255, 261, 271, 273, 353-354, 369, 379-380, 389, 391, 405, 476, 483, 490, 495, 524, 541, 577, 608-611, 663-664, 675-676, 682, 685-686, 698, 709 |
| src/llm\_minder/single.py                              |       35 |        3 |        4 |        1 |     89.74% | 54-55, 69 |
| src/llm\_minder/sqlite\_artifacts.py                   |      708 |       79 |      216 |       35 |     86.36% |119-\>exit, 127, 211, 219-220, 265-267, 272-277, 369, 373, 465-468, 488, 534-537, 540-541, 543-\>550, 546-547, 558, 595, 609, 615, 621-622, 644, 646-\>exit, 660-\>662, 663, 669-\>671, 680, 688-694, 731-732, 743-\>742, 758-\>757, 772-773, 785, 795-797, 813-\>828, 842, 867, 884-887, 891-893, 932-934, 1012-1014, 1035, 1102, 1109, 1112, 1116, 1127, 1158, 1254, 1320-1321, 1367-1368, 1376-1377, 1380-1381, 1385 |
| src/llm\_minder/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/strategies/errors.py                   |      264 |       19 |       96 |        4 |     93.61% |95-\>99, 116, 162-163, 168-169, 175-186, 397, 668-669, 706 |
| src/llm\_minder/strategies/rate\_limit.py              |       32 |        0 |        2 |        0 |    100.00% |           |
| src/llm\_minder/streaming.py                           |      214 |        1 |       74 |        3 |     98.61% |83-\>exit, 221-\>223, 277-\>280, 384 |
| src/llm\_minder/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/llm\_minder/token\_extractor.py                    |      136 |        3 |       56 |        5 |     95.83% |88-\>98, 101-\>107, 103-\>107, 146-147, 154, 206-\>208 |
| **TOTAL**                                              | **10068** |  **528** | **3340** |  **337** | **93.15%** |           |


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