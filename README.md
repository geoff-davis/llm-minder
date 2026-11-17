# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/geoff-davis/llm-minder/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                   |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|------------------------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| src/llm\_minder/\_\_init\_\_.py                        |       48 |        2 |        2 |        0 |     96.00% |   390-392 |
| src/llm\_minder/\_internal/\_\_init\_\_.py             |        0 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/admission.py                |      421 |       16 |      118 |       14 |     94.43% |106, 114, 131, 133, 135, 155, 171, 173, 267, 299-300, 389, 403, 408-409, 523-\>521, 605 |
| src/llm\_minder/\_internal/artifact\_codec.py          |      207 |       14 |       66 |        7 |     92.31% |69, 73, 77, 124, 153, 213, 324-\>323, 411, 437, 445, 490, 506-509 |
| src/llm\_minder/\_internal/attempt\_admission.py       |       47 |        3 |       18 |        2 |     92.31% |36-\>38, 62, 86-87 |
| src/llm\_minder/\_internal/backoff.py                  |       21 |        2 |       12 |        0 |     93.94% |     20-21 |
| src/llm\_minder/\_internal/budget.py                   |       74 |        0 |       24 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/cache\_groups.py            |       40 |        0 |        6 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/capacity.py                 |      140 |        9 |       32 |        2 |     92.44% |43-44, 85-94, 202-208 |
| src/llm\_minder/\_internal/classifier\_resolver.py     |       38 |        0 |        4 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/cleanup.py                  |      291 |       17 |      106 |        8 |     91.69% |144-\>exit, 150, 157-158, 163-166, 207-209, 213, 242, 362-\>364, 368-\>exit, 449, 482-\>495, 536-540 |
| src/llm\_minder/\_internal/error\_logging.py           |      107 |       16 |       40 |        8 |     82.31% |42-\>45, 93, 107-\>117, 115-116, 118, 121-132, 149-\>170, 151, 153-156, 167-168 |
| src/llm\_minder/\_internal/event\_dispatcher.py        |       90 |        3 |       30 |        1 |     96.67% |83, 122-\>124, 147, 163 |
| src/llm\_minder/\_internal/execution\_state.py         |       59 |        1 |        6 |        1 |     96.92% |       125 |
| src/llm\_minder/\_internal/executor\_host.py           |       73 |        0 |       10 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/guardrails.py               |      243 |       10 |       82 |        9 |     93.54% |76-\>exit, 167-\>169, 182, 184, 223, 231-232, 270-272, 282, 318, 364-\>exit |
| src/llm\_minder/\_internal/input\_validation.py        |       43 |        1 |       12 |        1 |     96.36% |        84 |
| src/llm\_minder/\_internal/item\_executor.py           |      840 |       20 |      260 |       33 |     95.18% |215, 258-\>exit, 260-\>exit, 264-\>exit, 273-\>exit, 281-\>exit, 315-\>exit, 332, 348, 438, 498, 558-\>exit, 577-\>579, 580-\>582, 596-599, 757, 783-\>791, 861-\>exit, 905, 907, 1061, 1136, 1258, 1478-1479, 1482-\>1487, 1500, 1589-\>exit, 1651, 1675-\>1679, 1696-\>1700, 1780-\>1784, 1795-\>1799, 1842-\>1848, 1848-\>1853, 1921-\>1932, 2022-\>2029, 2078, 2080-\>2082 |
| src/llm\_minder/\_internal/logical\_item.py            |       17 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/\_internal/messages.py                 |      188 |       10 |      100 |       10 |     93.06% |104, 126, 202, 207, 221, 226, 234, 266, 349, 368 |
| src/llm\_minder/\_internal/rate\_limit\_coordinator.py |      168 |        5 |       42 |        1 |     97.14% |122, 293, 370-376 |
| src/llm\_minder/\_internal/removed\_names.py           |       14 |        2 |        2 |        0 |     87.50% |     32-33 |
| src/llm\_minder/\_internal/responses\_translation.py   |      191 |       13 |      118 |       16 |     90.61% |113-\>115, 124, 153, 181, 190, 203, 205-\>207, 223, 228, 241, 252, 254-\>256, 273, 292, 296, 306 |
| src/llm\_minder/\_internal/store\_reader.py            |      336 |       25 |      114 |       11 |     92.00% |106, 182, 256-257, 273, 281, 393-395, 403, 407, 411, 415, 420, 440, 443, 458-459, 514, 517-518, 537-538, 551-552, 589-\>591 |
| src/llm\_minder/\_internal/strategy\_lifecycle.py      |      172 |        5 |       42 |        5 |     95.33% |91, 143-\>141, 218, 245, 251-\>254, 257, 330 |
| src/llm\_minder/artifacts.py                           |      555 |       55 |      178 |       20 |     89.22% |166, 208-210, 245-246, 307-308, 369-370, 411-412, 430-431, 460-\>462, 470, 483, 486, 505-506, 537, 552, 576-577, 600, 602-605, 607, 630-631, 640, 646-650, 673-674, 678, 734-735, 742, 838, 918, 924-\>923, 954, 992-994, 998-999, 1016, 1031, 1058-1059, 1120-1121 |
| src/llm\_minder/attempt\_admission.py                  |       14 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/base.py                                |      974 |       42 |      270 |       21 |     94.29% |132, 380, 389, 714-\>exit, 717-\>exit, 771, 808-\>810, 861-\>866, 1056-\>1059, 1177, 1179, 1561, 1603-1604, 1770, 1866-1871, 1896, 1911-1914, 1935-\>exit, 1972-1973, 2003, 2007, 2050-\>exit, 2061, 2089-2090, 2110-2118, 2134-\>2141, 2214, 2263, 2282, 2286-\>2292, 2288-2291, 2322-2323, 2338-2340 |
| src/llm\_minder/budget.py                              |       13 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/cache\_groups.py                       |      221 |        3 |       88 |        3 |     98.06% |305, 420, 462 |
| src/llm\_minder/call\_pool.py                          |      142 |        4 |       36 |        2 |     96.63% |289, 292, 295-296 |
| src/llm\_minder/callable\_strategy.py                  |      141 |        4 |       50 |        6 |     94.76% |56-\>exit, 69-\>exit, 88, 131, 172, 246 |
| src/llm\_minder/categories.py                          |       98 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/classifiers/\_\_init\_\_.py            |        6 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/classifiers/anthropic.py               |       82 |        2 |       46 |        0 |     98.44% |   102-103 |
| src/llm\_minder/classifiers/gemini.py                  |       92 |        7 |       50 |        5 |     91.55% |27-\>30, 41, 55-\>52, 60-61, 115-116, 223, 231 |
| src/llm\_minder/classifiers/openai.py                  |       71 |        5 |       40 |        1 |     94.59% |114, 146-147, 181-182 |
| src/llm\_minder/classifiers/openrouter.py              |       37 |        2 |       16 |        0 |     96.23% |     85-86 |
| src/llm\_minder/classifiers/pydantic\_ai.py            |       16 |        2 |        6 |        0 |     90.91% |     16-17 |
| src/llm\_minder/core/\_\_init\_\_.py                   |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/core/config.py                         |      252 |        4 |      150 |        4 |     98.01% |161, 163, 315, 677 |
| src/llm\_minder/core/protocols.py                      |        2 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/factory.py                             |       78 |        1 |       38 |        3 |     96.55% |122-\>exit, 133-\>exit, 239 |
| src/llm\_minder/gateway.py                             |        9 |        0 |        2 |        0 |    100.00% |           |
| src/llm\_minder/limits.py                              |      149 |        0 |       52 |        0 |    100.00% |           |
| src/llm\_minder/llm\_strategies.py                     |      217 |        6 |       56 |        7 |     95.24% |34-35, 69-\>67, 71, 348-\>exit, 360-\>exit, 447, 478-\>481, 712-\>714, 875-877 |
| src/llm\_minder/middleware/\_\_init\_\_.py             |        2 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/middleware/base.py                     |       11 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/models/\_\_init\_\_.py                 |        7 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/models/\_anthropic.py                  |      292 |       22 |      112 |       10 |     91.58% |66-67, 69, 94-95, 358, 437-438, 450-451, 462, 465, 494, 508-\>518, 511-\>518, 513-514, 535, 538-\>543, 540-541, 612-615, 632-\>634 |
| src/llm\_minder/models/\_deepseek.py                   |      155 |       20 |       66 |       11 |     83.26% |47, 70, 80-81, 84, 96-\>105, 100-104, 187, 191-192, 226, 229, 232-235, 363, 383-\>388 |
| src/llm\_minder/models/\_gemini.py                     |      313 |       17 |      122 |       21 |     91.26% |111-\>exit, 119, 122, 171, 185, 211, 390-\>393, 395, 401-\>exit, 426, 519, 522, 574, 604, 615, 641-642, 658-\>692, 669, 671, 707, 723-\>741, 762-\>767 |
| src/llm\_minder/models/\_openai.py                     |      159 |        5 |       86 |       14 |     92.24% |94-95, 291, 294-\>296, 297-\>299, 306, 311-\>316, 350-\>348, 354-\>352, 358-\>345, 368-\>345, 373-\>369, 375-\>369, 378-\>376, 395 |
| src/llm\_minder/models/\_openai\_compatible.py         |      255 |       10 |      108 |       11 |     94.21% |321-\>370, 324-\>327, 328-\>364, 344-345, 350, 371-\>375, 373-374, 385, 444-\>454, 447-\>454, 449-450, 470, 473, 476-\>484 |
| src/llm\_minder/models/\_openrouter.py                 |       34 |        2 |       10 |        1 |     93.18% |     64-65 |
| src/llm\_minder/models/\_responses.py                  |      159 |       18 |       70 |       12 |     84.28% |123-128, 169, 239, 252, 269-\>271, 272-\>274, 275-\>277, 289-292, 294-298, 306, 308, 315-316 |
| src/llm\_minder/models/\_sdk.py                        |       47 |       11 |        2 |        0 |     77.55% |22-24, 27-28, 31-32, 93-94, 101-102 |
| src/llm\_minder/models/\_shared.py                     |      241 |       10 |      114 |        3 |     95.77% |60-76, 439-\>442, 483-\>485 |
| src/llm\_minder/observers/\_\_init\_\_.py              |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/observers/base.py                      |       25 |        1 |        0 |        0 |     96.00% |        63 |
| src/llm\_minder/observers/metrics.py                   |      145 |        5 |       72 |        9 |     92.63% |66, 83-\>90, 100-\>exit, 125-\>119, 128-\>exit, 139, 145-146, 163, 186-\>exit |
| src/llm\_minder/parallel.py                            |      480 |       22 |      126 |       10 |     94.72% |118-119, 122-123, 317, 328, 332, 340, 344, 348, 352, 356, 368, 401-\>403, 460-\>462, 572-\>574, 574-\>576, 586, 600, 633, 666-667, 717-\>719, 785, 790, 885, 1075-\>1080, 1127 |
| src/llm\_minder/parsing.py                             |       63 |        0 |       18 |        0 |    100.00% |           |
| src/llm\_minder/provider\_calls.py                     |       22 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/provider\_output.py                    |      102 |        0 |       28 |        0 |    100.00% |           |
| src/llm\_minder/serialization.py                       |      250 |       39 |      102 |       21 |     82.39% |119, 133, 142-143, 225-226, 230, 255, 261, 271, 273, 353-354, 369, 379-380, 389, 391, 405, 476, 483, 490, 495, 524, 541, 584, 615-618, 670-671, 682-683, 689, 692-693, 705, 716 |
| src/llm\_minder/single.py                              |       39 |        3 |        6 |        1 |     91.11% | 56-57, 71 |
| src/llm\_minder/sqlite\_artifacts.py                   |      713 |       79 |      218 |       35 |     86.47% |119-\>exit, 127, 211, 219-220, 265-267, 272-277, 369, 373, 465-468, 488, 534-537, 540-541, 543-\>550, 546-547, 558, 595, 609, 615, 621-622, 644, 646-\>exit, 660-\>662, 663, 669-\>671, 680, 688-694, 731-732, 743-\>742, 758-\>757, 772-773, 785, 795-797, 813-\>828, 842, 867, 884-887, 891-893, 932-934, 1012-1014, 1035, 1102, 1109, 1112, 1116, 1127, 1158, 1250, 1316-1317, 1363-1364, 1372-1373, 1376-1377, 1381 |
| src/llm\_minder/strategies/\_\_init\_\_.py             |        3 |        0 |        0 |        0 |    100.00% |           |
| src/llm\_minder/strategies/errors.py                   |      267 |       11 |       98 |        4 |     95.89% |95-\>99, 116, 162-163, 168-169, 183-184, 397, 687-688, 725 |
| src/llm\_minder/strategies/rate\_limit.py              |       32 |        0 |        2 |        0 |    100.00% |           |
| src/llm\_minder/streaming.py                           |      219 |        1 |       76 |        3 |     98.64% |93-\>exit, 236-\>238, 292-\>295, 401 |
| src/llm\_minder/token\_estimation.py                   |       35 |        0 |        4 |        0 |    100.00% |           |
| src/llm\_minder/token\_extractor.py                    |      136 |        3 |       56 |        4 |     96.35% |88-\>98, 103-\>107, 146-147, 154, 206-\>208 |
| **TOTAL**                                              | **11249** |  **590** | **3820** |  **371** | **93.24%** |           |


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