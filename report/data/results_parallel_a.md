# Per-task results: 2x2GPU parallel, Job A

Configuration: 2x MI250X, 8h wall, --task-timeout 1800, tasks 1-20.
20/20 tasks completed. 9/20 PASS (45%).

| Task | Result | Steps | Wall (s) | Inf/step (s) |
|------|--------|-------|----------|--------------|
| bitcount | PASS | 4 | 437 | 106 |
| breadth_first_search | PASS | 6 | 647 | 107 |
| bucketsort | PASS | 3 | 321 | 106 |
| depth_first_search | FAIL | 5 | 527 | 105 |
| detect_cycle | FAIL | 4 | 434 | 107 |
| find_first_in_sorted | FAIL | 2 | 243 | 106 |
| find_in_sorted | FAIL | 1 | 1997 | - ¹ |
| flatten | PASS | 4 | 433 | 107 |
| gcd | PASS | 4 | 437 | 108 |
| get_factors | FAIL | 13 | 1825 | 140 |
| hanoi | FAIL | 3 | 327 | 108 |
| is_valid_parenthesization | PASS | 10 | 1235 | 123 |
| kheapsort | PASS | 7 | 771 | 110 |
| knapsack | PASS | 13 | 1902 | 146 |
| kth | FAIL | 2 | 218 | 107 |
| lcs_length | FAIL | 4 | 434 | 108 |
| levenshtein | FAIL | 5 | 1932 | 385 ¹ |
| lis | FAIL | 4 | 433 | 107 |
| longest_common_subsequence | PASS | 13 | 1832 | 140 |
| max_sublist_sum | FAIL | 4 | 1814 | 452 ¹ |

¹ Abnormally high wall time or inf/step: format error loop or severe context explosion.
Tasks hitting ~1800s were cut by the per-task timeout.
