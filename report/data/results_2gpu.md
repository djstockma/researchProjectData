# Per-task results: 2GPU serial

Configuration: 1 job, 2x MI250X, 8h wall time.
24/40 tasks completed before hitting wall limit. 10/24 PASS (42%).

| Task | Result | Steps | Wall (s) | Inf/step (s) |
|------|--------|-------|----------|--------------|
| bitcount | PASS | 4 | 474 | 104 |
| breadth_first_search | FAIL | 7 | 764 | 108 |
| bucketsort | PASS | 3 | 319 | 105 |
| depth_first_search | FAIL | 5 | 531 | 104 |
| detect_cycle | FAIL | 4 | 421 | 103 |
| find_first_in_sorted | FAIL | 2 | 239 | 103 |
| find_in_sorted | FAIL | 1 | 2201 | - ¹ |
| flatten | PASS | 4 | 435 | 106 |
| gcd | PASS | 6 | 668 | 110 |
| get_factors | FAIL | 8 | 965 | 119 |
| hanoi | FAIL | 3 | 324 | 106 |
| is_valid_parenthesization | PASS | 10 | 1217 | 121 |
| kheapsort | PASS | 7 | 763 | 108 |
| knapsack | FAIL | 14 | 2220 | 158 |
| kth | FAIL | 2 | 228 | 106 |
| lcs_length | FAIL | 4 | 422 | 105 |
| levenshtein | FAIL | 5 | 2025 | 403 ¹ |
| lis | FAIL | 4 | 431 | 106 |
| longest_common_subsequence | FAIL | 7 | 2209 | 315 ¹ |
| max_sublist_sum | PASS | 12 | 1524 | 126 |
| mergesort | PASS | 6 | 2273 | 376 ¹ |
| minimum_spanning_tree | FAIL | 2 | 217 | 105 |
| next_palindrome | PASS | 7 | 2069 | 294 ¹ |
| next_permutation | PASS | 10 | 1619 | 134 |

¹ Abnormally high wall time or inf/step: format error loop or severe context explosion.
