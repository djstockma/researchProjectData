# Per-task results: 2x2GPU parallel, Job B

Configuration: 2x MI250X, 8h wall, --task-timeout 1800, tasks 21-40.
20/20 tasks completed. 7/20 PASS (35%).

| Task | Result | Steps | Wall (s) | Inf/step (s) |
|------|--------|-------|----------|--------------|
| mergesort | PASS | 4 | 452 | 107 |
| minimum_spanning_tree | FAIL | 2 | 218 | 106 |
| next_palindrome | PASS | 9 | 1881 | 208 |
| next_permutation | FAIL | 4 | 1906 | 473 ¹ |
| pascal | FAIL | 1 | 1924 | - ¹ |
| possible_change | FAIL | 1 | 1804 | - ¹ |
| powerset | FAIL | 5 | 544 | 108 |
| quicksort | FAIL | 3 | 1851 | 615 ¹ |
| reverse_linked_list | FAIL | 1 | 1957 | - ¹ |
| rpn_eval | PASS | 8 | 877 | 109 |
| shortest_path_length | FAIL | 10 | 1848 | 184 |
| shortest_path_lengths | FAIL | 2 | 216 | 106 |
| shortest_paths | PASS | 5 | 541 | 107 |
| shunting_yard | FAIL | 7 | 1843 | 263 |
| sieve | FAIL | 6 | 646 | 107 |
| sqrt | FAIL | 11 | 1873 | 135 |
| subsequences | FAIL | 5 | 1917 | 382 ¹ |
| to_base | PASS | 5 | 542 | 108 |
| topological_ordering | PASS | 9 | 1108 | 122 |
| wrap | PASS | 5 | 553 | 110 |

¹ Abnormally high wall time or inf/step: format error loop or severe context explosion.
Tasks hitting ~1800s were cut by the per-task timeout.
