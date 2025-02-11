# Performance Comparison: Multiprocessing, Numpy Threads, and OpenMP

This repository presents a comparison of different parallel processing techniques, including multiprocessing, NumPy threads, and OpenMP, for operations such as nearest neighbor searches, summation, and dot product computations.

## 1. Multiprocessing vs. Simple Processing (50,000 Samples)

### Nearest Neighbors Results:
- **Name:** Name_23509, **Distance:** 16.3818
- **Name:** Name_37121, **Distance:** 16.5299

**Time taken with multiprocessing:** `4.0298` seconds

---

## 2. NumPy and Threads

### Nearest Neighbors Results:
#### Using 2 Threads:
- **Name:** Name_64055, **Distance:** 0.8196
- **Name:** Name_72923, **Distance:** 0.8252

**Time taken:** `3.7775` seconds

#### Using 4 Threads:
- **Name:** Name_64055, **Distance:** 0.8196
- **Name:** Name_72923, **Distance:** 0.8252

**Time taken:** `2.3921` seconds

---

## 3. OpenMP Sum and Dot Product

### Sum of Array (N = 100,000,000)
| Threads | Sum of Array | Time Taken (s) |
|---------|-------------|---------------|
| 2       | 5,000,015,328 | 0.143 |
| 4       | 5,000,015,328 | 0.089 |
| 6       | 5,000,015,328 | 0.069 |
| 8       | 5,000,015,328 | 0.051 |

### Dot Product Computation

#### Case 1: Matrix Dimensions
- **Row A (N):** 500
- **Column A (Row B, M):** 600
- **Column B (L):** 500

| Threads | Time Taken (s) |
|---------|---------------|
| 2       | 0.420 |
| 4       | 0.364 |
| 6       | 0.249 |
| 8       | 0.249 |

#### Case 2: Larger Matrix Dimensions
- **Row A (N):** 1000
- **Column A (Row B, M):** 2000
- **Column B (L):** 1000

| Threads | Time Taken (s) |
|---------|---------------|
| 8       | 4.508 |


