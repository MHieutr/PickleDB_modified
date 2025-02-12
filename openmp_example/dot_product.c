#include<stdio.h>
#include<time.h>
#include<omp.h>
#include<stdlib.h>

int** allocate_matrix(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
    }
    return matrix;
}

void fill_matrix_random(int** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 10; 
        }
    }
}

void free_matrix(int** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main(){
    int N, M, L;

    printf("Row A (N): ");
    scanf("%d", &N);
    printf("Column A (Row B, M): ");
    scanf("%d", &M);
    printf("Column B (L): ");
    scanf("%d", &L);

    if (M <= 0 || N <= 0 || L <= 0) {
        printf("Matrix Error!\n");
        return 1;
    }

    srand(time(NULL));
    int** A = allocate_matrix(N, M);
    int** B = allocate_matrix(M, L);
    int** C = allocate_matrix(N, L);

    fill_matrix_random(A, N, M);
    fill_matrix_random(B, M, L);

    omp_set_num_threads(8); // Modify the number of used threads for the task

    double start_time = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++){
        for (int j = 0; j < L; j++){
            C[i][j] = 0;
            for (int k = 0; k < M; k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    double end_time = omp_get_wtime();
    printf("Time taken with 2 Threads: %fs", end_time - start_time);

    free_matrix(A, N);
    free_matrix(B, M);
    free_matrix(C, N);

    return 0;
}

// Row A (N): 500 
// Column A (Row B, M): 600
// Column B (L): 500
// Time taken with 2 Threads: 0.420000s
// Time taken with 4 Threads: 0.364000s
// Time taken with 6 Threads: 0.249000s
// Time taken with 8 Threads: 0.249000s
// Row A (N): 1000
// Column A (Row B, M): 2000
// Column B (L): 1000
// Time taken with 8 Threads: 4.508000s