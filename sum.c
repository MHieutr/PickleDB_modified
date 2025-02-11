#include<stdio.h>
#include<stdlib.h>
#include<omp.h>


void initial_array(int* array, int size){
    for (int i= 0; i < size; i++){
        array[i] = rand() & 100;
    }
}

#define N 100000000
int main(){
    long long s = 0;

    int* array = (int*) malloc(N * sizeof(int));
    if (array == NULL){
        printf("Cannot allocate memory");
        return 1;
    }

    initial_array(array, N);

    omp_set_num_threads(8); // Modify the number of used threads for the task

    double start_time = omp_get_wtime();
    #pragma omp parallel for reduction(+: s)
    for (int i = 0; i < N; i++){
        s += array[i];
    }
    double end_time = omp_get_wtime();

    printf("Sum of array: %lld\n", s);
    printf("Time taken with 2 Threads: %fs", end_time - start_time);

    free(array);

    return 0;
    
}

// Sum of array: 5000015328
// Time taken with 2 Threads: 0.143000s
// Sum of array: 5000015328
// Time taken with 4 Threads: 0.089000s
// Sum of array: 5000015328
// Time taken with 6 Threads: 0.069000s
// Sum of array: 5000015328
// Time taken with 8 Threads: 0.051000s