#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <string.h>
#include <mpi.h>
#define NREP 75

//export OMPI_MCA_btl_vader_single_copy_mechanism=none
//bin/bash

void reduce_array_mpi_aligned_vec(float * array, int length, int reduction_operator, int my_rank, int nproc, float * root_results, float * own_result);
void reduce_array_mpi_unaligned_vec(float * array, int length, int reduction_operator, int my_rank, int nproc, float * root_results, float * own_result);
void standard_mpi_reduce(float * send_array, float * root_array, int length, int reduction_operator, int my_rank, float * result);

float reduce_array_aligned(float * array, int length, int reduction_operator);
float reduce_array_unaligned(float * array, int length, int reduction_operator);

float reduce_arr_to_flt_vect(float * array, int mode);
float reduce_arr_to_flt(float * array, int length, int reduction_operator);



int main(int argc, char * argv[]){
    int my_rank, nproc, nrep = NREP;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int length = atoi(argv[1]);
    int reduction_operator = atoi(argv[2]);
    float * array, * root_results, * standard_reduce;
    float * partial_array_aligned = (float *) aligned_alloc(32, length/nproc * sizeof(float));
    float * partial_array_unaligned = (float *) calloc(length/nproc, sizeof(float));
    float * partial_array = (float *) calloc(length/nproc, sizeof(float));
    float result = 0.0, result2 = 0.0, result3 = 0.0, seq_time = 0.08476;

    double start2, start3, start4, end2, end3, end4, final_unaligned = 0.0, final_aligned = 0.0;


    if (my_rank == 0) {
        array = (float *) calloc(length, sizeof(float));
        standard_reduce = (float *) calloc(length/nproc, sizeof(float));
        root_results = (float *) calloc(nproc, sizeof(float));
        //INIZIALIZZAZIONE
        for(int i=0; i<length; i++){
            array[i] = i % 5 + 1;
        }
    }

    MPI_Scatter(
        array,
        length/nproc,
        MPI_FLOAT,
        partial_array,
        length/nproc,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );
    memcpy(partial_array_unaligned, partial_array, length/nproc * sizeof(float));
    memcpy(partial_array_aligned, partial_array, length/nproc * sizeof(float));

    // using normal mpi reduce and sequential reduction on the root
    MPI_Barrier(MPI_COMM_WORLD);
    start2 = MPI_Wtime();
    for(int i = 0; i < nrep; i++){
        standard_mpi_reduce(partial_array_unaligned, standard_reduce, length/nproc, reduction_operator, my_rank, &result3);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    end2 = (MPI_Wtime() - start2) / nrep;

    // using aligned vectorization 
    for(int i = 0; i < nrep; i++){
        start3 = MPI_Wtime();
        
        reduce_array_mpi_aligned_vec(partial_array_aligned, length/nproc, reduction_operator, my_rank, nproc, root_results, &result);
        MPI_Barrier(MPI_COMM_WORLD);

        end3 = MPI_Wtime() - start3;
        final_aligned += end3;
        memcpy(partial_array_aligned, partial_array, length/nproc * sizeof(float));
    }
    final_aligned /= nrep;

    // using unaligned vectorization 
    for(int i = 0; i < nrep; i++){
        start4 = MPI_Wtime();
        
        reduce_array_mpi_unaligned_vec(partial_array_unaligned, length/nproc, reduction_operator, my_rank, nproc, root_results, &result2);
        MPI_Barrier(MPI_COMM_WORLD);

        end4 = MPI_Wtime() - start4;
        final_unaligned += end4;
        memcpy(partial_array_unaligned, partial_array, length/nproc * sizeof(float));
    }
    final_unaligned /= nrep;

    if (my_rank == 0) {
        printf("Times of execution\nTime mpi standard reduce: %f\nTime vect unaligned: %f\nTime vect aligned: %f\n", end2, final_unaligned, final_aligned);
        printf("\nImprovements\nWith standard MPI_Reduce: %.4fx\nWith aligned vectorization %.4fx\nWith unaligned vectorization %.4fx\n\n", seq_time/end2, seq_time/final_aligned, seq_time/final_unaligned);
        free(array);
        free(root_results);
        free(standard_reduce);
    }
    free(partial_array_aligned);
    free(partial_array_unaligned);
    free(partial_array);

    
    MPI_Finalize();

	return 0;
}

void standard_mpi_reduce(float * send_array, float * root_array, int length, int reduction_operator, int my_rank, float * result) {
    MPI_Op op;
    switch(reduction_operator) {
        case 0: op = MPI_SUM;
        break;
        case 1: op = MPI_PROD;
        break;
        case 2: op = MPI_MIN;
        break;
        case 3: op = MPI_MAX;
        break;
    }

    MPI_Reduce(
        send_array,
        root_array,
        length,
        MPI_FLOAT,
        op,
        0,
        MPI_COMM_WORLD     
    );

    if(my_rank == 0){
        *(result) = reduce_arr_to_flt(root_array, length, reduction_operator);
    }
}

void reduce_array_mpi_aligned_vec(float * array, int length, int reduction_operator, int my_rank, int nproc, float * root_results, float * own_result) {
    *(own_result) = reduce_array_aligned(array, length, reduction_operator);
    MPI_Gather(
        own_result,
        1,
        MPI_FLOAT,
        root_results,
        1,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    if(my_rank == 0) {
        *(own_result) = 0.0;
        *(own_result) = reduce_arr_to_flt(root_results, nproc, reduction_operator);
    }
}

void reduce_array_mpi_unaligned_vec(float * array, int length, int reduction_operator, int my_rank, int nproc, float * root_results, float * own_result) {
    *(own_result) = reduce_array_unaligned(array, length, reduction_operator);
    MPI_Gather(
        own_result,
        1,
        MPI_FLOAT,
        root_results,
        1,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    if(my_rank == 0) {
        *(own_result) = 0.0;
        *(own_result) = reduce_arr_to_flt(root_results, nproc, reduction_operator);
    }
}

// reduce an aligned array of length (multiple of 8) float in 1 float
float reduce_array_aligned(float * array, int length, int reduction_operator) {
    int isVectorsEven = 0;
    
    //#pragma omp unroll
    for(; length != 8; length >>= 1){
        if(isVectorsEven)
            length += 4;

        isVectorsEven = 0;

        #pragma omp unroll
        for (int i=0, j=0; i<length; i+=16, j+=8){
            if(i+8 < length) {
                __m256 vect1 = _mm256_load_ps(array+i);
                __m256 vect2 = _mm256_load_ps(array+(i+8));

                __m256 result;

                switch(reduction_operator){
                    case 0: result = _mm256_add_ps(vect1, vect2); break;
                    case 1: result = _mm256_mul_ps(vect1, vect2); break;
                    case 2: result = _mm256_min_ps(vect1, vect2); break;
                    case 3: result = _mm256_max_ps(vect1, vect2); break;
                }
                
                _mm256_store_ps(array + j, result);
            } else {
                memmove(array + j, array+i, 8 * sizeof(float));
                isVectorsEven = 1;
            }
            
        }
    }

    if (reduction_operator == 0)
        return reduce_arr_to_flt_vect(array, 0);
    else
        return reduce_arr_to_flt(array, 8, reduction_operator);
}

// reduce an unaligned array of length (multiple of 8) float in 1 float
float reduce_array_unaligned(float * array, int length, int reduction_operator) {
    int isVectorsEven = 0;
    
    //#pragma omp unroll
    for(; length != 8; length >>= 1){
        if(isVectorsEven)
            length += 4;

        isVectorsEven = 0;

        #pragma omp unroll
        for (int i=0, j=0; i<length; i+=16, j+=8){
            if(i+8 < length) {
                __m256 vect1 = _mm256_loadu_ps(array+i);
                __m256 vect2 = _mm256_loadu_ps(array+(i+8));

                __m256 result;

                switch(reduction_operator){
                    case 0: result = _mm256_add_ps(vect1, vect2); break;
                    case 1: result = _mm256_mul_ps(vect1, vect2); break;
                    case 2: result = _mm256_min_ps(vect1, vect2); break;
                    case 3: result = _mm256_max_ps(vect1, vect2); break;
                }
                
                _mm256_storeu_ps(array + j, result);
            } else {
                memmove(array + j, array+i, 8 * sizeof(float));
                isVectorsEven = 1;
            }
            
        }
    }
    if (reduction_operator == 0)
        return reduce_arr_to_flt_vect(array, 1);
    else
        return reduce_arr_to_flt(array, 8, reduction_operator);
}

// reduce an array of 8 floats into 1 float using vectorization
float reduce_arr_to_flt_vect(float * array, int mode) {
    __m256 hadd, vlow_256, vect;
    __m128 vlow, vhigh;

    if (mode == 0)
        vect = _mm256_load_ps(array);
    else 
        vect = _mm256_loadu_ps(array);

    hadd = _mm256_hadd_ps(vect, vect);

    vlow  = _mm256_castps256_ps128(hadd);
    vhigh = _mm256_extractf128_ps(hadd, 1);
    
    vlow  = _mm_add_ps(vlow, vhigh);

    vlow_256 = _mm256_castps128_ps256(vlow);
    hadd = _mm256_hadd_ps(vlow_256, vlow_256);

    return _mm256_cvtss_f32(hadd);   
}

// reduce an array of 8 floats into 1 float sequentially
float reduce_arr_to_flt(float * array, int length, int reduction_operator){
    float result;

    switch(reduction_operator) {
        case 0: result = 0;
        break;
        case 1: result = 1;
        break;
        default: result = array[0];
        break;
    }

    for(int i = 0; i < length; i++){
        switch(reduction_operator) {
            case 0: result += array[i];
            break;
            case 1: result *= array[i];
            break;
            case 2: if(array[i] < result) result = array[i];
            break;
            case 3: if(array[i] > result) result = array[i];
            break;
        }
    }
    return result;
}



