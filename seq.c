#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#define NREP 50

//export OMPI_MCA_btl_vader_single_copy_mechanism=none
//bin/bash

float reduce_arr_to_flt(float * array, int length, int reduction_operator);

int main(int argc, char * argv[]){
    float result;

    int length = atoi(argv[1]), nrep = NREP;
    int reduction_operator = atoi(argv[2]);

    float * array = (float *) calloc(length, sizeof(float));

    //INIZIALIZZAZIONE
    for(int i = 0; i < length; i++){
        array[i] = i % 5 + 1;
    }

    double start, end;

    // sequential version
    start = omp_get_wtime();

    for(int i = 0; i < nrep; i++)
        result = reduce_arr_to_flt(array, length, reduction_operator);
    
    end = (omp_get_wtime() - start) / nrep;

    printf("Time seq: %f\n", end);

    free(array);
	return 0;
}

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