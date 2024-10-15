#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void MatrixMulKernel(float* matmul, const float* a, const float* b, int n, int m, int size) {
    // Если сетка двухмерная
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < m) {
        for (int k = 0; k < size; ++k) {
            matmul[i * m + j] += a[i*size + k] * b[k*m + j];
        }
    }
}

// функция для сравнения двух массивов
bool arraysEqual(const float* a, const float* b, int size) {
    for (int i = 0; i < size; ++i) {
        if (abs(a[i] - b[i]) > 0.000001) return false;
    }
    return true;
}

// функция для выводы массива в виде матрицы
void printMatrix(const float* a, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            printf("%f ", a[i*m + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// функция для инициализации матриц
void InitMatrix(float* a, int size, float k){
    for(int i = 0; i < size; ++i){
        a[i] = k;
    }
}


int main(int argc, char* argv[])
{
    printf("N1=%d, M1=%d, N2=%d, M2=%d\n", N1, M1, N2, M2);
    printf("(GridDim, BlockDim) value: ((%d, %d), (%d, %d))\n", GRID_SIZE_X, GRID_SIZE_Y, BLOCK_SIZE_X, BLOCK_SIZE_Y);

    if (M1 != N2){
        printf("M1 != N2");
        return 0;
    }

    clock_t startCPU, endCPU; 
    int A_float = N1 * M1 * sizeof(float);
    int B_float = N2 * M2 * sizeof(float);
    int MatMul_float = N1 * M2 * sizeof(float);
	
    // Выделение памяти на хосте
   	float* A = (float*)calloc(N1 * M1, sizeof(float));
	float* B = (float*)calloc(N2 * M2, sizeof(float));
	float* MatMul = (float*)calloc(N1 * M2, sizeof(float));
    float* CUPMatMul = (float*)calloc(N1 * M2, sizeof(float));

    // Инициализация массивов
    InitMatrix(A, N1*M1, 1);
    InitMatrix(B, N2*M2, 3);

    // Выделение памяти на устройстве
    float* Adev = NULL;
    cudaError_t cuerr = cudaMalloc((void**)&Adev, A_float);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for a: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    float* Bdev = NULL;
    cuerr = cudaMalloc((void**)&Bdev, B_float);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for b: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    float* MatMuldev = NULL;
    cuerr = cudaMalloc((void**)&MatMuldev, MatMul_float);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot allocate device array for c: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cuerr = cudaEventCreate(&start);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA start event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    cuerr = cudaEventCreate(&stop);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot create CUDA end event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

	// Копирование данных с хоста на девайс
    cuerr = cudaMemcpy(Adev, A, A_float, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy a array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    cuerr = cudaMemcpy(Bdev, B, B_float, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy b array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    cuerr = cudaMemcpy(MatMuldev, MatMul, MatMul_float, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy MatMul array from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    // Установка точки старта
    cuerr = cudaEventRecord(start, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    //Запуск ядра
    MatrixMulKernel<<< dim3(GRID_SIZE_X, GRID_SIZE_Y), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y) >>>(MatMuldev, Adev, Bdev, N1, M2, N2);

    cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

	// Синхронизация устройств
    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

	// Установка точки окончания
    cuerr = cudaEventRecord(stop, 0);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot record CUDA event: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

	// Копирование результата на хост
    cuerr = cudaMemcpy(MatMul, MatMuldev, MatMul_float, cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess)
    {
        fprintf(stderr, "Cannot copy sum array from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 0;
    }

    startCPU = clock();
    // Умножение матриц на CPU
    for (int i = 0; i < N1; ++i){
        for(int j = 0; j < M2; ++j){
            for(int k = 0; k < N2; ++k){
                CUPMatMul[i * M2 + j] += A[i * M1 + k] * B[k * M2 + j];
            }
        }
    }
    endCPU = clock();
    float time_taken = float(endCPU - startCPU) / double(CLOCKS_PER_SEC);

    // Проверка
    if (!arraysEqual(MatMul, CUPMatMul, N1*M2)){
        printf("MatMul != CUPMatMul\n");
    }
    else printf("MatMul == CUPMatMul\n");

    // Расчет времени
    cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
    printf("CPU: %.8f seconds\n\n", time_taken);
    printf("GPU: %.8f seconds\n\n", gpuTime/1000);

    // Очищение памяти
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(Adev); 
    cudaFree(Bdev);
    cudaFree(MatMuldev);
    free(A);
    free(B);
    free(MatMul);
    free(CUPMatMul);

    return 0;
}
