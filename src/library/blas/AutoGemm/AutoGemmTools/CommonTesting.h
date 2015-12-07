
#ifndef COMMONTESTING_H
#define COMMONTESTING_H

//#include <Windows.h>
#if defined( __APPLE__ ) || defined( __MACOSX )
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

extern
cl_platform_id
getPlatform(const char *name);

extern 
cl_device_id
getDevice(
    cl_platform_id platform,
    const char *name);

extern 
cl_device_id
getPlatformDevice();

extern 
cl_kernel
createKernel(
    const char* source,
    cl_context context,
    const char* options,
    cl_int* error);

template<typename T>
void
randomMatrix(
    clblasOrder order,
    size_t rows,
    size_t columns,
    T *A,
    size_t lda)
{
    size_t r, c;
    MatrixAccessor<T> a(A, order, clblasNoTrans, rows, columns, lda);

    for (r = 0; r < rows; r++) {
        for (c = 0; c < columns; c++) {
#if RANDOM_DATA
            a[r][c] = random<T>();
#else
            a[r][c] = DATA_TYPE_CONSTRUCTOR(1, 0);
#endif
        }
    }
}

template<typename T>
bool
compareMatrices(
    clblasOrder order,
    size_t rows,
    size_t columns,
    T *blasMatrix,
    T *naiveMatrix,
    size_t ld)
{
    size_t r, c;
    MatrixAccessor<T> blas(blasMatrix, order, clblasNoTrans, rows, columns, ld);
    MatrixAccessor<T> naive(naiveMatrix, order, clblasNoTrans, rows, columns, ld);
    T blasVal, naiveVal;
    int numPrint = 96 * 96;
    bool equal = true;
    for (r = 0; r < rows; r++) {
        for (c = 0; c < columns; c++) {
            blasVal = blas[r][c];
            naiveVal = naive[r][c];
            if (isNAN(blasVal) && isNAN(naiveVal)) {
                continue;
            }
            if (blasVal != naiveVal) {
                equal = false;
            }

            if (blasVal != naiveVal) {
                if (numPrint-- > 0) {
#if CGEMM || ZGEMM
                    printf("MISMATCH C[%u][%u]: gpu= %4.1f + %4.1fi, cpu= %4.1f + %4.1fi\n",
                        r, c,
                        blasVal.s[0], blasVal.s[1],
                        naiveVal.s[0], naiveVal.s[1]);
#else
                    printf("MISMATCH C[%u][%u]: gpu= %4.1f, cpu= %4.1f\n",
                        r, c,
                        blasVal,
                        naiveVal);
#endif
                }
                else {
                    return equal;
                }
            }
        }
    }
    return equal;
}

#endif