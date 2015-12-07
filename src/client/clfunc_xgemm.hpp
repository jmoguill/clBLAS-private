/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


// $Id

#ifndef CLBLAS_BENCHMARK_XGEMM_HXX__
#define CLBLAS_BENCHMARK_XGEMM_HXX__

#include "clfunc_common.hpp"
#include <acml.h>
#include "common.h"

char
encodeTranspose(clblasTranspose value)
{
    switch (value) {
    case clblasNoTrans:      return 'N';
    case clblasTrans:        return 'T';
    case clblasConjTrans:    return 'C';
    }
    return '\0';
}

template <typename T>
struct xGemmBuffer
{
    clblasOrder order_;
    size_t m_;
    size_t n_;
    size_t k_;
    size_t lda_;
    size_t ldb_;
    size_t ldc_;
    size_t offA_;
    size_t offB_;
    size_t offC_;
    size_t a_num_vector;
    size_t b_num_vector;
    size_t c_num_vector;
    clblasTranspose transA;
    clblasTranspose transB;
    T* hA;
    T* hB;
    T* hC;
    T* hR;
    cl_mem dA;
    cl_mem dB;
    cl_mem dC;
    T alpha_;
    T beta_;
	cl_uint apiCallCount;
}; // struct buffer

template <typename T>
class xGemm : public clblasFunc
{
public:
    xGemm(StatisticalTimer& timer, cl_device_type devType, int devID, unsigned int iNumQueuesToUse) :
        clblasFunc(timer, devType, devID),
        numQueuesToUse(iNumQueuesToUse)
    {
        timer.getUniqueID("clGemm", 0);
    }

    ~xGemm()
    {
    }

    void call_func()
    {
		timer.Start(timer_id);
		xGemm_Function(true, buffer.apiCallCount);
		timer.Stop(timer_id);
        //read_gpu_buffer();
        //verfication();
    }

    double gflops()
    {
		return (2.0*buffer.m_*buffer.n_*buffer.k_) / (time_in_ns() / buffer.apiCallCount);
    }

	void setup_apiCallCount(cl_uint apiCallCount)
	{
		buffer.apiCallCount = apiCallCount;
	}
    std::string gflops_formula()
    {
        return "2.0*M*N*K/time";
    }

    void setup_buffer(int order_option, int side_option, int uplo_option,
                      int diag_option, int transA_option, int  transB_option,
                      size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                      size_t ldc, size_t offA, size_t offBX, size_t offCY,
                      double alpha, double beta)
    {
        DUMMY_ARGS_USAGE_3(side_option, uplo_option, diag_option);

        initialize_scalars(alpha, beta);

        buffer.m_ = M;
        buffer.n_ = N;
        buffer.k_ = K;
        buffer.offA_ = offA;
        buffer.offB_ = offBX;
        buffer.offC_ = offCY;

        if (order_option == 0)
        {
            order_ = clblasRowMajor;
            if (transA_option == 0)
            {
                buffer.transA = clblasNoTrans;
                buffer.a_num_vector = M;
                if (lda == 0)
                {
                    buffer.lda_ = K;
                }
                else if (lda < K)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.lda_ = lda;
                }
            }
            else
            {
                buffer.a_num_vector = K;
                if (transA_option == 1)
                {
                    buffer.transA = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer.transA = clblasConjTrans;
                }
                if (lda == 0)
                {
                    buffer.lda_ = M;
                }
                else if (lda < M)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.lda_ = lda;
                }
            }

            if (transB_option == 0)
            {
                buffer.b_num_vector = K;
                buffer.transB = clblasNoTrans;
                if (ldb == 0)
                {
                    buffer.ldb_ = N;
                }
                else if (ldb < N)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.ldb_ = ldb;
                }
            }
            else
            {
                buffer.b_num_vector = N;
                if (transB_option == 1)
                {
                    buffer.transB = clblasTrans;
                }
                else if (transB_option == 2)
                {
                    buffer.transB = clblasConjTrans;
                }

                if (ldb == 0)
                {
                    buffer.ldb_ = K;
                }
                else if (ldb < K)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.ldb_ = ldb;
                }
            }

            if (ldc == 0)
            {
                buffer.ldc_ = N;
            }
            else if (ldc < N)
            {
                std::cerr << "ldc:wrong size\n";
            }
            else
            {
                buffer.ldc_ = ldc;
            }
            buffer.c_num_vector = M;
        }
        else
        {
            order_ = clblasColumnMajor;
            if (transA_option == 0)
            {
                buffer.a_num_vector = K;
                buffer.transA = clblasNoTrans;
                if (lda == 0)
                {
                    buffer.lda_ = M;
                }
                else if (lda < M)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.lda_ = lda;
                }
            }
            else
            {
                buffer.a_num_vector = M;
                if (transA_option == 1)
                {
                    buffer.transA = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer.transA = clblasConjTrans;
                }


                if (lda == 0)
                {
                    buffer.lda_ = K;
                }
                else if (lda < K)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.lda_ = lda;
                }
            }

            if (transB_option == 0)
            {
                buffer.b_num_vector = N;
                buffer.transB = clblasNoTrans;

                if (ldb == 0)
                {
                    buffer.ldb_ = K;
                }
                else if (ldb < K)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.ldb_ = ldb;
                }
            }
            else
            {
                buffer.b_num_vector = K;
                if (transB_option == 1)
                {
                    buffer.transB = clblasTrans;
                }
                else if (transB_option == 2)
                {
                    buffer.transB = clblasConjTrans;
                }

                if (ldb == 0)
                {
                    buffer.ldb_ = N;
                }
                else if (ldb < N)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.ldb_ = ldb;
                }
            }

            if (ldc == 0)
            {
                buffer.ldc_ = M;
            }
            else if (ldc < M)
            {
                std::cerr << "ldc:wrong size\n";
            }
            else
            {
                buffer.ldc_ = ldc;
            }
            buffer.c_num_vector = N;
        }
        buffer.hA = new T[buffer.lda_*buffer.a_num_vector];
        buffer.hB = new T[buffer.ldb_*buffer.b_num_vector];
        buffer.hC = new T[buffer.ldc_*buffer.c_num_vector];
        buffer.hR = new T[buffer.ldc_*buffer.c_num_vector];


        cl_int err;
        buffer.dA = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                       (buffer.lda_*buffer.a_num_vector +
                                           buffer.offA_) * sizeof(T),
                                       NULL, &err);

        buffer.dB = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer.ldb_ * buffer.b_num_vector +
                                            buffer.offB_) * sizeof(T),
                                        NULL, &err);

        buffer.dC = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer.ldc_ * buffer.c_num_vector +
                                            buffer.offC_) * sizeof(T),
                                        NULL, &err);

    }

    void initialize_cpu_buffer()
    {
        srand(10);
        for (size_t i = 0; i < buffer.a_num_vector; ++i)
        {
            for (size_t j = 0; j < buffer.lda_; ++j)
            {
                buffer.hA[i*buffer.lda_+j] = random<T>(UPPER_BOUND<T>()) /
                                               randomScale<T>();
            }
        }

        for (size_t i = 0; i < buffer.b_num_vector; ++i)
        {
            for (size_t j = 0; j < buffer.ldb_; ++j)
            {
                buffer.hB[i*buffer.ldb_+j] = random<T>(UPPER_BOUND<T>()) /
                                               randomScale<T>();
            }
        }

        for (size_t i = 0; i < buffer.c_num_vector; ++i)
        {
            for (size_t j = 0; j < buffer.ldc_; ++j)
            {
                buffer.hR[i*buffer.ldc_ +j] = buffer.hC[i*buffer.ldc_+j] = random<T>(UPPER_BOUND<T>()) /
                                               randomScale<T>();

            }
        }
    }

    void initialize_gpu_buffer()
    {

		cl_int err;

        err = clEnqueueWriteBuffer(queues_[0], buffer.dA, CL_TRUE,
                                   buffer.offA_ * sizeof(T),
                                   buffer.lda_ * buffer.a_num_vector *
                                       sizeof(T),
                                   buffer.hA, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(queues_[0], buffer.dB, CL_TRUE,
                                   buffer.offB_ * sizeof(T),
                                   buffer.ldb_ * buffer.b_num_vector *
                                       sizeof(T),
                                   buffer.hB, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(queues_[0], buffer.dC, CL_TRUE,
                                   buffer.offC_ * sizeof(T),
                                   buffer.ldc_ * buffer.c_num_vector *
                                   sizeof(T),
                                   buffer.hC, 0, NULL, NULL);


    }

    void reset_gpu_write_buffer()
    {
        cl_int err;
        err = clEnqueueWriteBuffer(queues_[0], buffer.dC, CL_TRUE,
                                   buffer.offC_ * sizeof(T),
                                   buffer.ldc_ * buffer.c_num_vector *
                                       sizeof(T),
                                   buffer.hC, 0, NULL, NULL);
    }

	void read_gpu_buffer()
	{
		cl_int err;
		err = clEnqueueReadBuffer(queues_[0], buffer.dC, CL_TRUE,
			                      buffer.offC_ * sizeof(T), buffer.ldc_ * buffer.c_num_vector *
                                       sizeof(T),
								  buffer.hC, 0, NULL, NULL);
	}

	void roundtrip_func()
	{
	    timer.Start(timer_id);
		cl_int err;
        buffer.dA = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                       (buffer.lda_*buffer.a_num_vector +
                                           buffer.offA_) * sizeof(T),
                                       NULL, &err);

        buffer.dB = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer.ldb_ * buffer.b_num_vector +
                                            buffer.offB_) * sizeof(T),
                                        NULL, &err);

        buffer.dC = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer.ldc_ * buffer.c_num_vector +
                                            buffer.offC_) * sizeof(T),
                                        NULL, &err);
        err = clEnqueueWriteBuffer(queues_[0], buffer.dA, CL_TRUE,
                                   buffer.offA_ * sizeof(T),
                                   buffer.lda_ * buffer.a_num_vector *
                                       sizeof(T),
                                   buffer.hA, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(queues_[0], buffer.dB, CL_TRUE,
                                   buffer.offB_ * sizeof(T),
                                   buffer.ldb_ * buffer.b_num_vector *
                                       sizeof(T),
                                   buffer.hB, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(queues_[0], buffer.dC, CL_TRUE,
                                   buffer.offC_ * sizeof(T),
                                   buffer.ldc_ * buffer.c_num_vector *
                                   sizeof(T),
                                   buffer.hC, 0, NULL, NULL);
		xGemm_Function(false);
		err = clEnqueueReadBuffer(queues_[0], buffer.dC, CL_TRUE,
			                      buffer.offC_ * sizeof(T), buffer.ldc_ * buffer.c_num_vector *
                                       sizeof(T),
								  buffer.hC, 0, NULL, &event_);
		clWaitForEvents(1, &event_);
	    timer.Stop(timer_id);
	}
	void roundtrip_func_rect()
	{
	    timer.Start(timer_id);
		cl_int err;
		//rect
		size_t a_buffer_origin[3] = {0,0,0}; 
		size_t a_host_origin[3] = {0,0,0};
		size_t a_region[3] = {buffer.m_*sizeof(T),buffer.k_,1};
		size_t a_buffer_row_pitch=0*sizeof(T);//lda
		size_t a_buffer_slice_pitch=0;
		size_t a_host_row_pitch=buffer.lda_*sizeof(T);
		size_t a_host_slice_pitch=0;

		size_t b_buffer_origin[3] = {0,0,0}; 
		size_t b_host_origin[3] = {0,0,0};
		size_t b_region[3] = {buffer.k_*sizeof(T),buffer.n_,1};
		size_t b_buffer_row_pitch=0*sizeof(T);//ldb
		size_t b_buffer_slice_pitch=0;
		size_t b_host_row_pitch=buffer.ldb_*sizeof(T);
		size_t b_host_slice_pitch=0;

		size_t c_buffer_origin[3] = {0,0,0}; 
		size_t c_host_origin[3] = {0,0,0};
		size_t c_region[3] = {buffer.m_*sizeof(T),buffer.n_,1};
		size_t c_buffer_row_pitch=0*sizeof(T);//ldc
		size_t c_buffer_slice_pitch=0;
		size_t c_host_row_pitch=buffer.ldc_*sizeof(T);
		size_t c_host_slice_pitch=0;

        buffer.dA = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                       (buffer.k_*buffer.m_ +
                                           buffer.offA_) * sizeof(T),
                                       NULL, &err);

        buffer.dB = clCreateBuffer(ctx_, CL_MEM_READ_ONLY,
                                        (buffer.k_ * buffer.n_ +
                                            buffer.offB_) * sizeof(T),
                                        NULL, &err);

        buffer.dC = clCreateBuffer(ctx_, CL_MEM_READ_WRITE,
                                        (buffer.m_ * buffer.n_ +
                                            buffer.offC_) * sizeof(T),
                                        NULL, &err);
        /*
		err = clEnqueueWriteBuffer(queues_[0], buffer.dA, CL_TRUE,
                                   buffer.offA_ * sizeof(T),
                                   buffer.lda_ * buffer.a_num_vector *
                                       sizeof(T),
                                   buffer.hA, 0, NULL, NULL);
		
        err = clEnqueueWriteBuffer(queues_[0], buffer.dB, CL_TRUE,
                                   buffer.offB_ * sizeof(T),
                                   buffer.ldb_ * buffer.b_num_vector *
                                       sizeof(T),
                                   buffer.hB, 0, NULL, NULL);

        err = clEnqueueWriteBuffer(queues_[0], buffer.dC, CL_TRUE,
                                   buffer.offC_ * sizeof(T),
                                   buffer.ldc_ * buffer.c_num_vector *
                                   sizeof(T),
                                   buffer.hC, 0, NULL, NULL);*/
        err = clEnqueueWriteBufferRect(queues_[0], buffer.dA, CL_TRUE, a_buffer_origin, a_host_origin, a_region, a_buffer_row_pitch,
										a_buffer_slice_pitch, a_host_row_pitch, a_host_slice_pitch, buffer.hA, 0, NULL, NULL);
        err = clEnqueueWriteBufferRect(queues_[0], buffer.dB, CL_TRUE, b_buffer_origin, b_host_origin, b_region, b_buffer_row_pitch,
										b_buffer_slice_pitch, b_host_row_pitch, b_host_slice_pitch, buffer.hB, 0, NULL, NULL);
        err = clEnqueueWriteBufferRect(queues_[0], buffer.dC, CL_TRUE, c_buffer_origin, c_host_origin, c_region, c_buffer_row_pitch,
										c_buffer_slice_pitch, c_host_row_pitch, c_host_slice_pitch, buffer.hC, 0, NULL, NULL);

		if(buffer.transA==clblasNoTrans)
		{
			buffer.lda_=buffer.m_;
		}
		else
		{
			buffer.lda_=buffer.k_;
		}
		if(buffer.transB==clblasNoTrans)
		{
			buffer.ldb_=buffer.k_;
		}
		else
		{
			buffer.ldb_=buffer.n_;
		}
		buffer.ldc_=buffer.m_;
		xGemm_Function(false);
		/*
		err = clEnqueueReadBuffer(queues_[0], buffer.dC, CL_TRUE,
			                      buffer.offC_ * sizeof(T), buffer.ldc_ * buffer.c_num_vector *
                                       sizeof(T),
								  buffer.hC, 0, NULL, &event_);
		*/
		err = ::clEnqueueReadBufferRect(queues_[0], buffer.dC, CL_TRUE, c_buffer_origin, c_host_origin, c_region, c_buffer_row_pitch,
										c_buffer_slice_pitch, c_host_row_pitch, c_host_slice_pitch, buffer.hC, 0, NULL, &event_);
		clWaitForEvents(1, &event_);
	timer.Stop(timer_id);
	}	
	void allochostptr_roundtrip_func()
	{
	timer.Start(timer_id);

		cl_int err;
		// Create buffers with CL_MEM_ALLOC_HOST_PTR for zero copy
        buffer.dA = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                       (buffer.lda_*buffer.a_num_vector +
                                           buffer.offA_) * sizeof(T),
                                       NULL, &err);

        buffer.dB = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                        (buffer.ldb_ * buffer.b_num_vector +
                                            buffer.offB_) * sizeof(T),
                                        NULL, &err);

        buffer.dC = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                        (buffer.ldc_ * buffer.c_num_vector +
                                            buffer.offC_) * sizeof(T),
                                        NULL, &err);

		// map the buffers to pointers at host device
		T *map_a,*map_b,*map_c;
		map_a = (T*)clEnqueueMapBuffer(queues_[0], buffer.dA, CL_TRUE, CL_MAP_WRITE, 0, 
										  (buffer.lda_*buffer.a_num_vector +
                                           buffer.offA_) * sizeof(T),
										   0, NULL, NULL, &err);
		map_b = (T*)clEnqueueMapBuffer(queues_[0], buffer.dB, CL_TRUE, CL_MAP_WRITE, 0, 
										  (buffer.ldb_*buffer.b_num_vector +
                                           buffer.offB_) * sizeof(T),
										   0, NULL, NULL, &err);
	    map_c = (T*)clEnqueueMapBuffer(queues_[0], buffer.dC, CL_TRUE, CL_MAP_WRITE, 0, 
										  (buffer.lda_*buffer.c_num_vector +
                                           buffer.offC_) * sizeof(T),
										   0, NULL, NULL, &err);
		// memcpy the input A, B, C to the host pointers
		memcpy( map_a, buffer.hA, ( buffer.lda_*buffer.a_num_vector + buffer.offA_) * sizeof( T ) );
		memcpy( map_b, buffer.hB, ( buffer.ldb_*buffer.b_num_vector + buffer.offB_) * sizeof( T ) );
		memcpy( map_c, buffer.hC, ( buffer.ldc_*buffer.c_num_vector + buffer.offC_) * sizeof( T ) );
		// unmap the buffers
		clEnqueueUnmapMemObject(queues_[0], buffer.dA, map_a, 0, NULL, NULL);
		clEnqueueUnmapMemObject(queues_[0], buffer.dB, map_b, 0, NULL, NULL);
		clEnqueueUnmapMemObject(queues_[0], buffer.dC, map_c, 0, NULL, NULL);
		// calling clBLAS
		xGemm_Function(false);
		// map the C buffer again to read output
	    map_c = (T*)clEnqueueMapBuffer(queues_[0], buffer.dC, CL_TRUE, CL_MAP_READ, 0, 
										  (buffer.lda_*buffer.c_num_vector +
                                           buffer.offC_) * sizeof(T),
										   0, NULL, NULL, &err);
		memcpy( map_c, buffer.hC, ( buffer.ldc_*buffer.c_num_vector + buffer.offC_) * sizeof( T ) );
		clEnqueueUnmapMemObject(queues_[0], buffer.dC, map_c, 0, NULL, &event_);
		clWaitForEvents(1, &event_);

	timer.Stop(timer_id);
	}
	void usehostptr_roundtrip_func()
	{
	    timer.Start(timer_id);
		cl_int err;
        buffer.dA = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                       (buffer.lda_*buffer.a_num_vector +
                                           buffer.offA_) * sizeof(T),
                                       buffer.hA, &err);

        buffer.dB = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        (buffer.ldb_ * buffer.b_num_vector +
                                            buffer.offB_) * sizeof(T),
                                        buffer.hB, &err);

        buffer.dC = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                        (buffer.ldc_ * buffer.c_num_vector +
                                            buffer.offC_) * sizeof(T),
                                        buffer.hC, &err);
		xGemm_Function(true);
	    timer.Stop(timer_id);
	}
	void copyhostptr_roundtrip_func()
	{
	timer.Start(timer_id);
		cl_int err;
        buffer.dA = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       (buffer.lda_*buffer.a_num_vector +
                                           buffer.offA_) * sizeof(T),
                                       buffer.hA, &err);

        buffer.dB = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        (buffer.ldb_ * buffer.b_num_vector +
                                            buffer.offB_) * sizeof(T),
                                        buffer.hB, &err);

        buffer.dC = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        (buffer.ldc_ * buffer.c_num_vector +
                                            buffer.offC_) * sizeof(T),
                                        buffer.hC, &err);
		xGemm_Function(false);
		err = clEnqueueReadBuffer(queues_[0], buffer.dC, CL_TRUE,
			                      buffer.offC_ * sizeof(T), buffer.ldc_ * buffer.c_num_vector *
                                       sizeof(T),
								  buffer.hC, 0, NULL, &event_);
		clWaitForEvents(1, &event_);
	timer.Stop(timer_id);
	}
	void usepersismem_roundtrip_func()
	{
#if defined(CL_MEM_USE_PERSISTENT_MEM_AMD)
	timer.Start(timer_id);

		cl_int err;

        buffer.dA = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD,
                                       (buffer.lda_*buffer.a_num_vector +
                                           buffer.offA_) * sizeof(T),
                                       NULL, &err);

        buffer.dB = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_USE_PERSISTENT_MEM_AMD,
                                        (buffer.ldb_ * buffer.b_num_vector +
                                            buffer.offB_) * sizeof(T),
                                        NULL, &err);

        buffer.dC = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_USE_PERSISTENT_MEM_AMD,
                                        (buffer.ldc_ * buffer.c_num_vector +
                                            buffer.offC_) * sizeof(T),
                                        NULL, &err);

		// map the buffers to pointers at host devices
		T *map_a,*map_b,*map_c;
		map_a = (T*)clEnqueueMapBuffer(queues_[0], buffer.dA, CL_TRUE, CL_MAP_WRITE, 0, 
										  (buffer.lda_*buffer.a_num_vector +
                                           buffer.offA_) * sizeof(T),
										   0, NULL, NULL, &err);
		map_b = (T*)clEnqueueMapBuffer(queues_[0], buffer.dB, CL_TRUE, CL_MAP_WRITE, 0, 
										  (buffer.ldb_*buffer.b_num_vector +
                                           buffer.offB_) * sizeof(T),
										   0, NULL, NULL, &err);
	    map_c = (T*)clEnqueueMapBuffer(queues_[0], buffer.dC, CL_TRUE, CL_MAP_WRITE, 0, 
										  (buffer.lda_*buffer.c_num_vector +
                                           buffer.offC_) * sizeof(T),
										   0, NULL, NULL, &err);
		// memcpy the input A, B, C to the host pointers
		memcpy( map_a, buffer.hA, ( buffer.lda_*buffer.a_num_vector + buffer.offA_) * sizeof( T ) );
		memcpy( map_b, buffer.hB, ( buffer.ldb_*buffer.b_num_vector + buffer.offB_) * sizeof( T ) );
		memcpy( map_c, buffer.hC, ( buffer.ldc_*buffer.c_num_vector + buffer.offC_) * sizeof( T ) );
		// unmap the buffers
		clEnqueueUnmapMemObject(queues_[0], buffer.dA, map_a, 0, NULL, NULL);
		clEnqueueUnmapMemObject(queues_[0], buffer.dB, map_b, 0, NULL, NULL);
		clEnqueueUnmapMemObject(queues_[0], buffer.dC, map_c, 0, NULL, NULL);
		// calling clBLAS
		xGemm_Function(false);
		// map the C buffer again to read output
	    map_c = (T*)clEnqueueMapBuffer(queues_[0], buffer.dC, CL_TRUE, CL_MAP_READ, 0, 
										  (buffer.lda_*buffer.c_num_vector +
                                           buffer.offC_) * sizeof(T),
										   0, NULL, NULL, &err);
		memcpy( map_c, buffer.hC, ( buffer.ldc_*buffer.c_num_vector + buffer.offC_) * sizeof( T ) );
		clEnqueueUnmapMemObject(queues_[0], buffer.dC, map_c, 0, NULL, &event_);
		clWaitForEvents(1, &event_);

	timer.Stop(timer_id);
#else
		std::cout<<"CL_MEM_USE_PERSISTENT_MEM_AMD is only supported on AMD hardware"<<std::endl;
#endif

	}
	void roundtrip_setup_buffer(int order_option, int side_option, int uplo_option,
                      int diag_option, int transA_option, int  transB_option,
                      size_t M, size_t N, size_t K, size_t lda, size_t ldb,
                      size_t ldc, size_t offA, size_t offBX, size_t offCY,
                      double alpha, double beta)
    {
        DUMMY_ARGS_USAGE_3(side_option, uplo_option, diag_option);

        initialize_scalars(alpha, beta);

        buffer.m_ = M;
        buffer.n_ = N;
        buffer.k_ = K;
        buffer.offA_ = offA;
        buffer.offB_ = offBX;
        buffer.offC_ = offCY;

        if (order_option == 0)
        {
            order_ = clblasRowMajor;
            if (transA_option == 0)
            {
                buffer.transA = clblasNoTrans;
                buffer.a_num_vector = M;
                if (lda == 0)
                {
                    buffer.lda_ = K;
                }
                else if (lda < K)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.lda_ = lda;
                }
            }
            else
            {
                buffer.a_num_vector = K;
                if (transA_option == 1)
                {
                    buffer.transA = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer.transA = clblasConjTrans;
                }
                if (lda == 0)
                {
                    buffer.lda_ = M;
                }
                else if (lda < M)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.lda_ = lda;
                }
            }

            if (transB_option == 0)
            {
                buffer.b_num_vector = K;
                buffer.transB = clblasNoTrans;
                if (ldb == 0)
                {
                    buffer.ldb_ = N;
                }
                else if (ldb < N)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.ldb_ = ldb;
                }
            }
            else
            {
                buffer.b_num_vector = N;
                if (transB_option == 1)
                {
                    buffer.transB = clblasTrans;
                }
                else if (transB_option == 2)
                {
                    buffer.transB = clblasConjTrans;
                }

                if (ldb == 0)
                {
                    buffer.ldb_ = K;
                }
                else if (ldb < K)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.ldb_ = ldb;
                }
            }

            if (ldc == 0)
            {
                buffer.ldc_ = N;
            }
            else if (ldc < N)
            {
                std::cerr << "ldc:wrong size\n";
            }
            else
            {
                buffer.ldc_ = ldc;
            }
            buffer.c_num_vector = M;
        }
        else
        {
            order_ = clblasColumnMajor;
            if (transA_option == 0)
            {
                buffer.a_num_vector = K;
                buffer.transA = clblasNoTrans;
                if (lda == 0)
                {
                    buffer.lda_ = M;
                }
                else if (lda < M)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.lda_ = lda;
                }
            }
            else
            {
                buffer.a_num_vector = M;
                if (transA_option == 1)
                {
                    buffer.transA = clblasTrans;
                }
                else if (transA_option == 2)
                {
                    buffer.transA = clblasConjTrans;
                }


                if (lda == 0)
                {
                    buffer.lda_ = K;
                }
                else if (lda < K)
                {
                    std::cerr << "lda:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.lda_ = lda;
                }
            }

            if (transB_option == 0)
            {
                buffer.b_num_vector = N;
                buffer.transB = clblasNoTrans;

                if (ldb == 0)
                {
                    buffer.ldb_ = K;
                }
                else if (ldb < K)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.ldb_ = ldb;
                }
            }
            else
            {
                buffer.b_num_vector = K;
                if (transB_option == 1)
                {
                    buffer.transB = clblasTrans;
                }
                else if (transB_option == 2)
                {
                    buffer.transB = clblasConjTrans;
                }

                if (ldb == 0)
                {
                    buffer.ldb_ = N;
                }
                else if (ldb < N)
                {
                    std::cerr << "ldb:wrong size\n";
                    exit(1);
                }
                else
                {
                    buffer.ldb_ = ldb;
                }
            }

            if (ldc == 0)
            {
                buffer.ldc_ = M;
            }
            else if (ldc < M)
            {
                std::cerr << "ldc:wrong size\n";
            }
            else
            {
                buffer.ldc_ = ldc;
            }
            buffer.c_num_vector = N;
        }
        buffer.hA = new T[buffer.lda_*buffer.a_num_vector];
        buffer.hB = new T[buffer.ldb_*buffer.b_num_vector];
        buffer.hC = new T[buffer.ldc_*buffer.c_num_vector ];

    }
	void releaseGPUBuffer_deleteCPUBuffer()
	{
		//this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
		//need to do this before we eventually hit the destructor
		delete buffer.hA;
        delete buffer.hB;
        delete buffer.hC;
        delete buffer.hR;
        OPENCL_V_THROW( clReleaseMemObject(buffer.dA),
                        "releasing buffer A");
        OPENCL_V_THROW( clReleaseMemObject(buffer.dB),
                        "releasing buffer B");
        OPENCL_V_THROW( clReleaseMemObject(buffer.dC),
                        "releasing buffer C");

	}

   

protected:
    void initialize_scalars(double alpha, double beta)
    {
        buffer.alpha_ = makeScalar<T>(alpha);
        buffer.beta_ = makeScalar<T>(beta);
    }

private:
    xGemmBuffer<T> buffer;
	void xGemm_Function(bool flush, cl_uint apiCallCount = 1);
    unsigned int numQueuesToUse;
    cl_event events_[numQueues];
    void verfication();

}; // class xgemm

template<>
void 
xGemm<cl_float>::
xGemm_Function(bool flush, cl_uint apiCallCount )
{
    for (unsigned int i = 0; i < numQueues; i++) 
    {
        events_[i] = NULL;
    }
	for (unsigned int i = 0; i < apiCallCount; i++)
	{
		clblasSgemm(order_, buffer.transA, buffer.transB,
			buffer.m_, buffer.n_, buffer.k_, buffer.alpha_,
			buffer.dA, buffer.offA_, buffer.lda_,
			buffer.dB, buffer.offB_, buffer.ldb_,
			buffer.beta_, buffer.dC, buffer.offC_,
			buffer.ldc_, numQueuesToUse, queues_, 0, NULL, events_);
	}
	//flush==true if only the kernel time (library call) is timed
	//flush==false if memory time is also timed
	if (flush==true)
	{
      // check if any valid events returned
      cl_uint numValidEvents = 0;
      for (unsigned int i = 0; i < numQueuesToUse; i++) 
      {
        if (events_[i]) 
        {
            cl_uint clReferenceCount;
            cl_int err = clGetEventInfo(events_[i], CL_EVENT_REFERENCE_COUNT, sizeof(clReferenceCount), &clReferenceCount, NULL);
            if ( err == CL_SUCCESS) {
            //printf("events[%u/%u] has %u references\n", i, numQueuesToUse, clReferenceCount );
                numValidEvents++;
            } 
            else {
          //printf("events[%u/%u] invalid; err = %i\n", i, numQueuesToUse, err );
            }
        }
        else {
        //printf("events[%u/%u] is NULL\n", i, numQueuesToUse );
        }
      }
    
      for (unsigned int i = 0; i < numQueuesToUse; i++) {
        clFlush(queues_[i]);
      }
	  clWaitForEvents(numValidEvents, events_);
	}
}

template<>
void 
xGemm<cl_double>::
xGemm_Function(bool flush, cl_uint apiCallCount )
{
  for (unsigned int i = 0; i < numQueues; i++) {
    events_[i] = NULL;
  }
  for (unsigned int i = 0; i < apiCallCount; i++)
  {
	  clblasDgemm(order_, buffer.transA, buffer.transB,
                     buffer.m_, buffer.n_, buffer.k_, buffer.alpha_,
                     buffer.dA, buffer.offA_, buffer.lda_,
                     buffer.dB, buffer.offB_, buffer.ldb_,
                     buffer.beta_, buffer.dC, buffer.offC_,
                     buffer.ldc_, numQueuesToUse, queues_, 0, NULL, events_);
  }
	//flush==true if only the kernel time (library call) is timed
	//flush==false if memory time is also timed
	if (flush==true)
	{
    // check if any valid events returned
    cl_uint numValidEvents = 0;
    for (unsigned int i = 0; i < numQueuesToUse; i++) {
      if (events_[i]) {
        cl_uint clReferenceCount;
        cl_int err = clGetEventInfo(events_[i], CL_EVENT_REFERENCE_COUNT, sizeof(clReferenceCount), &clReferenceCount, NULL);
        if ( err == CL_SUCCESS) {
          //printf("events[%u/%u] has %u references\n", i, numQueuesToUse, clReferenceCount );
          numValidEvents++;
        } else {
          //printf("events[%u/%u] invalid; err = %i\n", i, numQueuesToUse, err );
        }
      } else {
        //printf("events[%u/%u] is NULL\n", i, numQueuesToUse );
      }
    }
    
    for (unsigned int i = 0; i < numQueuesToUse; i++) {
      clFlush(queues_[i]);
    }
		clWaitForEvents(numValidEvents, events_);
	}
}

template<>
void 
xGemm<cl_float2>::
xGemm_Function(bool flush, cl_uint apiCallCount )
{
  for (unsigned int i = 0; i < numQueues; i++) {
    events_[i] = NULL;
  }
  for (unsigned int i = 0; i < apiCallCount; i++)
	{
	  clblasCgemm(order_, buffer.transA, buffer.transB,
                     buffer.m_, buffer.n_, buffer.k_, buffer.alpha_,
                     buffer.dA, buffer.offA_, buffer.lda_,
                     buffer.dB, buffer.offB_, buffer.ldb_,
                     buffer.beta_, buffer.dC, buffer.offC_,
                     buffer.ldc_, numQueuesToUse, queues_, 0, NULL, events_);
  }
	//flush==true if only the kernel time (library call) is timed
	//flush==false if memory time is also timed
	if (flush==true)
	{
    // check if any valid events returned
    cl_uint numValidEvents = 0;
    for (unsigned int i = 0; i < numQueuesToUse; i++) {
      if (events_[i]) {
        cl_uint clReferenceCount;
        cl_int err = clGetEventInfo(events_[i], CL_EVENT_REFERENCE_COUNT, sizeof(clReferenceCount), &clReferenceCount, NULL);
        if ( err == CL_SUCCESS) {
          //printf("events[%u/%u] has %u references\n", i, numQueuesToUse, clReferenceCount );
          numValidEvents++;
        } else {
          //printf("events[%u/%u] invalid; err = %i\n", i, numQueuesToUse, err );
        }
      } else {
        //printf("events[%u/%u] is NULL\n", i, numQueuesToUse );
      }
    }
    
    for (unsigned int i = 0; i < numQueuesToUse; i++) {
      clFlush(queues_[i]);
    }
		clWaitForEvents(numValidEvents, events_);
	}
}

template<>
void 
xGemm<cl_double2>::
xGemm_Function(bool flush, cl_uint apiCallCount )
{
  for (unsigned int i = 0; i < numQueues; i++) {
    events_[i] = NULL;
  }
  for (unsigned int i = 0; i < apiCallCount; i++)
	{
	  clblasZgemm(order_, buffer.transA, buffer.transB,
                     buffer.m_, buffer.n_, buffer.k_, buffer.alpha_,
                     buffer.dA, buffer.offA_, buffer.lda_,
                     buffer.dB, buffer.offB_, buffer.ldb_,
                     buffer.beta_, buffer.dC, buffer.offC_,
                     buffer.ldc_, numQueuesToUse, queues_, 0, NULL, events_);
  }
	//flush==true if only the kernel time (library call) is timed
	//flush==false if memory time is also timed
	if (flush==true)
	{
    // check if any valid events returned
    cl_uint numValidEvents = 0;
    for (unsigned int i = 0; i < numQueuesToUse; i++) {
      if (events_[i]) {
        cl_uint clReferenceCount;
        cl_int err = clGetEventInfo(events_[i], CL_EVENT_REFERENCE_COUNT, sizeof(clReferenceCount), &clReferenceCount, NULL);
        if ( err == CL_SUCCESS) {
          //printf("events[%u/%u] has %u references\n", i, numQueuesToUse, clReferenceCount );
          numValidEvents++;
        } else {
          //printf("events[%u/%u] invalid; err = %i\n", i, numQueuesToUse, err );
        }
      } else {
        //printf("events[%u/%u] is NULL\n", i, numQueuesToUse );
      }
    }
    for (unsigned int i = 0; i < numQueuesToUse; i++) {
      clFlush(queues_[i]);
    }

		clWaitForEvents(numValidEvents, events_);
	}
}

template<>
double
xGemm<cl_float2>::
gflops()
{
    return (8.0*buffer.m_*buffer.n_*buffer.k_)/(time_in_ns() / buffer.apiCallCount);
}

template<>
double
xGemm<cl_double2>::
gflops()
{
    return (8.0*buffer.m_*buffer.n_*buffer.k_)/(time_in_ns() / buffer.apiCallCount);
}

template<>
std::string
xGemm<cl_float2>::
gflops_formula()
{
    return "8.0*M*N*K/time";
}

template<>
std::string
xGemm<cl_double2>::
gflops_formula()
{
    return "8.0*M*N*K/time";
}


template<>
void
xGemm<cl_float>::
verfication()
{
    /*
        sgemm(encodeTranspose(buffer.transA), encodeTranspose(buffer.transB),
        buffer.m_, buffer.n_, buffer.k_,
        buffer.alpha_,
        buffer.hA + buffer.offA_, buffer.lda_,
        buffer.hB + buffer.offB_, buffer.ldb_,
        buffer.beta_,
        buffer.hR + buffer.offC_, buffer.ldc_);
    */
    
    //perform GEMM NN
    for (int i = 0; i < buffer.m_; i++) {
        for (int j = 0; j < buffer.n_; j++) {
            float tmp = 0.0f;
            for (int k = 0; k < buffer.k_; k++) {
                tmp = tmp + buffer.hA[i + k * buffer.lda_] * buffer.hB[k + j * buffer.ldb_];
            }
            buffer.hR[i + j * buffer.ldc_] = buffer.hR[i + j * buffer.ldc_] * buffer.beta_ + tmp * buffer.alpha_;
        }
    }
    
    float norm_acml = slange('F', buffer.m_, buffer.n_, (float*)buffer.hR, buffer.ldc_);

    float alpha = -1.0f;

    float norm_clblas = slange('F', buffer.m_, buffer.n_, (float*)buffer.hC, buffer.ldc_);

    std::cout << "norm of ACML C: " << norm_acml << "; norm of clBLAS C:" << norm_clblas <<  std::endl;
    
    float error = 0.0f;
    for (int i = 0; i < buffer.m_; i++)
    {
        for (int j = 0; j < buffer.n_; j++)
        {
            float err = fabs(buffer.hR[i + j * buffer.ldc_] - buffer.hC[i + j * buffer.ldc_]);
            //std::cout << err << std::endl; 
            error = fmax(err, error);
        }
    }

    saxpy(buffer.ldc_ * buffer.n_, alpha, (float*)buffer.hR, 1, (float*)buffer.hC, 1);

    float norm = slange('F', buffer.m_, buffer.n_, (float*)buffer.hC, buffer.ldc_) / norm_acml;

    std::cout << "biggest error: " << error << ";norm error compared to ACML :" << norm << std::endl;
}

template<>
void
xGemm<cl_double>::
verfication()
{

}

template<>
void
xGemm<cl_float2>::
verfication()
{

}

template<>
void
xGemm<cl_double2>::
verfication()
{

    zgemm(encodeTranspose(buffer.transA), encodeTranspose(buffer.transB),
        buffer.m_, buffer.n_, buffer.k_,
        (doublecomplex*)&(buffer.alpha_),
        (doublecomplex*)buffer.hA, buffer.lda_,
        (doublecomplex*)buffer.hB, buffer.ldb_,
        (doublecomplex*)&(buffer.beta_),
        (doublecomplex*)buffer.hR, buffer.ldc_);

    
    /*
    void
    zgemm(char transa, char transb, int m, int n, int k, doublecomplex *alpha, doublecomplex *a, int lda, doublecomplex *b, int ldb, doublecomplex *beta, doublecomplex *c, int ldc)
    {
    zgemm_(&transa, &transb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc);
    }
    */

    double norm = zlange('F', buffer.m_, buffer.n_, (doublecomplex*)buffer.hR, buffer.ldc_);

    doublecomplex alpha;
    alpha.real = -1.0; alpha.imag = 0.0;

    zaxpy(buffer.m_ * buffer.n_, &alpha, (doublecomplex*)buffer.hR, 1, (doublecomplex*)buffer.hC, 1);

    norm = zlange('F', buffer.m_, buffer.n_, (doublecomplex*)buffer.hC, buffer.ldc_) / norm;

}
#endif // ifndef CLBLAS_BENCHMARK_XGEMM_HXX__
