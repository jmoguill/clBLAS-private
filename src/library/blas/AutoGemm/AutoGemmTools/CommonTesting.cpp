#include <iostream>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <string>
//#include <Windows.h>
#if defined( __APPLE__ ) || defined( __MACOSX )
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

cl_platform_id
getPlatform(const char *name)
{
    cl_int err;
    cl_uint nrPlatforms, i;
    cl_platform_id *list, platform;
    char platformName[64];

    err = clGetPlatformIDs(0, NULL, &nrPlatforms);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) {
        return NULL;
    }

    list = (cl_platform_id*)malloc(nrPlatforms * sizeof(*list));
    if (list == NULL) {
        return NULL;
    }

    err = clGetPlatformIDs(nrPlatforms, list, NULL);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) {
        free(list);
        return NULL;
    }

    platform = NULL;
    for (i = 0; i < nrPlatforms; i++) {
        err = clGetPlatformInfo(list[i], CL_PLATFORM_NAME,
            sizeof(platformName), platformName, NULL);
        assert(err == CL_SUCCESS);
        if ((err == CL_SUCCESS) && (strcmp(platformName, name) == 0)) {
            platform = list[i];
            break;
        }
    }

    free(list);
    return platform;
}

cl_device_id
getDevice(
    cl_platform_id platform,
    const char *name)
{

    cl_int err;
    cl_uint nrDevices, i;
    cl_device_id *list, device;
    char deviceName[64];

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &nrDevices);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) {
        return NULL;
    }
    list = (cl_device_id*)malloc(nrDevices * sizeof(*list));
    assert(list);
    if (list == NULL) {
        return NULL;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, nrDevices, list, NULL);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) {
        free(list);
        return NULL;
    }

    device = NULL;
    for (i = 0; i < nrDevices; i++) {
        err = clGetDeviceInfo(list[i], CL_DEVICE_NAME,
            sizeof(deviceName), deviceName, NULL);
        assert(err == CL_SUCCESS);
        if ((err == CL_SUCCESS) && (strcmp(deviceName, name) == 0)) {
            device = list[i];
            break;
        }
    }

    free(list);
    return device;
}

cl_device_id
getPlatformDevice()
{
    cl_int err;
    cl_uint num_platforms, max_devices, num_devices;
    max_devices = 1024;
    cl_platform_id *platforms = NULL;
    cl_device_id *device_list = NULL;
    cl_device_id device;
    clGetPlatformIDs(0, NULL, &num_platforms);
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id)*num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);

    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) {
        free(platforms);
        return NULL;
    }

    device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, device_list, NULL);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) {
        free(platforms);
        free(device_list);
        return NULL;
    }

    std::cout << "Number of platforms : " << num_platforms << std::endl;
    std::cout << "Number of GPUs :  " << num_devices << std::endl;
    std::cout << "Testing on GPU 0" << std::endl;

    char buffer[1024];

    for (int i = 0; i<num_devices; i++) {

        clGetDeviceInfo(device_list[i], CL_DEVICE_PLATFORM, sizeof(buffer), buffer, NULL);
        std::cout << "Platform : " << buffer << std::endl;

        clGetDeviceInfo(device_list[i], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
        std::cout << "GPU " << i << ": " << buffer << std::endl;

    }

    device = device_list[0];

    free(device_list);
    free(platforms);

    return device;
}


cl_kernel
createKernel(
    const char* source,
    cl_context context,
    const char* options,
    cl_int* error)
{

    printf("BuildOptions: %s\n", options);

    //printf("Kernel Source:\n%s", source );
    cl_int err;
    cl_device_id device;
    cl_program program;
    cl_kernel kernel;
    size_t logSize;
    char *log;

    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device), &device, NULL);
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) {
        if (error != NULL) {
            *error = err;
        }
        return NULL;
    }

    program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    assert(err == CL_SUCCESS);
    assert(program != NULL);
    if (program == NULL) {
        if (error != NULL) {
            *error = err;
        }
        return NULL;
    }

    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        log = (char*)malloc(logSize + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        printf("=== Build Log [%lu]===\n%s\n", logSize, log);
        free(log);
    }
    assert(err == CL_SUCCESS);
    if (err != CL_SUCCESS) {
        clReleaseProgram(program);
        if (error != NULL) {
            *error = err;
        }
        return NULL;
    }

    kernel = NULL;
    cl_uint num_kernels_ret;
    err = clCreateKernelsInProgram(program, 0, NULL, &num_kernels_ret);
    err = clCreateKernelsInProgram(program, 1, &kernel, NULL);
    assert(err == CL_SUCCESS);
    assert(kernel != NULL);
    clReleaseProgram(program);

    // kernel name
    size_t length;
    char kernelName[64];
    err = clGetKernelInfo(
        kernel,
        CL_KERNEL_FUNCTION_NAME,
        64,
        kernelName,
        &length);
    printf("KernelName[%lu]: %s\n", length, kernelName);

    // kernel arguments
    cl_uint numArguments;
    err = clGetKernelInfo(
        kernel,
        CL_KERNEL_NUM_ARGS,
        sizeof(numArguments),
        &numArguments,
        NULL);

    if (error != NULL) {
        *error = err;
    }
    return kernel;
}



