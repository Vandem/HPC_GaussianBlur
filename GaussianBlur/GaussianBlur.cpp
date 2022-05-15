#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#if _WIN32
#include <CL/cl.h>
#elif __APPLE__
#include <OpenCL/opencl.h>
#endif

#include <fstream>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

std::string cl_errorstring(cl_int err)
{
	switch (err)
	{
	case CL_SUCCESS:									return std::string("Success");
	case CL_DEVICE_NOT_FOUND:							return std::string("Device not found");
	case CL_DEVICE_NOT_AVAILABLE:						return std::string("Device not available");
	case CL_COMPILER_NOT_AVAILABLE:						return std::string("Compiler not available");
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:				return std::string("Memory object allocation failure");
	case CL_OUT_OF_RESOURCES:							return std::string("Out of resources");
	case CL_OUT_OF_HOST_MEMORY:							return std::string("Out of host memory");
	case CL_PROFILING_INFO_NOT_AVAILABLE:				return std::string("Profiling information not available");
	case CL_MEM_COPY_OVERLAP:							return std::string("Memory copy overlap");
	case CL_IMAGE_FORMAT_MISMATCH:						return std::string("Image format mismatch");
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:					return std::string("Image format not supported");
	case CL_BUILD_PROGRAM_FAILURE:						return std::string("Program build failure");
	case CL_MAP_FAILURE:								return std::string("Map failure");
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:				return std::string("Misaligned sub buffer offset");
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:	return std::string("Exec status error for events in wait list");
	case CL_INVALID_VALUE:                    			return std::string("Invalid value");
	case CL_INVALID_DEVICE_TYPE:              			return std::string("Invalid device type");
	case CL_INVALID_PLATFORM:                 			return std::string("Invalid platform");
	case CL_INVALID_DEVICE:                   			return std::string("Invalid device");
	case CL_INVALID_CONTEXT:                  			return std::string("Invalid context");
	case CL_INVALID_QUEUE_PROPERTIES:         			return std::string("Invalid queue properties");
	case CL_INVALID_COMMAND_QUEUE:            			return std::string("Invalid command queue");
	case CL_INVALID_HOST_PTR:                 			return std::string("Invalid host pointer");
	case CL_INVALID_MEM_OBJECT:               			return std::string("Invalid memory object");
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  			return std::string("Invalid image format descriptor");
	case CL_INVALID_IMAGE_SIZE:               			return std::string("Invalid image size");
	case CL_INVALID_SAMPLER:                  			return std::string("Invalid sampler");
	case CL_INVALID_BINARY:                   			return std::string("Invalid binary");
	case CL_INVALID_BUILD_OPTIONS:            			return std::string("Invalid build options");
	case CL_INVALID_PROGRAM:                  			return std::string("Invalid program");
	case CL_INVALID_PROGRAM_EXECUTABLE:       			return std::string("Invalid program executable");
	case CL_INVALID_KERNEL_NAME:              			return std::string("Invalid kernel name");
	case CL_INVALID_KERNEL_DEFINITION:        			return std::string("Invalid kernel definition");
	case CL_INVALID_KERNEL:                   			return std::string("Invalid kernel");
	case CL_INVALID_ARG_INDEX:                			return std::string("Invalid argument index");
	case CL_INVALID_ARG_VALUE:                			return std::string("Invalid argument value");
	case CL_INVALID_ARG_SIZE:                 			return std::string("Invalid argument size");
	case CL_INVALID_KERNEL_ARGS:             			return std::string("Invalid kernel arguments");
	case CL_INVALID_WORK_DIMENSION:          			return std::string("Invalid work dimension");
	case CL_INVALID_WORK_GROUP_SIZE:          			return std::string("Invalid work group size");
	case CL_INVALID_WORK_ITEM_SIZE:           			return std::string("Invalid work item size");
	case CL_INVALID_GLOBAL_OFFSET:            			return std::string("Invalid global offset");
	case CL_INVALID_EVENT_WAIT_LIST:          			return std::string("Invalid event wait list");
	case CL_INVALID_EVENT:                    			return std::string("Invalid event");
	case CL_INVALID_OPERATION:                			return std::string("Invalid operation");
	case CL_INVALID_GL_OBJECT:                			return std::string("Invalid OpenGL object");
	case CL_INVALID_BUFFER_SIZE:              			return std::string("Invalid buffer size");
	case CL_INVALID_MIP_LEVEL:                			return std::string("Invalid mip-map level");
	case CL_INVALID_GLOBAL_WORK_SIZE:         			return std::string("Invalid gloal work size");
	case CL_INVALID_PROPERTY:                 			return std::string("Invalid property");
	default:                                  			return std::string("Unknown error code");
	}
}

void checkStatus(cl_int err)
{
	if (err != CL_SUCCESS) {
		printf("OpenCL Error: %s \n", cl_errorstring(err).c_str());
		exit(EXIT_FAILURE);
	}
}

void printCompilerError(cl_program program, cl_device_id device)
{
	cl_int status;
	size_t logSize;
	char* log;

	// get log size
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	checkStatus(status);

	// allocate space for log
	log = static_cast<char*>(malloc(logSize));
	if (!log)
	{
		exit(EXIT_FAILURE);
	}

	// read the log
	status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
	checkStatus(status);

	// print the log
	printf("Build Error: %s\n", log);
}

void printVector(int32_t* vector, unsigned int elementSize, const char* label)
{
	printf("%s:\n", label);

	for (unsigned int i = 0; i < elementSize; ++i)
	{
		printf("%d ", vector[i]);
	}

	printf("\n");
}

float* generateKernel(int diameter, float sigma = 1.0f)
{
	int x, y, mean;
	float sum;

	float* kernel(new float[diameter * diameter]);
	mean = diameter / 2;
	sum = 0.0; // For accumulating the kernel values

	for (x = 0; x < diameter; ++x)
		for (y = 0; y < diameter; ++y) {
			kernel[y * diameter + x] = (float)(exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) / (2 * M_PI * sigma * sigma));

			// Accumulate the kernel values
			sum += kernel[y * diameter + x];
		}

	// Normalize the kernel
	for (x = 0; x < diameter; ++x)
		for (y = 0; y < diameter; ++y)
			kernel[y * diameter + x] /= sum;

	return kernel;
}

int main(int argc, char** argv)
{
	Mat img = cv::imread("lena.jpg");
	Mat out = Mat(img.rows, img.cols, img.type());

	int32_t radius = 9;
	int32_t diameter = 2 * radius + 1;
	float sigma = 1.0f;

	int32_t width = img.cols;
	int32_t height = img.rows;

	size_t dataSizeImg = width * height * 3 * sizeof(unsigned char);

	float* gaussKernel = generateKernel(diameter, sigma);
	size_t dataSizeKernel = sizeof(float) * diameter * diameter;

	// used for checking error status of api calls
	cl_int status;

	// retrieve the number of platforms
	cl_uint numPlatforms = 0;
	checkStatus(clGetPlatformIDs(0, NULL, &numPlatforms));

	if (numPlatforms == 0)
	{
		printf("Error: No OpenCL platform available!\n");
		exit(EXIT_FAILURE);
	}

	// select the platform
	cl_platform_id platform;
	checkStatus(clGetPlatformIDs(1, &platform, NULL));

	// retrieve the number of devices
	cl_uint numDevices = 0;
	checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));

	if (numDevices == 0)
	{
		printf("Error: No OpenCL device available for platform!\n");
		exit(EXIT_FAILURE);
	}

	// select the device
	cl_device_id device;
	checkStatus(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

	// create context
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
	checkStatus(status);

	// create command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, &status);
	checkStatus(status);

	cl_mem bufferImageIn = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSizeImg, NULL, &status);
	checkStatus(status);
	cl_mem bufferGaussKernel = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSizeKernel, NULL, &status);
	checkStatus(status);
	cl_mem bufferImageOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSizeImg, NULL, &status);
	checkStatus(status);

	checkStatus(clEnqueueWriteBuffer(commandQueue, bufferImageIn, CL_TRUE, 0, dataSizeImg, img.data, 0, NULL, NULL));
	checkStatus(clEnqueueWriteBuffer(commandQueue, bufferGaussKernel, CL_TRUE, 0, dataSizeKernel, gaussKernel, 0, NULL, NULL));

	// read the kernel source
	const char* kernelFileName = "kernel.cl";
	std::ifstream ifs(kernelFileName);
	if (!ifs.good())
	{
		printf("Error: Could not open kernel with file name %s!\n", kernelFileName);
		exit(EXIT_FAILURE);
	}

	std::string programSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	const char* programSourceArray = programSource.c_str();
	size_t programSize = programSource.length();

	// create the program
	cl_program program = clCreateProgramWithSource(context, 1, static_cast<const char**>(&programSourceArray), &programSize, &status);
	checkStatus(status);

	// build the program
	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		printCompilerError(program, device);
		exit(EXIT_FAILURE);
	}

	// create the vector addition kernel
	//cl_kernel kernel = clCreateKernel(program, "vector_add", &status);
	cl_kernel kernel = clCreateKernel(program, "calculate_pixel", &status);
	checkStatus(status);

	// set the kernel arguments
	checkStatus(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferImageIn));
	checkStatus(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferGaussKernel));
	checkStatus(clSetKernelArg(kernel, 2, sizeof(radius), &radius));
	checkStatus(clSetKernelArg(kernel, 3, sizeof(width), &width));
	checkStatus(clSetKernelArg(kernel, 4, sizeof(height), &height));
	checkStatus(clSetKernelArg(kernel, 5, sizeof(cl_mem), &bufferImageOut));

	// output device capabilities
	size_t maxWorkGroupSize;
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL));
	printf("Device Capabilities: Max work items in single group: %zu\n", maxWorkGroupSize);

	cl_uint maxWorkItemDimensions;
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWorkItemDimensions, NULL));
	printf("Device Capabilities: Max work item dimensions: %u\n", maxWorkItemDimensions);

	size_t* maxWorkItemSizes = static_cast<size_t*>(malloc(maxWorkItemDimensions * sizeof(size_t)));
	checkStatus(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, maxWorkItemDimensions * sizeof(size_t), maxWorkItemSizes, NULL));
	printf("Device Capabilities: Max work items in group per dimension:");
	for (cl_uint i = 0; i < maxWorkItemDimensions; ++i)
		printf(" %u:%zu", i, maxWorkItemSizes[i]);
	printf("\n");
	free(maxWorkItemSizes);

	// execute the kernel
	// ndrange capabilites only need to be checked when we specify a local work group size manually
	// in our case we provide NULL as local work group size, which means groups get formed automatically
	size_t globalWorkSize[2] = { width, height };
	checkStatus(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL));

	// read the device output buffer to the host output array
	checkStatus(clEnqueueReadBuffer(commandQueue, bufferImageOut, CL_TRUE, 0, dataSizeImg, out.data, 0, NULL, NULL));

	// output result
	imshow("img", img);
	imshow("out", out);

	// release opencl objects
	checkStatus(clReleaseKernel(kernel));
	checkStatus(clReleaseProgram(program));
	checkStatus(clReleaseCommandQueue(commandQueue));
	checkStatus(clReleaseContext(context));

	waitKey(0);

	exit(EXIT_SUCCESS);
}
