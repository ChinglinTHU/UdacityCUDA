/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

__global__ void get_max_luminance(
	const float* const d_input,
	float* maxInBlocks,
	const unsigned int pageSize)
{
	extern __shared__ float sdata[];
	int blockSize = blockDim.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int globalId = bid * blockSize + tid;
	int range = blockSize;
	if ((bid + 1) * blockSize > pageSize)
	{
		range = pageSize - (gridDim.x - 1) * blockSize;
	}
	if (tid < range)
	{
		sdata[tid] = d_input[globalId];
	}
	__syncthreads();
	int temp = blockSize;
	while(temp > 1)
	{
		int temprange = temp / 2;
		if (tid + temprange < temp)
		{
			sdata[tid] = fmaxf(sdata[tid], sdata[tid + temprange]);
		}
		temp -= temprange;
	}
	if (tid == 0)
	{
		maxInBlocks[bid] = sdata[0];
	}
}
__global__ void get_min_luminance(
	const float* const d_input,
	float* minInBlocks,
	const unsigned int pageSize)
{
	extern __shared__ float sdata[];
	int blockSize = blockDim.x;
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int globalId = bid * blockSize + tid;
	int range = blockSize;
	if ((bid + 1) * blockSize > pageSize)
	{
		range = pageSize - (gridDim.x - 1) * blockSize;
	}
	if (tid < range)
	{
		sdata[tid] = d_input[globalId];
	}
	__syncthreads();
	int temp = blockSize;
	while (temp > 1)
	{
		int temprange = temp / 2;
		if (tid + temprange < temp)
		{
			sdata[tid] = fminf(sdata[tid], sdata[tid + temprange]);
		}
		temp -= temprange;
	}
	if (tid == 0)
	{
		minInBlocks[bid] = sdata[0];
	}
}

__global__ void get_cdf(
	const float* const d_input,
	unsigned int* d_cdf,
	const float min_luminance,
	const float max_luminance,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins
)
{	
	int global_x = threadIdx.x + blockIdx.x * blockDim.x;
	int global_y = threadIdx.y + blockIdx.y * blockDim.y;
	if (global_x < numRows && global_y < numCols)
	{
		int pixelId = global_x * numCols + global_y;
		float data = d_input[pixelId];
		int binId = (data - min_luminance) / (max_luminance - min_luminance) * numBins;
		if (binId < numBins)
		{
			atomicAdd(&d_cdf[binId], 1);			
		}
	}
	
	
	
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
	2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
	int maxThreadsPerBlock = 1024;
	unsigned int numPixels = numCols * numRows;
	unsigned int numMegaPixels = numPixels / (1 << 20);	
	unsigned int pageSize = numPixels % (1 << 20);	
	float *minInBlocks, *maxInBlocks;
	float *d_min, *d_max;
	checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
	//int numBlocksUsed = 0;
	//float *h_intermediate;
	if (numMegaPixels == 0) // can be reduced in two rounds
	{
		dim3 blockSize(maxThreadsPerBlock, 1, 1);
		dim3 gridSize(pageSize / blockSize.x + 1, 1, 1);		
		if (gridSize.x > 1)
		{
			checkCudaErrors(cudaMalloc(&minInBlocks, sizeof(float)*gridSize.x));
			checkCudaErrors(cudaMalloc(&maxInBlocks, sizeof(float)*gridSize.x));
			get_min_luminance << <gridSize, blockSize, blockSize.x * sizeof(float) >> > (d_logLuminance, minInBlocks, pageSize);
			get_max_luminance << <gridSize, blockSize, blockSize.x * sizeof(float) >> > (d_logLuminance, maxInBlocks, pageSize);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			int next = pow(2, ceil(log(gridSize.x) / log(2)));						
			get_min_luminance << <1, next, next * sizeof(float) >> > (minInBlocks, d_min, gridSize.x);
			get_max_luminance << <1, next, next * sizeof(float) >> > (maxInBlocks, d_max, gridSize.x);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

			//h_intermediate = (float *)malloc(sizeof(float)*gridSize.x);
			//cudaMemcpy(h_intermediate, maxInBlocks, sizeof(float) * gridSize.x, cudaMemcpyDeviceToHost);
			//numBlocksUsed = gridSize.x;
		}
		else
		{
			get_min_luminance << <gridSize, blockSize, blockSize.x * sizeof(float) >> > (d_logLuminance, d_min, pageSize);
			get_max_luminance << <gridSize, blockSize, blockSize.x * sizeof(float) >> > (d_logLuminance, d_max, pageSize);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		}
	}
	checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));	
	//std::cout << min_logLum << '\t' << max_logLum << std::endl;
	float dynamicRange = max_logLum - min_logLum;
	dim3 blockSize(32, 32, 1);
	dim3 gridSize(numRows / 32 + 1, numCols / 32 + 1, 1);
	get_cdf << <gridSize, blockSize >> > (d_logLuminance, d_cdf, min_logLum, max_logLum, numRows, numCols, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	unsigned int* hist = (unsigned int*)malloc(sizeof(unsigned int) * numBins);
	unsigned int* cdf = (unsigned int*)malloc(sizeof(unsigned int) * numBins);
	cudaMemcpy(hist, d_cdf, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost);
	cdf[0] = 0;
	for (int i = 1; i < numBins; ++i)
	{
		//printf("%5d\n", cdf[i]);
		cdf[i] = hist[i-1] + cdf[i-1];		
	}
	cudaMemcpy(d_cdf, cdf, sizeof(unsigned int) * numBins, cudaMemcpyHostToDevice);
}
