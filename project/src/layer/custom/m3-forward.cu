#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
// Kernel fusion for unrolling and matrix-multiplication 
__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int Width_grid = Width_out/TILE_WIDTH;
    if(0!=Width_out%TILE_WIDTH){Width_grid++;}
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_3d(i2,i1,i0) output[(size_t)(i1)*(Batch * Height_out * Width_out)+(i2)*(Height_out * Width_out)+i0]
    // TODO: Insert your input matrix unrolling kernel code here
    int b = blockIdx.z;
    int c = blockIdx.x;
    int h = (blockIdx.y/Width_grid)*TILE_WIDTH+threadIdx.y;
    int w = (blockIdx.y%Width_grid)*TILE_WIDTH+threadIdx.x;
    int w_base = c*K*K;
    if(h<Height_out && w<Width_out)
    {
        for(int p = 0;p<K;p++)
        {
            for(int q = 0;q<K;q++)
            {
                int h_unroll = w_base + p*K + q;
                int w_unroll = h*Width_out+w;
                out_3d(b,h_unroll,w_unroll) = in_4d(b,c,h+p,w+q);
            }
        }
    }

    #undef in_4d
    #undef out_3d
}

// Tiled matrix multiplication kernel. Computes C = AB
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}


// Fused kernel: Unroll input and perform matrix multiplication
__global__ void fused_kernel(const float *input, const float *mask, float *output,
                                         int Batch, int Channel, int Height, int Width, int K,int Map_out) 
{
    // Shared memory for tiles
    __shared__ float shared_mask[TILE_WIDTH][TILE_WIDTH];
    __shared__ float shared_input[TILE_WIDTH][TILE_WIDTH];
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int numAColumns = Channel*K*K;
    int numARows = Map_out;
    int numBRows = Channel*K*K;
    int numBColumns = Batch*Height_out*Width_out;
    int tx = threadIdx.x;  
    int ty = threadIdx.y;  
    int bx = blockIdx.x;   
    int by = blockIdx.y;   
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float value = 0.0;
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    for (int t = 0; t < (numAColumns - 1) / TILE_WIDTH + 1; ++t) 
    {
        if (Row < numARows && t * TILE_WIDTH + tx < numAColumns) 
        {
            shared_mask[ty][tx] = mask[(size_t)Row * numAColumns + (t * TILE_WIDTH + tx)];
        } 
        else 
        {
            shared_mask[ty][tx] = 0.0; 
        }
        if (t * TILE_WIDTH + ty < numBRows && Col < numBColumns) 
        {
            // int c = (t * TILE_WIDTH + ty )/(K*K);
            // int b = Col/(Height_out*Width_out);
            // int row_remain = (t * TILE_WIDTH + ty ) - c*K*K;
            // int col_remain = Col - b*Height_out*Width_out;
            // int p = row_remain/K;
            // int q = row_remain - p*K;
            // int h = col_remain/Width_out;
            // int w = col_remain - h*Width_out;
            // shared_input[ty][tx] = in_4d(b,c,h+p,w+q);
            // input[((size_t)t * TILE_WIDTH + ty) * numBColumns + Col];
            int c = (t * TILE_WIDTH + ty )/(K*K);
            int b = Col/(Height_out*Width_out);
            int row_remain = (t * TILE_WIDTH + ty )%(K*K);
            int col_remain = Col%(Height_out*Width_out);
            int p = row_remain/K;
            int q = row_remain%K;
            int h = col_remain/Width_out;
            int w = col_remain%Width_out;
            shared_input[ty][tx] = in_4d(b,c,h+p,w+q);
        } 
        else 
        {
            shared_input[ty][tx] = 0.0; 
        }
        __syncthreads(); // Ensure all threads have loaded their tiles

        // Perform partial matrix multiplication
        if(Row<numARows && Col<numBColumns)
        {
            for (int i = 0; i < TILE_WIDTH; ++i) 
            {
                value += shared_mask[ty][i] * shared_input[i][tx];
            }
        }
        __syncthreads(); // Ensure all threads complete computation before loading next tile
    }
    if (Row < numARows && Col < numBColumns) 
    {
        int m = Row;
        int b = Col/(Height_out*Width_out);
        int temp = Col%(Height_out*Width_out);
        output[b*Map_out*Height_out*Width_out+m*Height_out*Width_out+temp] = value;
        // output[Row * numBColumns + Col] = value;
    }
    #undef in_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void**)device_output_ptr, Batch*Map_out*(Height-K+1)*(Width-K+1)*sizeof(float));
    cudaMalloc((void**)device_input_ptr, Batch*Channel*Height*Width*sizeof(float));
    cudaMalloc((void**)device_mask_ptr,Map_out*Channel*K*K*sizeof(float));
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    cudaMemcpy(*device_input_ptr, host_input,Batch*Channel*Height*Width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr,host_mask,Map_out*Channel*K*K*sizeof(float),cudaMemcpyHostToDevice);
    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;
    int Width_grid = Width_out/TILE_WIDTH;
    int Height_grid = Height_out/TILE_WIDTH;
    if(0!=Width_out%TILE_WIDTH){Width_grid++;}
    if(0!=Height_out%TILE_WIDTH){Height_grid++;}
    dim3 DimGrid((Width_unrolled+TILE_WIDTH-1)/TILE_WIDTH,(Map_out+TILE_WIDTH-1)/TILE_WIDTH,1);
    dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,1);
    fused_kernel<<<DimGrid,DimBlock>>>(device_input,device_mask,device_output,Batch, Channel, Height, Width,K,Map_out);
    // float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    // float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    // cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    // cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    // dim3 DimGrid(Channel,Width_grid*Height_grid,Batch);
    // dim3 DimBlock(TILE_WIDTH,TILE_WIDTH,1);
    // matrix_unrolling_kernel<<<DimGrid,DimBlock>>>(device_input,unrolled_matrix,Batch,Channel,Height,Width,K);
    // cudaDeviceSynchronize();
    // TODO: Set the kernel dimensions and call the matmul kernel
    // dim3 DimGrid2((Width_unrolled+TILE_WIDTH-1)/TILE_WIDTH,(Map_out+TILE_WIDTH-1)/TILE_WIDTH,1);
    // dim3 DimBlock2(TILE_WIDTH,TILE_WIDTH,1);
    // matrixMultiplyShared<<<DimGrid2,DimBlock2>>>(device_mask,unrolled_matrix,matmul_output,Map_out,Height_unrolled,Height_unrolled,Width_unrolled,Map_out,Width_unrolled);
    // cudaDeviceSynchronize();
    // Permute the result of matrix multiplication
    // const int out_image_size = Height_out * Width_out;
    // dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    // matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
    //     matmul_output, device_output, Map_out, Batch, out_image_size
    // );
    cudaDeviceSynchronize();
    // cudaFree(matmul_output);
    // cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    cudaMemcpy(host_output,device_output,Batch*Map_out*(Height-K+1)*(Width-K+1)*sizeof(float),cudaMemcpyDeviceToHost);
    // TODO: Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}