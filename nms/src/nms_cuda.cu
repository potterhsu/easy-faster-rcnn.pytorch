#include "nms_cuda.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

typedef unsigned long long MaskType;

const long numThreadsPerBlock = sizeof(MaskType) * 8;

__device__ inline float iou(const float *bbox1, const float *bbox2) {
    float intersectionLeft = max(bbox1[0], bbox2[0]);
    float intersectionTop = max(bbox1[1], bbox2[1]);
    float intersectionRight = min(bbox1[2], bbox2[2]);
    float intersectionBottom = min(bbox1[3], bbox2[3]);
    float intersectionWidth = max(intersectionRight - intersectionLeft, 0.f);
    float intersectionHeight = max(intersectionBottom - intersectionTop, 0.f);
    float intersectionArea = intersectionWidth * intersectionHeight;
    float bbox1Area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    float bbox2Area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
    return intersectionArea / (bbox1Area + bbox2Area - intersectionArea);
}

__global__ void nms_kernel(const float *bboxes, long numBoxes, float threshold, MaskType *suppressionMask) {
    int i;
    int bidX = blockIdx.x;
    int bidY = blockIdx.y;
    int tid = threadIdx.x;
    const long blockBoxStartX = bidX * numThreadsPerBlock;
    const long blockBoxStartY = bidY * numThreadsPerBlock;
    const long blockBoxEndX = min(blockBoxStartX + numThreadsPerBlock, numBoxes);
    const long blockBoxEndY = min(blockBoxStartY + numThreadsPerBlock, numBoxes);
    const long currentBoxY = blockBoxStartY + tid;

    if (currentBoxY < blockBoxEndY) {
        MaskType suppression = 0;

        const float *currentBox = bboxes + currentBoxY * 4;
        for (i = 0; i < blockBoxEndX - blockBoxStartX; ++i) {
            long targetBoxX = blockBoxStartX + i;
            if (targetBoxX > currentBoxY) {
                const float *targetBox = bboxes + targetBoxX * 4;
                if (iou(currentBox, targetBox) > threshold) {
                    suppression |= 1ULL << i;
                }
            }
        }

        const long numBlockCols = DIVUP(numBoxes, numThreadsPerBlock);
        suppressionMask[currentBoxY * numBlockCols + bidX] = suppression;
    }
}

void nms(const float *bboxesInDevice, long numBoxes, float threshold, long *keepIndices, long *numKeepBoxes) {
    int i, j;
    const long numBlockCols = DIVUP(numBoxes, numThreadsPerBlock);

    MaskType *suppressionMaskInDevice;
    cudaMalloc(&suppressionMaskInDevice, sizeof(MaskType) * numBoxes * numBlockCols);

    dim3 blocks(numBlockCols, numBlockCols);
    dim3 threads(numThreadsPerBlock);
    nms_kernel<<<blocks, threads>>>(bboxesInDevice, numBoxes, threshold, suppressionMaskInDevice);

    MaskType *suppressionMask = (MaskType *) malloc(sizeof(MaskType) * numBoxes * numBlockCols);
    cudaMemcpy(suppressionMask, suppressionMaskInDevice, sizeof(MaskType) * numBoxes * numBlockCols, cudaMemcpyDeviceToHost);

    MaskType *maskRow = (MaskType *) malloc(sizeof(MaskType) * numBlockCols);
    memset(maskRow, 0, sizeof(MaskType) * numBlockCols);
    long nKeepBoxes = 0;
    for (i = 0; i < numBoxes; ++i) {
        long block = i / numThreadsPerBlock;
        long offset = i % numThreadsPerBlock;
        if (!(maskRow[block] & (1ULL << offset))) {
            keepIndices[nKeepBoxes++] = i;
            for (j = 0; j < numBlockCols; ++j) {
                maskRow[j] |= suppressionMask[i * numBlockCols + j];
            }
        }
    }
    *numKeepBoxes = nKeepBoxes;

    cudaFree(suppressionMaskInDevice);
    free(suppressionMask);
    free(maskRow);
}
