#include <THC/THC.h>
#include "nms_cuda.h"

extern THCState *state;

int suppress(THCudaTensor *bboxes, float threshold, THCudaLongTensor *keepIndices) {
    if (!((THCudaTensor_nDimension(state, bboxes) == 2) && (THCudaTensor_size(state, bboxes, 1) == 4)))
        return 0;

    long numBoxes = THCudaTensor_size(state, bboxes, 0);
    THLongTensor *keepIndicesTmp = THLongTensor_newWithSize1d(numBoxes);

    long numKeepBoxes;
    nms(THCudaTensor_data(state, bboxes), numBoxes, threshold, THLongTensor_data(keepIndicesTmp), &numKeepBoxes);

    THLongTensor_resize1d(keepIndicesTmp, numKeepBoxes);
    THCudaLongTensor_resize1d(state, keepIndices, numKeepBoxes);
    THCudaLongTensor_copyCPU(state, keepIndices, keepIndicesTmp);

    THLongTensor_free(keepIndicesTmp);

    return 1;
}
