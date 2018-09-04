#ifdef __cplusplus
extern "C" {
#endif

void nms(const float *bboxesInDevice, long numBoxes, float threshold, long *keepIndices, long *numKeepBoxes);

#ifdef __cplusplus
}
#endif