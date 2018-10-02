import os

from torch.utils.ffi import create_extension

ffi = create_extension(
    name='_ext.nms',
    headers=['src/nms.h'],
    sources=['src/nms.c'],
    extra_objects=[os.path.join(os.path.dirname(os.path.abspath(__file__)), it) for it in ['src/nms_cuda.o']],
    relative_to=__file__,
    with_cuda=True
)

if __name__ == '__main__':
    ffi.build()
