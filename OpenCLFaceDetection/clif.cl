constant float rgb_to_grayscale_coeff[4] = { 0.299, 0.587, 0.114, 0 };
constant float4 rgb_to_grayscale_coeff_v = (0.299, 0.587, 0.114, 0);

kernel void bgrToGrayscale(global uchar* src,
                           global uchar* dst,
                           uint width,
                           uint height,
                           uint stride)
{
    int coord = (get_global_id(1) * stride) + (get_global_id(0) * 3);
    
    uint temp = (uint)(rgb_to_grayscale_coeff[2] * src[coord] +
                       rgb_to_grayscale_coeff[1] * src[coord + 1] +
                       rgb_to_grayscale_coeff[0] * src[coord + 2]);
    temp = clamp((uint)temp, (uint)0, (uint)255);
    
    dst[(get_global_id(1) * width) + get_global_id(0)] = (uchar)temp;
}

kernel void bgraToGrayscale(global uchar4* src,
                            global uchar* dst,
                            uint width,
                            uint height,
                            uint stride)
{
    int coord = (get_global_id(1) * stride) + (get_global_id(0) * 4);
    uchar4 data = src[coord];
    float result1 = data.x * rgb_to_grayscale_coeff_v.x;
    float result2 = data.y * rgb_to_grayscale_coeff_v.y;
    float result3 = data.z * rgb_to_grayscale_coeff_v.z;
    uint result = clamp((uint)(result1 + result2 + result3), (uint)0, (uint)255);
    dst[(get_global_id(1) * width) + get_global_id(0)] = (uchar)result;
}

kernel void bgrToGrayscalePerRow(global uchar3* src,
                                 global uchar4* dst,
                                 uint width,
                                 uint stride)
{
    int src_start = (get_global_id(0) * stride);
    int dst_start = (get_global_id(0) * (width >> 2));
    uchar3 src_pixel;
    uchar4 dst_pixels;
    uint temp;
    uint dst_index = 0;
    for(uint i = 0; i < width; i+=4) {
        src_pixel = src[src_start + i];
        temp = (uint)(rgb_to_grayscale_coeff[2] * src_pixel.x +
                      rgb_to_grayscale_coeff[1] * src_pixel.y +
                      rgb_to_grayscale_coeff[0] * src_pixel.z);
        dst_pixels.x = (uchar)clamp((uint)temp, (uint)0, (uint)255);
        
        src_pixel = src[src_start + i + 1];
        temp = (uint)(rgb_to_grayscale_coeff[2] * src_pixel.x +
                      rgb_to_grayscale_coeff[1] * src_pixel.y +
                      rgb_to_grayscale_coeff[0] * src_pixel.z);
        dst_pixels.y = (uchar)clamp((uint)temp, (uint)0, (uint)255);
        
        src_pixel = src[src_start + i + 2];
        temp = (uint)(rgb_to_grayscale_coeff[2] * src_pixel.x +
                      rgb_to_grayscale_coeff[1] * src_pixel.y +
                      rgb_to_grayscale_coeff[0] * src_pixel.z);
        dst_pixels.z = (uchar)clamp((uint)temp, (uint)0, (uint)255);
        
        src_pixel = src[src_start + i + 3];
        temp = (uint)(rgb_to_grayscale_coeff[2] * src_pixel.x +
                      rgb_to_grayscale_coeff[1] * src_pixel.y +
                      rgb_to_grayscale_coeff[0] * src_pixel.z);
        dst_pixels.w = (uchar)clamp((uint)temp, (uint)0, (uint)255);
        
        dst[dst_start + dst_index] = dst_pixels;
        dst_index++;
    }
}


// Input is grayscale 8U image
// Output is a (width + 1) * (height + 1) 32U image
kernel void integralImageSumRows(global uchar* src,
                                 global uint* dst,
                                 global ulong* dst_square,
                                 uint width,
                                 uint stride)
{
    // First row is 0
    int src_start = get_global_id(0) * stride;
    int dst_start = (get_global_id(0) + 1) * (width + 1);
    
    dst[dst_start] = 0;
    uint sum = 0;
    uint sum_square = 0;
    for(uint col = 0; col < width; col++) {
        uchar src_el = src[src_start + col];
        sum += src_el;
        sum_square += (src_el * src_el);
        dst[dst_start + col + 1] = sum;
        dst_square[dst_start + col + 1] = sum_square;
    }
}

// Input is a 32U image
// Output is a 32U image
kernel void integralImageSumCols(global uint* src,
                                 global ulong* src_square,
                                 global uint* dst,
                                 global ulong* dst_square,
                                 uint width,
                                 uint height)
{
    // First columns is 0
    int start = (get_global_id(0) + 1);
    
    uint sum = 0;
    for(uint row = 0; row < height; row ++) {
        uint src_el = src[start + (row * (width + 1))];
        sum += src_el;
        dst[start + (row * (width + 1))] = sum;
        dst_square[start + (row * (width + 1))] = sum;
    }
}


kernel void invert(global uchar* bmp,
                   global uchar* temp,
                   uint width,
                   uint height,
                   uint stride)
{    
    int coord = (get_global_id(1) * stride) + (get_global_id(0) * 3);
    
    uchar pix = bmp[coord];
    temp[coord] = 255 - pix;
    pix = bmp[coord + 1];
    temp[coord + 1] = 255 - pix;
    pix = bmp[coord + 2];
    temp[coord + 2] = 255 - pix;
}
