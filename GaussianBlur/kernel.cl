__kernel void calculate_pixel(
    __global const unsigned char* ImgIn,
    __global const float* GaussKernel,
    int radius,
    int width,
    int height,
    __global unsigned char* ImgOut)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    int diameter = radius * 2;

    if (col == 0 && row == 0)
    {
        printf("radius: %d \n", radius);
        printf("diameter: %d \n", diameter);
        printf("width: %d \n", width);
        printf("height: %d \n", height);
    }

    for (int c = 0; c < 3; c++)
    {
        float sum = 0;
        float sumKernel = 0;

        for (int y = -radius; y <= radius; y++)
        {
            for (int x = -radius; x <= radius; x++)
            {
                float kernelValue = GaussKernel[(x + radius) * diameter + y + radius];

                //if (row == 0 && col == 0 && c == 0) {
                //    printf("%f \n", kernelValue);
                //}

                int rowPixel = row;
                int colPixel = col;

                if ((row + y) >= 0 && (row + y) < height)
                {
                    rowPixel += y;
                }

                if ((col + x) >= 0 && (col + x) < width)
                {
                    colPixel += x;
                }

                float color = ImgIn[rowPixel * 3 * width + colPixel * 3 + c];
                sum += color * kernelValue;
                sumKernel += kernelValue;
            }
        }

        ImgOut[3 * row * width + 3 * col + c] = sum / sumKernel;
    }
}