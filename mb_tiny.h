#ifndef MB_TINY_H
#define MB_TINY_H

#ifdef __cplusplus
extern "C" {
#endif
typedef struct convLayer {
	const float* weights;
	const float* bias;
	unsigned in_channels;
	unsigned out_channels;
	unsigned kernel_size;
	unsigned stride;
	unsigned padding;
	unsigned groups;
}* convLayer_t;
typedef struct detection {
	float x1, y1, x2, y2;
	float score;
}* detection_t;
// 使用static struct mb_tiny_context ctx_struct; 创建一个静态的 mb_tiny_context_t 变量
typedef struct mb_tiny_context {
	convLayer_t layers;
	float buf_a[307200];  // 3*240*320 (输入图像缓冲区)
	float buf_b[614400];  // 128*15*20 (中间特征图缓冲区)
	float cls1[2 * 4420];  // ([1, 3600, 2]) ([1, 600, 2]) ([1, 160, 2]) ([1, 60, 2])
	float reg1[4 * 4420];  // ([1, 3600, 4]) ([1, 600, 4]) ([1, 160, 4]) ([1, 60, 4])
	float* cls_2_3_4[3];  // {&cls1[7200], &cls1[8400], &cls1[8720]}
	float* reg_2_3_4[3];  // {&reg1[14400], &reg1[16800], &reg1[17440]}
	float priors[4420][4];
}* mb_tiny_context_t;
extern float score_threshold, iou_threshold;
extern int top_k;
void mb_tiny_init(mb_tiny_context_t ctx, const convLayer_t layers);
unsigned mb_tiny_detect(
    mb_tiny_context_t ctx, const unsigned char* input_rgb, unsigned width, unsigned height, detection_t detections);
void mb_tiny_draw(unsigned char* image, unsigned width, unsigned height, const detection_t detections, unsigned num_detections,
    const unsigned char color[3], unsigned char thickness);

#ifdef __cplusplus
}
#endif

#endif	// MB_TINY_H
