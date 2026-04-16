#include "mb_tiny.h"
#include "./math.h"
#define MEAN_VAL 127.0f
#define STD_VAL 128.0f
#define CENTER_VARIANCE 0.1f
#define SIZE_VARIANCE 0.2f
#define NUM_CLASSES 2
#ifdef DEBUG
static inline void print_range(const char* name, const float* x, unsigned len) {
	float max = x[0], min = x[0];
	for(unsigned i = 1; i < len; i++) {
		if(x[i] > max) max = x[i];
		if(x[i] < min) min = x[i];
	}
	__builtin_printf("%s: min=%.6f, max=%.6f\n", name, min, max);
}
#define conv2d_fun /* empty for debug */
#else
#define print_range(...)
#define conv2d_fun static inline
#endif
float score_threshold = 0.6;
float iou_threshold = 0.3;
int top_k = -1;
#define W 320
#define H 240
#define CHW_INDEX(c, y, x, h, w) ((c) * (h) * (w) + (y) * (w) + (x))
#define HWC_INDEX(y, x, w) ((y) * (w) * 3 + (x) * 3)

static inline float clipf(float x, float min, float max) { return fminf(fmaxf(x, min), max); }

static inline float relu(float x) { return fmaxf(x, 0.0f); }

static inline void resize_and_normalize(const unsigned char* img, unsigned w, unsigned h, float* out) {
	// 将输入图像调整为WxH并归一化
	const float scale_x = (float)w / W;
	const float scale_y = (float)h / H;
	const float inv_std = 1.0f / STD_VAL;
	const float mean_offset = -MEAN_VAL * inv_std;
#pragma omp parallel for schedule(guided)
	for(unsigned y = 0; y < H; y++) {
		// 预计算y相关的值
		const float src_y = (y + 0.5f) * scale_y - 0.5f;
		unsigned y0 = floorf(src_y);
		const unsigned y1 = fminf(y0 + 1, h - 1);
		y0 = fmaxf(0, y0);
		const float dy = src_y - y0;
		const float dy_1 = 1.0f - dy;
		for(unsigned x = 0; x < W; x++) {
			// 双线性插值采样（中心点对齐）
			const float src_x = (x + 0.5f) * scale_x - 0.5f;

			unsigned x0 = floorf(src_x);
			const unsigned x1 = fminf(x0 + 1, w - 1);
			x0 = fmaxf(0, x0);

			const float dx = src_x - x0;
			const float dx_1 = 1.0f - dx;

			// 预计算双线性插值权重
			const float w00 = dx_1 * dy_1;
			const float w01 = dx * dy_1;
			const float w10 = dx_1 * dy;
			const float w11 = dx * dy;

			// 预计算输入像素指针
			const unsigned base_y0_x0 = HWC_INDEX(y0, x0, w);
			const unsigned base_y0_x1 = HWC_INDEX(y0, x1, w);
			const unsigned base_y1_x0 = HWC_INDEX(y1, x0, w);
			const unsigned base_y1_x1 = HWC_INDEX(y1, x1, w);

			for(unsigned c = 0; c < 3; c++) {  // RGB通道
				const float val = img[base_y0_x0 + c] * w00 + img[base_y0_x1 + c] * w01 +
						  img[base_y1_x0 + c] * w10 + img[base_y1_x1 + c] * w11;

				// 归一化: (val - mean) / std = val * inv_std + mean_offset
				out[CHW_INDEX(c, y, x, H, W)] = val * inv_std + mean_offset;
			}
		}
	}
}

// 循环顺序：oh→ow→oc，提高OpenMP效率。out直接赋值无需初始化(sum = bias;...;out = sum)。
conv2d_fun void conv2d_forward_relu(const convLayer_t layer, const float* in, unsigned in_h, unsigned in_w, float* out) {
	const unsigned out_channels = layer->out_channels;
	const unsigned in_channels = layer->in_channels;
	const unsigned kernel_size = layer->kernel_size;
	const unsigned stride = layer->stride;
	const unsigned padding = layer->padding;
	const unsigned groups = layer->groups;
	const unsigned in_channels_per_group = in_channels / groups;
	const unsigned out_channels_per_group = out_channels / groups;
	const unsigned kernel_area = kernel_size * kernel_size;
	const unsigned weight_ic_stride = kernel_area;
	const unsigned weight_oc_stride = in_channels * kernel_area;

	const unsigned out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
	const unsigned out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
	const unsigned in_hw = in_h * in_w;
	const unsigned is_depthwise = (groups == in_channels);

#pragma omp parallel for schedule(guided)
	for(unsigned oh = 0; oh < out_h; oh++) {
		const int ih_base = (int)oh * stride - padding;

		for(unsigned ow = 0; ow < out_w; ow++) {
			const int iw_base = (int)ow * stride - padding;

			for(unsigned oc = 0; oc < out_channels; oc++) {
				float sum = layer->bias[oc];

				const unsigned group_idx = oc / out_channels_per_group;
				const unsigned ic_start = group_idx * in_channels_per_group;
				const unsigned ic_end = ic_start + in_channels_per_group;

				// 对于depthwise conv (groups=in_channels)，权重布局不同
				const unsigned weight_oc_base = is_depthwise ? 0 : (oc * weight_oc_stride);

				for(unsigned ic = ic_start; ic < ic_end; ic++) {
					// depthwise: weight[ic * kernel_area + kh * kernel_size + kw]
					// normal:    weight[oc * in_channels * kernel_area + ic * kernel_area + kh * kernel_size
					// + kw]
					const unsigned weight_ic_base =
					    is_depthwise ? (ic * weight_ic_stride) : (weight_oc_base + ic * weight_ic_stride);
					const unsigned input_c_base = ic * in_hw;

					for(unsigned kh = 0; kh < kernel_size; kh++) {
						const int ih = ih_base + (int)kh;
						if(ih < 0 || ih >= (int)in_h) continue;

						const unsigned weight_kh_base = weight_ic_base + kh * kernel_size;
						const unsigned input_h_base = input_c_base + ih * in_w;

						for(unsigned kw = 0; kw < kernel_size; kw++) {
							const int iw = iw_base + (int)kw;
							if(iw < 0 || iw >= (int)in_w) continue;

							sum += layer->weights[weight_kh_base + kw] * in[input_h_base + iw];
						}
					}
				}

				out[CHW_INDEX(oc, oh, ow, out_h, out_w)] = relu(sum);
			}
		}
	}
}

// 专用函数: ConvReLU with kernel_size=1, stride=1, padding=0, groups=1 (Pointwise Conv + ReLU)
// 优化: 无空间维度循环，无边界检查，直接矩阵乘法
// 输入: CHW格式 (in_channels, in_h, in_w)
// 输出: CHW格式 (out_channels, in_h, in_w)，尺寸不变
conv2d_fun void conv1x1_relu_forward(const convLayer_t layer, const float* in, unsigned in_h, unsigned in_w, float* out) {
	const unsigned out_channels = layer->out_channels;
	const unsigned in_channels = layer->in_channels;
	const unsigned hw = in_h * in_w;

	// 权重布局: weight[oc * in_channels + ic] (1x1卷积)

#pragma omp parallel for schedule(guided)
	for(unsigned pos = 0; pos < hw; pos++) {
		// 对每个空间位置，执行完整的通道变换
		for(unsigned oc = 0; oc < out_channels; oc++) {
			float sum = layer->bias[oc];

			const unsigned weight_oc_base = oc * in_channels;

			// 1x1卷积 = 全连接层，对所有输入通道加权求和
			for(unsigned ic = 0; ic < in_channels; ic++) {
				sum += layer->weights[weight_oc_base + ic] * in[ic * hw + pos];
			}

			// ReLU激活并输出
			out[oc * hw + pos] = relu(sum);
		}
	}
}

// 融合: ConvReLU(depthwise) + Conv2d(pointwise) + permute + reshape
// 对应: ConvReLU(in_ch, in_ch, k=3, s=1, p=1, groups=in_ch) + Conv2d(in_ch, out_ch, k=1) + permute + reshape
// 输入: CHW格式 (in_channels, in_h, in_w)
// 输出: HWC展平格式 (in_h*in_w, out_channels)，即 [h*w + w][oc]
// 注意: buf_dw 需要预先分配，大小为 num_positions * in_channels
conv2d_fun void detection_head_fused(const convLayer_t depthwise_layer,	 // depthwise conv + ReLU
    const convLayer_t pointwise_layer,	// pointwise conv (1x1)
    const float* in, unsigned in_h, unsigned in_w, float* out,
    float* buf_dw  // buf_dw: 临时缓冲区，大小为 num_positions * in_channels
) {
	const unsigned in_channels = depthwise_layer->in_channels;
	const unsigned out_channels = pointwise_layer->out_channels;
	const unsigned dw_kernel_size = depthwise_layer->kernel_size;
	const unsigned dw_padding = depthwise_layer->padding;

	// depthwise conv 输出尺寸（stride=1, padding=1, kernel=3 时尺寸不变）
	const unsigned feat_h = in_h;  // stride=1, padding=1, kernel=3 => 输出尺寸不变
	const unsigned feat_w = in_w;
	const unsigned num_positions = feat_h * feat_w;
	const unsigned in_hw = in_h * in_w;

	// 预计算depthwise权重步长 (groups=in_channels, 每个通道独立)
	const unsigned dw_weight_stride = dw_kernel_size * dw_kernel_size;

	// 预计算pointwise权重步长 (1x1 conv): weight[oc * in_channels + ic]
	const unsigned pw_weight_oc_stride = in_channels;

#pragma omp parallel for schedule(guided)
	for(unsigned pos = 0; pos < num_positions; pos++) {
		const unsigned fh = pos / feat_w;
		const unsigned fw = pos % feat_w;

		// 为当前position分配独立的buf_dw
		float* buf_dw_pos = buf_dw + pos * in_channels;

		// Step 1: Depthwise Conv + ReLU
		// 对于每个输入通道，执行3x3卷积，结果存入buf_dw_pos
		for(unsigned ic = 0; ic < in_channels; ic++) {
			float sum = depthwise_layer->bias[ic];

			const int ih_base = (int)fh - dw_padding;
			const int iw_base = (int)fw - dw_padding;
			const unsigned weight_base = ic * dw_weight_stride;
			const unsigned input_base = ic * in_hw;

			for(unsigned kh = 0; kh < dw_kernel_size; kh++) {
				const int ih = ih_base + (int)kh;
				if(ih < 0 || ih >= (int)in_h) continue;

				const unsigned weight_kh_base = weight_base + kh * dw_kernel_size;
				const unsigned input_h_base = input_base + ih * in_w;

				for(unsigned kw = 0; kw < dw_kernel_size; kw++) {
					const int iw = iw_base + (int)kw;
					if(iw < 0 || iw >= (int)in_w) continue;

					sum += depthwise_layer->weights[weight_kh_base + kw] * in[input_h_base + iw];
				}
			}

			// ReLU激活
			buf_dw_pos[ic] = relu(sum);
		}

		// Step 2: Pointwise Conv (1x1) + 直接输出到展平格式
		for(unsigned oc = 0; oc < out_channels; oc++) {
			float sum = pointwise_layer->bias[oc];

			const unsigned weight_oc_base = oc * pw_weight_oc_stride;

			// 1x1卷积：对buf_dw_pos中的所有通道加权求和
			for(unsigned ic = 0; ic < in_channels; ic++) {
				sum += pointwise_layer->weights[weight_oc_base + ic] * buf_dw_pos[ic];
			}

			// 输出格式: [position][channel]
			out[pos * out_channels + oc] = sum;
		}
	}
}

// 融合循环: nn.Conv2d().permute(0,2,3,1).reshape(N, -1, C)
// 输入: CHW格式 (in_channels, in_h, in_w)
// 输出: HWC展平格式 (out_h*out_w, out_channels)，即 [oh*out_w + ow][oc]
conv2d_fun void conv2d_forward_permute_reshape(
    const convLayer_t layer, const float* in, unsigned in_h, unsigned in_w, float* out) {
	const unsigned out_channels = layer->out_channels;
	const unsigned in_channels = layer->in_channels;
	const unsigned kernel_size = layer->kernel_size;
	const unsigned stride = layer->stride;
	const unsigned padding = layer->padding;
	const unsigned groups = layer->groups;
	const unsigned in_channels_per_group = in_channels / groups;
	const unsigned out_channels_per_group = out_channels / groups;
	const unsigned kernel_area = kernel_size * kernel_size;
	const unsigned weight_ic_stride = kernel_area;
	const unsigned weight_oc_stride = in_channels * kernel_area;

	const unsigned out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
	const unsigned out_w = (in_w + 2 * padding - kernel_size) / stride + 1;
	const unsigned num_positions = out_h * out_w;
	const unsigned in_hw = in_h * in_w;

#pragma omp parallel for schedule(guided)
	for(unsigned pos = 0; pos < num_positions; pos++) {
		const unsigned oh = pos / out_w;
		const unsigned ow = pos % out_w;
		const int ih_base = (int)oh * stride - padding;
		const int iw_base = (int)ow * stride - padding;

		for(unsigned oc = 0; oc < out_channels; oc++) {
			float sum = layer->bias[oc];

			const unsigned group_idx = oc / out_channels_per_group;
			const unsigned ic_start = group_idx * in_channels_per_group;
			const unsigned ic_end = ic_start + in_channels_per_group;
			const unsigned weight_oc_base = oc * weight_oc_stride;

			for(unsigned ic = ic_start; ic < ic_end; ic++) {
				const unsigned weight_ic_base = weight_oc_base + ic * weight_ic_stride;
				const unsigned input_c_base = ic * in_hw;

				for(unsigned kh = 0; kh < kernel_size; kh++) {
					const int ih = ih_base + (int)kh;
					if(ih < 0 || ih >= (int)in_h) continue;

					const unsigned weight_kh_base = weight_ic_base + kh * kernel_size;
					const unsigned input_h_base = input_c_base + ih * in_w;

					for(unsigned kw = 0; kw < kernel_size; kw++) {
						const int iw = iw_base + (int)kw;
						if(iw < 0 || iw >= (int)in_w) continue;

						sum += layer->weights[weight_kh_base + kw] * in[input_h_base + iw];
					}
				}
			}

			// 输出格式: [position][channel] = [oh*out_w + ow][oc]
			out[pos * out_channels + oc] = sum;
		}
	}
}

// Backbone 网络前向传播
// 输入: 输入图像 (3, 240, 320) in ctx->buf_a
// 输出: backbone特征图 (64, 30, 40) in ctx->buf_b
static inline void backbone_forward(mb_tiny_context_t ctx) {
	// Layer 0: ConvReLU(3, 16, k=3, s=2, p=1, g=1) -> (16, 120, 160)
	conv2d_forward_relu(&ctx->layers[0], ctx->buf_a, H, W, ctx->buf_b);
	print_range("Layer0: ConvReLU(3,16,k=3,s=2)", ctx->buf_b, 16 * 120 * 160);

	// Layer 1: ConvReLU(16, 16, k=3, s=1, p=1, g=16) depthwise -> (16, 120, 160)
	conv2d_forward_relu(&ctx->layers[1], ctx->buf_b, 120, 160, ctx->buf_a);
	print_range("Layer1: DW(16,16,k=3,s=1)", ctx->buf_a, 16 * 120 * 160);

	// Layer 2: ConvReLU(16, 32, k=1, s=1, p=0, g=1) pointwise -> (32, 120, 160)
	conv1x1_relu_forward(&ctx->layers[2], ctx->buf_a, 120, 160, ctx->buf_b);
	print_range("Layer2: PW(16,32,k=1)", ctx->buf_b, 32 * 120 * 160);

	// Layer 3: ConvReLU(32, 32, k=3, s=2, p=1, g=32) depthwise -> (32, 60, 80)
	conv2d_forward_relu(&ctx->layers[3], ctx->buf_b, 120, 160, ctx->buf_a);
	print_range("Layer3: DW(32,32,k=3,s=2)", ctx->buf_a, 32 * 60 * 80);

	// Layer 4: ConvReLU(32, 32, k=1, s=1, p=0, g=1) pointwise -> (32, 60, 80)
	conv1x1_relu_forward(&ctx->layers[4], ctx->buf_a, 60, 80, ctx->buf_b);
	print_range("Layer4: PW(32,32,k=1)", ctx->buf_b, 32 * 60 * 80);

	// Layer 5: ConvReLU(32, 32, k=3, s=1, p=1, g=32) depthwise -> (32, 60, 80)
	conv2d_forward_relu(&ctx->layers[5], ctx->buf_b, 60, 80, ctx->buf_a);
	print_range("Layer5: DW(32,32,k=3,s=1)", ctx->buf_a, 32 * 60 * 80);

	// Layer 6: ConvReLU(32, 32, k=1, s=1, p=0, g=1) pointwise -> (32, 60, 80)
	conv1x1_relu_forward(&ctx->layers[6], ctx->buf_a, 60, 80, ctx->buf_b);
	print_range("Layer6: PW(32,32,k=1)", ctx->buf_b, 32 * 60 * 80);

	// Layer 7: ConvReLU(32, 32, k=3, s=2, p=1, g=32) depthwise -> (32, 30, 40)
	conv2d_forward_relu(&ctx->layers[7], ctx->buf_b, 60, 80, ctx->buf_a);
	print_range("Layer7: DW(32,32,k=3,s=2)", ctx->buf_a, 32 * 30 * 40);

	// Layer 8: ConvReLU(32, 64, k=1, s=1, p=0, g=1) pointwise -> (64, 30, 40)
	conv1x1_relu_forward(&ctx->layers[8], ctx->buf_a, 30, 40, ctx->buf_b);
	print_range("Layer8: PW(32,64,k=1)", ctx->buf_b, 64 * 30 * 40);

	// Layer 9: ConvReLU(64, 64, k=3, s=1, p=1, g=64) depthwise -> (64, 30, 40)
	conv2d_forward_relu(&ctx->layers[9], ctx->buf_b, 30, 40, ctx->buf_a);
	print_range("Layer9: DW(64,64,k=3,s=1)", ctx->buf_a, 64 * 30 * 40);

	// Layer 10: ConvReLU(64, 64, k=1, s=1, p=0, g=1) pointwise -> (64, 30, 40)
	conv1x1_relu_forward(&ctx->layers[10], ctx->buf_a, 30, 40, ctx->buf_b);
	print_range("Layer10: PW(64,64,k=1)", ctx->buf_b, 64 * 30 * 40);

	// Layer 11: ConvReLU(64, 64, k=3, s=1, p=1, g=64) depthwise -> (64, 30, 40)
	conv2d_forward_relu(&ctx->layers[11], ctx->buf_b, 30, 40, ctx->buf_a);
	print_range("Layer11: DW(64,64,k=3,s=1)", ctx->buf_a, 64 * 30 * 40);

	// Layer 12: ConvReLU(64, 64, k=1, s=1, p=0, g=1) pointwise -> (64, 30, 40)
	conv1x1_relu_forward(&ctx->layers[12], ctx->buf_a, 30, 40, ctx->buf_b);
	print_range("Layer12: PW(64,64,k=1)", ctx->buf_b, 64 * 30 * 40);

	// Layer 13: ConvReLU(64, 64, k=3, s=1, p=1, g=64) depthwise -> (64, 30, 40)
	conv2d_forward_relu(&ctx->layers[13], ctx->buf_b, 30, 40, ctx->buf_a);
	print_range("Layer13: DW(64,64,k=3,s=1)", ctx->buf_a, 64 * 30 * 40);

	// Layer 14: ConvReLU(64, 64, k=1, s=1, p=0, g=1) pointwise -> (64, 30, 40)
	conv1x1_relu_forward(&ctx->layers[14], ctx->buf_a, 30, 40, ctx->buf_b);
	print_range("Layer14: PW(64,64,k=1)", ctx->buf_b, 64 * 30 * 40);
}

// Stage 1: cls_head_1 + reg_head_1 + transition_1
// 输入: backbone输出 (64, 30, 40) in ctx->buf_b
// 输出:
//   - ctx->cls1[0:7200] (3600*2), ctx->reg1[0:14400] (3600*4)
//   - transition_1输出 (128, 15, 20) in ctx->buf_b
static inline void stage1_heads_transition(mb_tiny_context_t ctx) {
	// === Detection Heads Stage 1 ===
	// cls_head_1: DW(64,64,k=3) + PW(64,6,k=1) -> (3600, 2)
	detection_head_fused(&ctx->layers[15], &ctx->layers[16], ctx->buf_b, 30, 40, ctx->cls1, ctx->buf_a);

	// reg_head_1: DW(64,64,k=3) + PW(64,12,k=1) -> (3600, 4)
	detection_head_fused(&ctx->layers[17], &ctx->layers[18], ctx->buf_b, 30, 40, ctx->reg1, ctx->buf_a);

	// === Transition 1: (64, 30, 40) -> (128, 15, 20) ===
	// Layer 19: DW conv (64, 64, k=3, s=2, p=1) feat -> buf_a
	conv2d_forward_relu(&ctx->layers[19], ctx->buf_b, 30, 40, ctx->buf_a);

	// Layer 20: PW conv (64, 128, k=1) buf_a -> buf_b
	conv1x1_relu_forward(&ctx->layers[20], ctx->buf_a, 15, 20, ctx->buf_b);

	// Layer 21: DW conv (128, 128, k=3, s=1, p=1) buf_b -> buf_a
	conv2d_forward_relu(&ctx->layers[21], ctx->buf_b, 15, 20, ctx->buf_a);

	// Layer 22: PW conv (128, 128, k=1) buf_a -> buf_b
	conv1x1_relu_forward(&ctx->layers[22], ctx->buf_a, 15, 20, ctx->buf_b);

	// Layer 23: DW conv (128, 128, k=3, s=1, p=1) buf_b -> buf_a
	conv2d_forward_relu(&ctx->layers[23], ctx->buf_b, 15, 20, ctx->buf_a);

	// Layer 24: PW conv (128, 128, k=1) buf_a -> buf_b
	conv1x1_relu_forward(&ctx->layers[24], ctx->buf_a, 15, 20, ctx->buf_b);
}

// Stage 2: cls_head_2 + reg_head_2 + transition_2
// 输入: transition_1输出 (128, 15, 20) in ctx->buf_b
// 输出:
//   - ctx->cls_2_3_4[0] (600*2), ctx->reg_2_3_4[0] (600*4)
//   - transition_2输出 (256, 8, 10) in ctx->buf_b
static inline void stage2_heads_transition(mb_tiny_context_t ctx) {
	// === Detection Heads Stage 2 ===
	// cls_head_2: DW(128,128,k=3) + PW(128,4,k=1) -> (600, 2)
	detection_head_fused(&ctx->layers[25], &ctx->layers[26], ctx->buf_b, 15, 20, ctx->cls_2_3_4[0], ctx->buf_a);

	// reg_head_2: DW(128,128,k=3) + PW(128,8,k=1) -> (600, 4)
	detection_head_fused(&ctx->layers[27], &ctx->layers[28], ctx->buf_b, 15, 20, ctx->reg_2_3_4[0], ctx->buf_a);

	// === Transition 2: (128, 15, 20) -> (256, 8, 10) ===
	// Layer 29: DW conv (128, 128, k=3, s=2, p=1) buf_b -> buf_a
	conv2d_forward_relu(&ctx->layers[29], ctx->buf_b, 15, 20, ctx->buf_a);
	print_range("trans2_layer29", ctx->buf_a, 128 * 8 * 10);

	// Layer 30: PW conv (128, 256, k=1) buf_a -> buf_b
	conv1x1_relu_forward(&ctx->layers[30], ctx->buf_a, 8, 10, ctx->buf_b);
	print_range("trans2_layer30", ctx->buf_b, 256 * 8 * 10);

	// Layer 31: DW conv (256, 256, k=3, s=1, p=1) buf_b -> buf_a
	conv2d_forward_relu(&ctx->layers[31], ctx->buf_b, 8, 10, ctx->buf_a);
	print_range("trans2_layer31", ctx->buf_a, 256 * 8 * 10);

	// Layer 32: PW conv (256, 256, k=1) buf_a -> buf_b
	conv1x1_relu_forward(&ctx->layers[32], ctx->buf_a, 8, 10, ctx->buf_b);
	print_range("trans2_layer32", ctx->buf_b, 256 * 8 * 10);
}

// Stage 3: cls_head_3 + reg_head_3 + extra_layers
// 输入: transition_2输出 (256, 8, 10) in ctx->buf_b
// 输出:
//   - ctx->cls_2_3_4[1] (160*2), ctx->reg_2_3_4[1] (160*4)
//   - extra_layers输出 (256, 4, 5) in ctx->buf_a
static inline void stage3_heads_extra(mb_tiny_context_t ctx) {
	// === Detection Heads Stage 3 ===
	// cls_head_3: DW(256,256,k=3) + PW(256,4,k=1) -> (160, 2)
	detection_head_fused(&ctx->layers[33], &ctx->layers[34], ctx->buf_b, 8, 10, ctx->cls_2_3_4[1], ctx->buf_a);

	// reg_head_3: DW(256,256,k=3) + PW(256,8,k=1) -> (160, 4)
	detection_head_fused(&ctx->layers[35], &ctx->layers[36], ctx->buf_b, 8, 10, ctx->reg_2_3_4[1], ctx->buf_a);

	// === Extra Layers: (256, 8, 10) -> (256, 4, 5) ===
	// Layer 37: PW conv (256, 64, k=1) buf_b -> buf_a
	conv1x1_relu_forward(&ctx->layers[37], ctx->buf_b, 8, 10, ctx->buf_a);

	// Layer 38: DW conv (64, 64, k=3, s=2, p=1) buf_a -> buf_b
	conv2d_forward_relu(&ctx->layers[38], ctx->buf_a, 8, 10, ctx->buf_b);

	// Layer 39: PW conv (64, 256, k=1) buf_b -> buf_a
	conv1x1_relu_forward(&ctx->layers[39], ctx->buf_b, 4, 5, ctx->buf_a);
}

// Stage 4: cls_head_4 + reg_head_4
// 输入: extra_layers输出 (256, 4, 5) in ctx->buf_a
// 输出: ctx->cls_2_3_4[2] (60*2), ctx->reg_2_3_4[2] (60*4)
static inline void stage4_heads(mb_tiny_context_t ctx) {
	// cls_head_4: 普通卷积 (256, 6, k=3, s=1, p=1) -> (60, 2)
	conv2d_forward_permute_reshape(&ctx->layers[40], ctx->buf_a, 4, 5, ctx->cls_2_3_4[2]);

	// reg_head_4: 普通卷积 (256, 12, k=3, s=1, p=1) -> (60, 4)
	conv2d_forward_permute_reshape(&ctx->layers[41], ctx->buf_a, 4, 5, ctx->reg_2_3_4[2]);
}

// 生成先验框 (priors)
// 对应 Python: generate_priors(image_size=(320, 240), min_boxes=[[10,16,24],[32,48],[64,96],[128,192,256]], strides=[8,16,32,64])
static inline void generate_priors(mb_tiny_context_t ctx) {
	const unsigned num_strides = 4;
	const float strides[] = {8.0f, 16.0f, 32.0f, 64.0f};
	const unsigned num_min_boxes[] = {3, 2, 2, 3};
	const float min_boxes[][3] = {
	    {10.0f, 16.0f, 24.0f}, {32.0f, 48.0f, 0.0f}, {64.0f, 96.0f, 0.0f}, {128.0f, 192.0f, 256.0f}};

	unsigned prior_idx = 0;

	for(unsigned s = 0; s < num_strides; s++) {
		const float stride = strides[s];
		const float scale_w = W / stride;
		const float scale_h = H / stride;

		const unsigned fm_w = ceilf(W / stride);
		const unsigned fm_h = ceilf(H / stride);

		for(unsigned j = 0; j < fm_h; j++) {
			for(unsigned i = 0; i < fm_w; i++) {
				const float x_center = (i + 0.5f) / scale_w;
				const float y_center = (j + 0.5f) / scale_h;

				for(unsigned k = 0; k < num_min_boxes[s]; k++) {
					const float w = min_boxes[s][k] / W;
					const float h = min_boxes[s][k] / H;

					ctx->priors[prior_idx][0] = clipf(x_center, 0.0f, 1.0f);
					ctx->priors[prior_idx][1] = clipf(y_center, 0.0f, 1.0f);
					ctx->priors[prior_idx][2] = clipf(w, 0.0f, 1.0f);
					ctx->priors[prior_idx][3] = clipf(h, 0.0f, 1.0f);

					prior_idx++;
				}
			}
		}
	}
}

// Softmax 函数 (沿最后一个维度)
// 输入: scores [num_priors, num_classes]
// 输出: probabilities [num_priors, num_classes] (原地更新)
static inline void softmax_2d(float* scores, unsigned num_priors, unsigned num_classes) {
#pragma omp parallel for schedule(guided)
	for(unsigned i = 0; i < num_priors; i++) {
		float* row = &scores[i * num_classes];

		// 找到最大值用于数值稳定性
		float max_val = row[0];
		for(unsigned c = 1; c < num_classes; c++) {
			if(row[c] > max_val) max_val = row[c];
		}

		// 计算 exp 并求和
		float sum_exp = 0.0f;
		for(unsigned c = 0; c < num_classes; c++) {
			row[c] = expf(row[c] - max_val);
			sum_exp += row[c];
		}

		// 归一化
		const float inv_sum = 1.0f / (sum_exp + 1e-6f);
		for(unsigned c = 0; c < num_classes; c++) { row[c] *= inv_sum; }
	}
}

// 解码边界框并应用score_threshold 过滤
// 返回: 通过阈值的检测数量
static inline unsigned decode_boxes(const float* locations, const float (*priors)[4], float* cls_scores, detection_t detections,
    unsigned num_priors, float threshold_score, unsigned num_classes) {
	unsigned num_valid = 0;
	for(unsigned i = 0; i < num_priors; i++) {
		const float face_score = cls_scores[i * num_classes + 1];

		// 应用 score_threshold 过滤
		if(face_score < threshold_score) continue;

		const float* loc = &locations[i * 4];
		const float* prior = priors[i];

		// 解码中心点和宽高
		const float cx = loc[0] * CENTER_VARIANCE * prior[2] + prior[0];
		const float cy = loc[1] * CENTER_VARIANCE * prior[3] + prior[1];
		const float w = expf(loc[2] * SIZE_VARIANCE) * prior[2];
		const float h = expf(loc[3] * SIZE_VARIANCE) * prior[3];

		// 转换为 x1, y1, x2, y2 并裁剪到 [0, 1]
		detections[num_valid].x1 = clipf(cx - w / 2.0f, 0.0f, 1.0f);
		detections[num_valid].y1 = clipf(cy - h / 2.0f, 0.0f, 1.0f);
		detections[num_valid].x2 = clipf(cx + w / 2.0f, 0.0f, 1.0f);
		detections[num_valid].y2 = clipf(cy + h / 2.0f, 0.0f, 1.0f);
		detections[num_valid].score = face_score;

		num_valid++;
	}

	return num_valid;
}

// NMS 非极大值抑制
// 返回: 保留的检测数量
static inline unsigned standard_nms(
    detection_t detections, unsigned num_detections, float threshold_iou, unsigned k_top, unsigned* suppressed_buf) {
	if(num_detections == 0) return 0;

	// Step 1: 按分数降序排序
	// 简单选择排序：按分数降序排列
	for(unsigned i = 0; i < num_detections - 1; i++) {
		unsigned max_idx = i;
		for(unsigned j = i + 1; j < num_detections; j++) {
			if(detections[j].score > detections[max_idx].score) { max_idx = j; }
		}
		if(max_idx != i) {
			struct detection temp = detections[i];
			detections[i] = detections[max_idx];
			detections[max_idx] = temp;
		}
	}

	// Step 2: 初始化 suppressed 数组
	for(unsigned i = 0; i < num_detections; i++) { suppressed_buf[i] = 0; }

	unsigned num_kept = 0;

	for(unsigned i = 0; i < num_detections; i++) {
		if(suppressed_buf[i]) continue;

		// 检查是否达到 top_k
		if(k_top > 0 && num_kept >= k_top) break;

		// 保留当前检测框
		num_kept++;

		const float area_i = (detections[i].x2 - detections[i].x1) * (detections[i].y2 - detections[i].y1);

		// 计算与后续所有检测框的 IoU
		for(unsigned j = i + 1; j < num_detections; j++) {
			if(suppressed_buf[j]) continue;

			// 计算交集
			const float xx1 = fmaxf(detections[i].x1, detections[j].x1);
			const float yy1 = fmaxf(detections[i].y1, detections[j].y1);
			const float xx2 = fminf(detections[i].x2, detections[j].x2);
			const float yy2 = fminf(detections[i].y2, detections[j].y2);

			const float w = fmaxf(0.0f, xx2 - xx1);
			const float h = fmaxf(0.0f, yy2 - yy1);
			const float intersection = w * h;

			// 计算并集
			const float area_j = (detections[j].x2 - detections[j].x1) * (detections[j].y2 - detections[j].y1);
			const float union_area = area_i + area_j - intersection;

			// 计算 IoU
			const float iou = intersection / (union_area + 1e-6f);

			// 如果 IoU 超过阈值，抑制该检测框
			if(iou > threshold_iou) { suppressed_buf[j] = 1; }
		}
	}

	// Step 3: 压缩数组，将未被抑制的检测框移到前面
	unsigned write_idx = 0;
	for(unsigned i = 0; i < num_detections; i++) {
		if(!suppressed_buf[i]) {
			if(write_idx != i) { detections[write_idx] = detections[i]; }
			write_idx++;
		}
	}

	return write_idx;
}

void mb_tiny_init(mb_tiny_context_t ctx, const convLayer_t layers) {
	ctx->layers = layers;
	// 初始化指针数组
	ctx->cls_2_3_4[0] = &ctx->cls1[7200];  // stage2: 600*2
	ctx->cls_2_3_4[1] = &ctx->cls1[8400];  // stage3: 160*2
	ctx->cls_2_3_4[2] = &ctx->cls1[8720];  // stage4: 60*2
	ctx->reg_2_3_4[0] = &ctx->reg1[14400];	// stage2: 600*4
	ctx->reg_2_3_4[1] = &ctx->reg1[16800];	// stage3: 160*4
	ctx->reg_2_3_4[2] = &ctx->reg1[17440];	// stage4: 60*4
	// 生成 prior boxes
	generate_priors(ctx);
}

unsigned mb_tiny_detect(
    mb_tiny_context_t ctx, const unsigned char* input_rgb, unsigned width, unsigned height, detection_t detections) {
	if(!ctx || !input_rgb || !detections || width == 0 || height == 0) return 0;
	// Step 1: 图像预处理 (RGB -> normalized CHW)
	resize_and_normalize(input_rgb, width, height, ctx->buf_a);

	// Step 2: Backbone 特征提取
	backbone_forward(ctx);
	print_range("backbone (64x30x40)", ctx->buf_b, 64 * 30 * 40);

	// Step 3: Stage 1 - heads + transition_1
	stage1_heads_transition(ctx);
	print_range("cls1 stage1 (3600x2)", ctx->cls1, 3600 * 2);
	print_range("reg1 stage1 (3600x4)", ctx->reg1, 3600 * 4);
	print_range("feat1 (128x15x20)", ctx->buf_b, 128 * 15 * 20);

	// Step 4: Stage 2 - heads + transition_2
	stage2_heads_transition(ctx);
	print_range("cls2 (600x2)", &ctx->cls1[7200], 600 * 2);
	print_range("reg2 (600x4)", &ctx->reg1[14400], 600 * 4);
	print_range("feat2 (256x8x10)", ctx->buf_b, 256 * 8 * 10);

	// Step 5: Stage 3 - heads + extra_layers
	stage3_heads_extra(ctx);
	print_range("cls3 (160x2)", &ctx->cls1[8400], 160 * 2);
	print_range("reg3 (160x4)", &ctx->reg1[16800], 160 * 4);
	print_range("feat3 (256x4x5)", ctx->buf_a, 256 * 4 * 5);

	// Step 6: Stage 4 - heads
	stage4_heads(ctx);
	print_range("cls4 (60x2)", &ctx->cls1[8720], 60 * 2);
	print_range("reg4 (60x4)", &ctx->reg1[17440], 60 * 4);
	print_range("all_cls (4420x2)", ctx->cls1, 4420 * 2);
	print_range("all_reg (4420x4)", ctx->reg1, 4420 * 4);

	// Step 7: 对分类分数应用 softmax
	softmax_2d(ctx->cls1, 4420, NUM_CLASSES);

	// Step 8: 合并所有分类输出并应用 softmax + 解码边界框 + score_threshold 过滤
	// 需要将 cls1, cls_2_3_4[0], cls_2_3_4[1], cls_2_3_4[2] 合并为连续的内存
	// 由于它们已经在连续内存中 (cls1[0:8840])，直接传递即可
	unsigned num_detections = decode_boxes(ctx->reg1, ctx->priors, ctx->cls1, detections, 4420, score_threshold, NUM_CLASSES);

	if(num_detections == 0) return 0;

	// Step 9: NMS 非极大值抑制
	return standard_nms(detections, num_detections, iou_threshold, top_k, (unsigned*)ctx->buf_a);
}
// 在图像上绘制检测框
// 对应 Python: draw_boxes_on_image(image, boxes, scores, color=(0,255,0), thickness=2)
void mb_tiny_draw(unsigned char* image, unsigned width, unsigned height, const detection_t detections, unsigned num_detections,
    const unsigned char color[3], unsigned char thickness) {
	for(unsigned i = 0; i < num_detections; i++) {
		const detection_t det = &detections[i];

		// 将归一化坐标转换为像素坐标
		unsigned x1 = clipf(det->x1 * width, 0.0f, (float)(width - 1));
		unsigned y1 = clipf(det->y1 * height, 0.0f, (float)(height - 1));
		unsigned x2 = clipf(det->x2 * width, 0.0f, (float)(width - 1));
		unsigned y2 = clipf(det->y2 * height, 0.0f, (float)(height - 1));

		// 绘制边框（上下左右四条线）
		for(unsigned t = 0; t < thickness; t++) {
			// 上边
			if(y1 - t >= 0) {
				for(unsigned x = x1; x <= x2; x++) {
					image[HWC_INDEX(y1 - t, x, width)] = color[0];
					image[HWC_INDEX(y1 - t, x, width) + 1] = color[1];
					image[HWC_INDEX(y1 - t, x, width) + 2] = color[2];
				}
			}
			// 下边
			if(y2 + t < height) {
				for(unsigned x = x1; x <= x2; x++) {
					image[HWC_INDEX(y2 + t, x, width)] = color[0];
					image[HWC_INDEX(y2 + t, x, width) + 1] = color[1];
					image[HWC_INDEX(y2 + t, x, width) + 2] = color[2];
				}
			}
			// 左边
			if(x1 - t >= 0) {
				for(unsigned y = y1; y <= y2; y++) {
					image[HWC_INDEX(y, x1 - t, width)] = color[0];
					image[HWC_INDEX(y, x1 - t, width) + 1] = color[1];
					image[HWC_INDEX(y, x1 - t, width) + 2] = color[2];
				}
			}
			// 右边
			if(x2 + t < width) {
				for(unsigned y = y1; y <= y2; y++) {
					image[HWC_INDEX(y, x2 + t, width)] = color[0];
					image[HWC_INDEX(y, x2 + t, width) + 1] = color[1];
					image[HWC_INDEX(y, x2 + t, width) + 2] = color[2];
				}
			}
		}
	}
}
