#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mb_tiny.h"

#ifdef _OPENMP
#include <omp.h>
#else
#ifdef _WIN32
#include <windows.h>
double omp_get_wtime(void) {
	static LARGE_INTEGER frequency = {.QuadPart = 0};
	if(frequency.QuadPart == 0) { QueryPerformanceFrequency(&frequency); }
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	return now.QuadPart / (double)frequency.QuadPart;
}
#else
#include <sys/time.h>
double omp_get_wtime(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1000000.0;
}
#endif
#endif

extern const struct convLayer mb_tiny[42];

// 读取 PPM 文件 (P6 格式)
static unsigned char* read_ppm(const char* file_path, int* width, int* height) {
	FILE* fp = fopen(file_path, "rb");
	if(!fp) {
		fprintf(stderr, "Cannot open file: %s\n", file_path);
		return NULL;
	}

	// 读取 magic number
	char magic[3];
	fscanf(fp, "%2s", magic);
	if(*(short*)magic != 0x3650) {
		fprintf(stderr, "Unsupported PPM format: %s\n", magic);
		fclose(fp);
		return NULL;
	}

	// 跳过注释行
	int ch;
	while((ch = fgetc(fp)) == '#') { while((ch = fgetc(fp)) != '\n' && ch != EOF); }
	ungetc(ch, fp);

	// 读取宽度和高度
	fscanf(fp, "%d %d", width, height);

	// 读取最大值
	int max_val;
	fscanf(fp, "%d", &max_val);
	fgetc(fp);  // 跳过换行符

	// 分配图像缓冲区
	int num_pixels = (*width) * (*height);
	unsigned char* image = (unsigned char*)malloc(num_pixels * 3);
	if(!image) {
		fprintf(stderr, "Memory allocation failed\n");
		fclose(fp);
		return NULL;
	}

	// 读取像素数据
	size_t bytes_read = fread(image, 1, num_pixels * 3, fp);
	if(bytes_read != (size_t)(num_pixels * 3)) {
		fprintf(stderr, "Incomplete image data read\n");
		free(image);
		fclose(fp);
		return NULL;
	}

	fclose(fp);
	return image;
}

// 写入 PPM 文件 (P6 格式)
static int write_ppm(const char* file_path, const unsigned char* image, int width, int height) {
	FILE* fp = fopen(file_path, "wb");
	if(!fp) {
		fprintf(stderr, "Cannot create file: %s\n", file_path);
		return -1;
	}

	// 写入头部
	fprintf(fp, "P6\n%d %d\n255\n", width, height);

	// 写入像素数据
	size_t bytes_written = fwrite(image, 1, width * height * 3, fp);
	if(bytes_written != (size_t)(width * height * 3)) {
		fprintf(stderr, "Failed to write image data\n");
		fclose(fp);
		return -1;
	}

	fclose(fp);
	return 0;
}

int main(int argc, char* argv[]) {
	// 默认输入输出文件
	const char* input_file = "test.ppm";
	char output_file[512];

	// 如果提供了命令行参数，使用第一个参数作为输入文件
	if(argc > 1) { input_file = argv[1]; }

	// 生成输出文件名
	snprintf(output_file, sizeof(output_file), "%s", input_file);
	// 替换 .ppm 为 _result.ppm
	char* ext = strstr(output_file, ".ppm");
	if(ext) {
		*ext = '\0';
		strcat(output_file, "_result_c.ppm");
	} else {
		strcat(output_file, "_result_c.ppm");
	}

	printf("Loading image: %s\n", input_file);

	// Step 1: 读取 PPM 图像
	int width, height;
	unsigned char* image = read_ppm(input_file, &width, &height);
	if(!image) {
		fprintf(stderr, "Failed to read image\n");
		return 1;
	}
	printf("Image size: %dx%d\n", width, height);

	// Step 2: 初始化模型上下文
	static struct mb_tiny_context ctx_struct;
	mb_tiny_init(&ctx_struct, (const convLayer_t)mb_tiny);

	// Step 3: 分配检测结果数组（最多 4420 个检测框）
	static struct detection detections_array[4420];	 // 使用 static 避免栈溢出

	// Step 4: 运行人脸检测
	printf("Running face detection...\n");
	double start_time = omp_get_wtime();
	int num_detections = mb_tiny_detect(&ctx_struct, image, width, height, detections_array);
	printf("Detection took %.6f seconds\n", omp_get_wtime() - start_time);
	printf("Detected %d face(s)\n", num_detections);

	// Step 5: 打印检测结果
	for(int i = 0; i < num_detections; i++) {
		printf("  Face %d: score=%.4f, box=[%.4f, %.4f, %.4f, %.4f]\n", i + 1, detections_array[i].score,
		    detections_array[i].x1, detections_array[i].y1, detections_array[i].x2, detections_array[i].y2);
	}

	// Step 6: 在图像上绘制检测框
	if(num_detections > 0) {
		const unsigned char green_color[3] = {0, 255, 0};  // 绿色
		const unsigned char thickness = 2;
		mb_tiny_draw(image, width, height, detections_array, num_detections, green_color, thickness);

		// Step 7: 保存结果图像
		printf("Saving result to: %s\n", output_file);
		if(write_ppm(output_file, image, width, height) == 0) {
			printf("Result image saved successfully\n");
		} else {
			fprintf(stderr, "Failed to save result image\n");
		}
	} else {
		printf("No face detected\n");
	}

	// 清理资源
	free(image);

	return 0;
}
