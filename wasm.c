// python -m http.server
#define EXPORT_C(name) __attribute__((used, visibility("default"), export_name(#name))) name
#include "mb_tiny.h"
#define MAX_W 4096
#define MAX_H 4096
unsigned char image_buffer[MAX_W * MAX_H * 3];
extern const struct convLayer mb_tiny[42];
struct mb_tiny_context ctx_struct;
struct detection detections_array[4420];
void EXPORT_C(wasm_init)(){
	mb_tiny_init(&ctx_struct, (const convLayer_t)mb_tiny);
}
unsigned EXPORT_C(wasm_image_buffer)(){
	return (unsigned)image_buffer;
}
unsigned EXPORT_C(wasm_detect)(unsigned width, unsigned height){
	const unsigned char green_color[3] = {0, 255, 0};
	const unsigned char thickness = 2;
	unsigned num_detections = mb_tiny_detect(&ctx_struct, image_buffer, width, height, detections_array);
	mb_tiny_draw(image_buffer, width, height, detections_array, num_detections, green_color, thickness);
	return num_detections;
}