#ifndef MATH_H
#define MATH_H
static inline float fmaxf(float a, float b) { return a > b ? a : b; }
static inline float fminf(float a, float b) { return a < b ? a : b; }
static inline float expf(float x) {
	float yf = 12102203 * x;
	int yi = (int)yf + 1064872507;
	return (*(float*)(&yi));
}
static inline float ceilf(float x) {
	int ix = (int)x;
	return (float)(ix >= x ? ix : ix + 1);
}
static inline float floorf(float x) {
	int ix = (int)x;
	return (float)(ix <= x ? ix : ix - 1);
}
#endif