#pragma once

class cmdf
{
public:
	static int filter2(float * img_out, const float * img, int img_row, int img_col, float nv, float poster);
	static int filter2(float * y2, float * y1, const float * img, int img_row, int img_col, float nv);
	static int filter1(float * img_out, const float * img, int img_row, int img_col, float nv);
	static int runmrSPConf1(float * y2, float * y1, const float * img, int img_row, int img_col, float nv);
	static int runmrSPConf2(float * y1, const float * img, int img_row, int img_col, float nv);
	static int runmrSPConf3(float * y1, const float * img, int img_row, int img_col, float nv);
	static int runmrSPConf4(float * y2, float * y1, const float * img, int img_row, int img_col, float nv);
};
