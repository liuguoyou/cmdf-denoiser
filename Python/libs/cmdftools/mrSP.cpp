// mrSP.cpp : Defines the entry point for the console application.
//

#include <thread>
#include <algorithm>
#include "fft2d.h"
#include "mrSP.h"
#include <string.h>
#include <math.h>

static const int max_threads = 64;

static void DCT3X3row(
	float *dst,
	const float *src,
	float nv,	
	int img_row, int img_col)
{
	
	const float *p_col0 = src;
	const float *p_col1 = &src[img_row];
	const float *p_col2 = &src[img_row + img_row];
	float *p_o0 = dst;
	float *p_o1 = &dst[img_row];
	float *p_o2 = &dst[img_row + img_row];

	float nv_norm_6 = nv * 6.0f * 6.0f;
	float nv_norm_18 = nv * 18.0f * 6.0f;
	float nv_norm_36 = nv * 36.0f * 6.0f;
	float nv_norm_4 = nv * 4.0f * 6.0f;
	float nv_norm_12 = nv * 12.0f * 6.0f;
	
	
	int img_row_lim = img_row - 2;

	float v0 = *p_col0++;
	float v1 = *p_col1++;
	float v2 = *p_col2++;


	float C = (v0)+(v2);
	float dct_prv_2 = C + v1;
	float dct_prv_5 = v0 - v2;
	float dct_prv_8 = C - v1 - v1;

	v0 = *p_col0++;
	v1 = *p_col1++;
	v2 = *p_col2++;

	C = (v0)+(v2);
	float dct_prv_3 = C + v1;
	float dct_prv_6 = v0 - v2;
	float dct_prv_9 = C - v1 - v1;

	for (int l = 0;l<img_row_lim;l++) {

		float dct_prv_1 = dct_prv_2;dct_prv_2 = dct_prv_3;
		float dct_prv_4 = dct_prv_5;dct_prv_5 = dct_prv_6;
		float dct_prv_7 = dct_prv_8;dct_prv_8 = dct_prv_9;

		float v0 = *p_col0++;
		float v1 = *p_col1++;
		float v2 = *p_col2++;

		C = (v0)+(v2);
		dct_prv_3 = C + v1;
		dct_prv_6 = v0 - v2;
		dct_prv_9 = C - v1 - v1;

		C = dct_prv_1 + dct_prv_3;
		float dc = (C + dct_prv_2) / 9;
		float ot_2 = (dct_prv_1 - dct_prv_3);
		float ot_3 = (C - dct_prv_2 - dct_prv_2);

		C = dct_prv_4 + dct_prv_6;
		float ot_4 = (C + dct_prv_5);
		float ot_5 = (dct_prv_4 - dct_prv_6);
		float ot_6 = (C - dct_prv_5 - dct_prv_5);

		C = dct_prv_7 + dct_prv_9;
		float ot_7 = (C + dct_prv_8);
		float ot_8 = (dct_prv_7 - dct_prv_9);
		float ot_9 = (C - dct_prv_8 - dct_prv_8);

		ot_2 *= ((1.0f - expf(-ot_2*ot_2 / nv_norm_6)) / (6.0f));
		ot_3 *= ((1.0f - expf(-ot_3*ot_3 / nv_norm_18)) / (18.0f));
		ot_4 *= ((1.0f - expf(-ot_4*ot_4 / nv_norm_6)) / (6.0f));
		ot_5 *= ((1.0f - expf(-ot_5*ot_5 / nv_norm_4)) / (4.0f));
		ot_6 *= ((1.0f - expf(-ot_6*ot_6 / nv_norm_12)) / (12.0f));
		ot_7 *= ((1.0f - expf(-ot_7*ot_7 / nv_norm_18)) / (18.0f));
		ot_8 *= ((1.0f - expf(-ot_8*ot_8 / nv_norm_12)) / (12.0f));
		ot_9 *= ((1.0f - expf(-ot_9*ot_9 / nv_norm_36)) / (36.0f));

		C = dc + ot_7;
		float X1 = (C + ot_4);
		float X2 = (dc - ot_7 - ot_7);
		float X3 = (C - ot_4);

		C = ot_2 + ot_8;
		float Y1 = (C + ot_5);
		float Y2 = (ot_2 - ot_8 - ot_8);
		float Y3 = (C - ot_5);

		C = ot_3 + ot_9;
		float Z1 = (C + ot_6);
		float Z2 = (ot_3 - ot_9 - ot_9);
		float Z3 = (C - ot_6);

		float C1 = X1 + Z1;
		float C2 = X2 + Z2;
		float C3 = X3 + Z3;

		v0 = C1 + Y1;
		v1 = C2 + Y2;
		v2 = C3 + Y3;

		(*p_o0) += (v0 / 9);p_o0++;
		(*p_o1) += (v1 / 9);p_o1++;
		(*p_o2) += (v2 / 9);p_o2++;

		v0 = X1 - Z1 - Z1;
		v1 = X2 - Z2 - Z2;
		v2 = X3 - Z3 - Z3;

		(*p_o0) += (v0 / 9);p_o0++;
		(*p_o1) += (v1 / 9);p_o1++;
		(*p_o2) += (v2 / 9);p_o2++;

		v0 = C1 - Y1;
		v1 = C2 - Y2;
		v2 = C3 - Y3;

		(*p_o0) += (v0 / 9);
		(*p_o1) += (v1 / 9);
		(*p_o2) += (v2 / 9);

		p_o0--;
		p_o1--;
		p_o2--;
	}
}

static void DCT3X3jump(
	float *dst,
	const float *src,
	float nv,
	int img_row, int img_col, int offest, int jump) {
	
	int img_col_lim = img_col - 2;

	for (int x = offest;x < img_col_lim;x+= jump) {
		
		DCT3X3row(
			&dst[img_row*x],
			&src[img_row*x],
			nv,
			img_row, img_col);
	}
}

static void DCT3X3img(float *dst,
	const float *src,
	float nv,
	int img_row, int img_col, int th_num)
{

	memset(dst, 0, img_col*img_row*sizeof(float));
	std::thread t[max_threads];
	
	//max_th = std::min(max_th, max_threads);
	for (int offest = 0; offest < 3; ++offest) {
		for (int i = 0; i < th_num; ++i) {
			t[i] = std::thread(DCT3X3jump, dst, src, nv, img_row, img_col, i * 3 + offest, th_num * 3);
		}
		for (int i = 0; i < th_num; ++i) {
			t[i].join();
		}
	}
	
	int img_size = img_row*img_col;
	for (int l = 0;l<img_row;l++) {
		dst[l] *= 3;
	}
	for (int l = 0, idx = img_size - 1;l<img_row;l++, idx--) {
		dst[idx] *= 3;
	}

	for (int l = 0, idx = 0;l<img_col;l++, idx += img_row) {
		dst[idx] *= 3;
	}
	for (int l = 0, idx = img_row - 1;l<img_col;l++, idx += img_row) {
		dst[idx] *= 3;
	}

	int idx = img_row * 2;
	for (int l = img_row;l<idx;l++) {
		dst[l] *= 1.5;
	}

	idx = img_size - img_row;
	for (int l = img_size - img_row * 2;l<idx;l++) {
		dst[l] *= 1.5;
	}

	for (int l = 0, idx = 1;l<img_col;l++, idx += img_row) {
		dst[idx] *= 1.5;
	}
	for (int l = 0, idx = img_row - 2;l<img_col;l++, idx += img_row) {
		dst[idx] *= 1.5;
	}
}

static void SubBilateralRow(
	float *dst, const float *src, float nv,
	int img_rows, int img_cols, int rad, int x, const float *h)
{

	float Sp[3 * 3];
	int xm;
	dst = dst + x*img_rows;

	xm = (x - rad);	if (xm < 0) xm = x;int jx1 = xm*img_rows;
	xm = (x - 0);	int jx2 = xm*img_rows;
	xm = (x + rad);	if (xm >= img_cols) xm = x;int jx3 = xm*img_rows;

	for (int y = 0; y<img_rows; y++)
	{

		int ym1 = (y - rad);	if (ym1 < 0) ym1 = y;
		int ym2 = (y + rad);	if (ym2 >= img_rows) ym2 = y;

		Sp[0] = src[ym1 + jx1];
		Sp[1] = src[y + jx1];
		Sp[2] = src[ym2 + jx1];

		Sp[3] = src[ym1 + jx2];
		Sp[4] = src[y + jx2];
		Sp[5] = src[ym2 + jx2];

		Sp[6] = src[ym1 + jx3];
		Sp[7] = src[y + jx3];
		Sp[8] = src[ym2 + jx3];

		float mu = 0;
		float su = 0;
		for (int i = 0; i < 9; i++)
		{
			float dif = Sp[4] - Sp[i];
			float ratio = dif*dif / (nv);
			if (ratio < 5.0f)
			{
				float p = h[i] * expf(-ratio);
				mu += p;
				su += (p*Sp[i]);
			}
		}

		dst[y] = su / (mu);
	}
}

static void GenBilateralRow(
	float *dst, const float *src, float nv,
	int img_rows, int img_cols, const float *H, int x)
{
	//int img_rows_minus_one = img_rows - 1;
	//int img_cols_minus_one = img_cols - 1;
	float buf[5 * 5]; //max rad is 2
	//int xs = std::max(x - rad, 0);
	//int xe = std::min(x + rad, img_cols - 1);
	const int rad = 2;

	dst = dst + x*img_rows;
	const int br = (2 * rad + 1);

	for (int dx = -rad; dx <= rad; dx++)
	{		
		int xm = (dx + x);
		if (xm < 0) xm = -xm - 1;
		if (xm >= img_cols) xm = 2 * img_cols - xm - 1;

		int bm = (dx + rad)*br;
		for (int y = 0; y < (br - 1); y++)
		{
			int ym = y - rad;
			if (ym < 0) ym = -ym - 1;
			if (ym >= img_rows) ym = 2 * img_rows - ym - 1;
			buf[bm + y] = src[xm*img_rows + ym];
		}
	}

	const int bmc = rad*br + br - 1;
	const int center = (rad)*br + rad;
	int xm;
	xm = (x - 2);	if (xm < 0) xm = -xm - 1;int jx1 = xm*img_rows;
	xm = (x - 1);	if (xm < 0) xm = -xm - 1;int jx2 = xm*img_rows;
	xm = (x - 0);	int jx3 = xm*img_rows;
	xm = (x + 1);	if (xm >= img_cols) xm = 2 * img_cols - xm - 1;int jx4 = xm*img_rows;
	xm = (x + 2);	if (xm >= img_cols) xm = 2 * img_cols - xm - 1;int jx5 = xm*img_rows;

	for (int y = 0; y<img_rows; y++)
	{
		
		int ym = y + 2;	if (ym >= img_rows) ym = 2 * img_rows - ym - 1;
		buf[0 * br + 4] = src[jx1 + ym];
		buf[1 * br + 4] = src[jx2 + ym];
		buf[2 * br + 4] = src[jx3 + ym];
		buf[3 * br + 4] = src[jx4 + ym];
		buf[4 * br + 4] = src[jx5 + ym];

		float Sr = buf[center];

		float flt = 0;
		float psum = 0;

		for (int i = 0; i < 25; i++)
		{
			float Sp = buf[i];
			float delta = Sr - Sp;
			float e_power = delta*delta / nv;
			float p = H[i] * expf(-e_power);
			flt += p*Sp;
			psum += p;
		}

		for (int dx = 0; dx <br; dx++)
		{
			float *pbuf = buf + dx*br;
			pbuf[0] = pbuf[1];
			pbuf[1] = pbuf[2];
			pbuf[2] = pbuf[3];
			pbuf[3] = pbuf[4];
		}

		dst[y] = flt / psum;
	}
}

static void SteerBilateralRow(
	float *dst, const float *src, const float *src2, float nv,
	int img_rows, int img_cols, const float *H, int x)
{
	
	float buf[5 * 5]; //max rad is 2
	float buf2[5 * 5];
	const int rad = 2;

	dst = dst + x*img_rows;
	const int br = (2 * rad + 1);

	for (int dx = -rad; dx <= rad; dx++)
	{
		int xm = (dx + x);
		if (xm < 0) xm = -xm - 1;
		if (xm >= img_cols) xm = 2 * img_cols - xm - 1;

		int bm = (dx + rad)*br;
		for (int y = 0; y < (br - 1); y++)
		{
			int ym = y - rad;
			if (ym < 0) ym = -ym - 1;
			if (ym >= img_rows) ym = 2 * img_rows - ym - 1;
			buf[bm + y] = src[xm*img_rows + ym];
			buf2[bm + y] = src2[xm*img_rows + ym];
		}
	}

	const int bmc = rad*br + br - 1;
	const int center = (rad)*br + rad;
	int xm;
	xm = (x - 2);	if (xm < 0) xm = -xm - 1;int jx1 = xm*img_rows;
	xm = (x - 1);	if (xm < 0) xm = -xm - 1;int jx2 = xm*img_rows;
	xm = (x - 0);	int jx3 = xm*img_rows;
	xm = (x + 1);	if (xm >= img_cols) xm = 2 * img_cols - xm - 1;int jx4 = xm*img_rows;
	xm = (x + 2);	if (xm >= img_cols) xm = 2 * img_cols - xm - 1;int jx5 = xm*img_rows;

	for (int y = 0; y<img_rows; y++)
	{

		int ym = y + 2;	if (ym >= img_rows) ym = 2 * img_rows - ym - 1;
		buf[0 * br + 4] = src[jx1 + ym];
		buf[1 * br + 4] = src[jx2 + ym];
		buf[2 * br + 4] = src[jx3 + ym];
		buf[3 * br + 4] = src[jx4 + ym];
		buf[4 * br + 4] = src[jx5 + ym];

		buf2[0 * br + 4] = src2[jx1 + ym];
		buf2[1 * br + 4] = src2[jx2 + ym];
		buf2[2 * br + 4] = src2[jx3 + ym];
		buf2[3 * br + 4] = src2[jx4 + ym];
		buf2[4 * br + 4] = src2[jx5 + ym];

		float Sr = buf2[center];

		float flt = 0;
		float psum = 0;

		for (int i = 0; i < 25; i++)
		{
			float Sp = buf2[i];
			float delta = Sr - Sp;
			float e_power = delta*delta / nv;
			float p = H[i] * expf(-e_power);
			flt += p*buf[i];
			psum += p;
		}

		for (int dx = 0; dx <br; dx++)
		{
			float *pbuf = buf + dx*br;
			pbuf[0] = pbuf[1];
			pbuf[1] = pbuf[2];
			pbuf[2] = pbuf[3];
			pbuf[3] = pbuf[4];
			pbuf = buf2 + dx*br;
			pbuf[0] = pbuf[1];
			pbuf[1] = pbuf[2];
			pbuf[2] = pbuf[3];
			pbuf[3] = pbuf[4];
		}

		dst[y] = flt / psum;
	}
}

static void SubBilateraljump(
	float *dst,
	const float *src,
	float nv,
	int img_row, int img_col, int offest, int jump, int rad, const float *h) {

	for (int x = offest;x < img_col;x += jump) {

		SubBilateralRow(
			dst,
			src,
			nv,
			img_row, img_col, rad, x, h);
	}
}

static void GenBilateraljump(
	float *dst,
	const float *src,
	float nv,
	int img_row, int img_col, int offest, int jump, const float *H) {

	for (int x = offest;x < img_col;x += jump) {

		GenBilateralRow(
			dst,
			src,
			nv,
			img_row, img_col, H, x);
	}
}

static void SteerBilateraljump(
	float *dst,
	const float *src,
	const float *src2,
	float nv,
	int img_row, int img_col, int offest, int jump, const float *H) {

	for (int x = offest;x < img_col;x += jump) {

		SteerBilateralRow(
			dst,
			src,
			src2,
			nv,
			img_row, img_col, H, x);
	}
}

static void SubBilateral(float *dst,
	const float *src,
	float nv,
	int img_row, int img_col, int rad, float scale_HorzVert, float scale_Diagonal, int th_num)
{
	
	float h[3 * 3] = { scale_Diagonal, scale_HorzVert, scale_Diagonal,
	scale_HorzVert, 1,				scale_HorzVert,
	scale_Diagonal, scale_HorzVert, scale_Diagonal };

	std::thread t[max_threads];

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(SubBilateraljump, dst, src, nv, img_row, img_col, i , th_num, rad, h);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}	
}

static void GenBilateral(float *dst,
	const float *src,
	float nv,
	int img_row, int img_col, const float *H, int th_num)
{

	std::thread t[max_threads];

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(GenBilateraljump, dst, src, nv, img_row, img_col, i, th_num, H);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}
}

static void SteerBilateral(float *dst,
	const float *src, const float *src2,
	float nv,
	int img_row, int img_col, const float *H, int th_num)
{

	std::thread t[max_threads];

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(SteerBilateraljump, dst, src, src2, nv, img_row, img_col, i, th_num, H);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}
}

static void GetHalfBlock8(const float *img2D, float *block, int off_tmp)
{
	for (int k = 0; k<8; k++)
	{
		float *b = block + k;
		for (int m = 0; m<4; m++)
		{
			b[m * 8] = *img2D++;
		}
		img2D += off_tmp;
	}
}

static void GetHalfBlock16(const float *img2D, float *block, int off_tmp)
{
	for (int k = 0; k<16; k++)
	{
		float *b = block + k;
		for (int m = 0; m<8; m++)
		{
			b[m * 16] = *img2D++;
		}
		img2D += off_tmp;
	}
}

static void GetNLblockFirst(const float *img2D, float *block, int xc, int yc, int rows, int cols, int os)
{
	
	int xe = xc + 8 + os, ye = yc + os + 4;
	int xs = xc - os;
	for (int x = xs; x < xe; x++)
	{
		bool ccnd = false;
		if (x < 0 || x >= cols)
			ccnd = true;
		int xm = x*rows;		
		for (int y = yc - os; y < ye; y++)
		{
			if (y < 0 || y >= rows || ccnd)
				*block++ = 512;
			else
				*block++ = img2D[xm + y];
		}
		block += 4;
	}

}

static void GetNLblockLast(const float *img2D, float *block, int xc, int yc, int rows, int cols, int os)
{	

	int xe = xc + 8 + os, ye = yc + os + 8;
	int xs = xc - os; int ys = yc + os + 4;
	int jmp = 2 * os + 4;

	for (int x = xs; x < xe; x++)
	{
		bool ccnd = false;
		if (x < 0 || x >= cols)
			ccnd = true;
		int xm = x*rows;
		block += jmp;

		for (int y = ys; y < ye; y++)
		{
			if (y < 0 || y >= rows || ccnd)
				*block++ = 512;
			else
				*block++ = img2D[xm + y];
		}
	}

}

static void NLblockShift(float *block,int os)
{
	int sizex = 2 * os + 8;
	int sizey = sizex - 4;

	for (int x = sizex; x > 0; --x)
	{		
		for (int y = sizey; y > 0 ; --y)
		{
			*block = block[4];
			block++;
		}
		block += 4;
	}
}

static void Add2Dblk8(float *img2D, const float *block, int off_tmp)
{
	for (int k = 0; k<8; k++)
	{
		for (int m = 0; m<8; m++)
		{
			(*img2D++) += (*block++);
		}
		img2D += off_tmp;
	}
}

static void Add2Dblk16(float *img2D, const float *block, int off_tmp)
{
	for (int k = 0; k<16; k++)
	{
		for (int m = 0; m<16; m++)
		{
			(*img2D++) += (*block++);
		}
		img2D += off_tmp;
	}
}

static void RescaleOverlap(float *imgin, int img_col, int img_row, int bsize)
{
	
	int imsize = img_col*img_row;
	int strp = (bsize/2) * img_row;
	int endp = imsize - strp;

	for (int k = 0; k<strp; k++)
	{
		imgin[k] *= 2;
	}
	for (int k = endp; k<imsize; k++)
	{
		imgin[k] *= 2;
	}

	strp = 0; endp = img_row - 1;
	int bsize2 = bsize / 2;
	for (int k = 0; k<img_col; k++)
	{
		for (int l = 0; l<bsize2; l++)
		{
			imgin[strp + l] *= 2;
			imgin[endp - l] *= 2;
		}

		strp += img_row;
		endp += img_row;
	}
}

static void ShrinkCoef(float *fft_r, float *fft_i, float shrinkval, int n)
{
	// n for ifft and 4 for overlapping
	float scale = (float)n * 4;

	for (int k = 0; k<n; k++)
	{
		float ar = fft_r[k];
		float ai = fft_i[k];
		float shval = expf(-shrinkval / (ar*ar + ai*ai + 0.0001f)) / scale;
		fft_r[k] *= shval;
		fft_i[k] *= shval;
	}

}


void dft8shrinkrow(float *dst, const float *src, int img_row, int img_col, float nv, int col)
{

	fft2d m_fft2d;
	const int bsize = 8;
	const int bsize2 = bsize * bsize;
	const int HALF_BLK_SIZE = bsize / 2;
	const int hsize = bsize / 2;
	const int harea = (bsize2 / 2);
	int off_tmp, bound_col, bound_row, off_tmp_O;
	float block[bsize2], sfft_r[bsize2]
		, sfft_i[bsize2];
	float ffto_r_x[bsize2], ffto_i_x[bsize2];

	nv *= bsize2;

	off_tmp = img_row - hsize;
	bound_col = img_col - bsize;
	bound_row = img_row - bsize;
	off_tmp_O = img_row - bsize;
	
	if (col > bound_col)
		return; 

	
	const float *ginrblk = src + col*img_row;
	float *goutrblk = dst + col*img_row;
	GetHalfBlock8(ginrblk, block, off_tmp);

	m_fft2d.fft8HorzHalf(block, sfft_r, sfft_i);
	ginrblk += hsize;
	for (int j = 0; j <= bound_row; j += hsize)
	{
		GetHalfBlock8(ginrblk, block + harea, off_tmp);
		ginrblk += hsize;
		m_fft2d.fft8HorzHalf(block + harea, sfft_r + harea, sfft_i + harea);
		m_fft2d.fft8_2d_v(sfft_r, sfft_i, ffto_r_x, ffto_i_x);

		ShrinkCoef(ffto_r_x, ffto_i_x, nv, bsize2);

		m_fft2d.ifftx8_2D(ffto_r_x, ffto_i_x);

		Add2Dblk8(goutrblk, ffto_r_x, off_tmp_O);

		goutrblk += hsize;
		memcpy(sfft_r, sfft_r + harea, harea * sizeof(float));
		memcpy(sfft_i, sfft_i + harea, harea * sizeof(float));
		memcpy(block, block + (bsize2 / 2), (bsize2 / 2)*sizeof(float));
	}

}

void dft16shrinkrow(float *dst, const float *src, int img_row, int img_col, float nv, int col)
{

	fft2d m_fft2d;
	const int bsize = 16;
	const int bsize2 = bsize * bsize;
	const int HALF_BLK_SIZE = bsize / 2;
	const int hsize = bsize / 2;
	const int harea = (bsize2 / 2);
	int off_tmp, bound_col, bound_row, off_tmp_O;
	float block[bsize2], sfft_r[bsize2]
		, sfft_i[bsize2];
	float ffto_r_x[bsize2], ffto_i_x[bsize2];

	nv *= bsize2;

	off_tmp = img_row - hsize;
	bound_col = img_col - bsize;
	bound_row = img_row - bsize;
	off_tmp_O = img_row - bsize;
	
	if (col > bound_col)
		return;


	const float *ginrblk = src + col*img_row;
	float *goutrblk = dst + col*img_row;
	GetHalfBlock16(ginrblk, block, off_tmp);

	m_fft2d.fft16HorzHalf(block, sfft_r, sfft_i);
	ginrblk += hsize;
	for (int j = 0; j <= bound_row; j += hsize)
	{
		GetHalfBlock16(ginrblk, block + harea, off_tmp);
		ginrblk += hsize;
		m_fft2d.fft16HorzHalf(block + harea, sfft_r + harea, sfft_i + harea);
		m_fft2d.fft16_2d_v(sfft_r, sfft_i, ffto_r_x, ffto_i_x);

		ShrinkCoef(ffto_r_x, ffto_i_x, nv, bsize2);

		m_fft2d.ifftx16_2DNS(ffto_r_x, ffto_i_x);

		Add2Dblk16(goutrblk, ffto_r_x, off_tmp_O);

		goutrblk += hsize;
		memcpy(sfft_r, sfft_r + harea, harea * sizeof(float));
		memcpy(sfft_i, sfft_i + harea, harea * sizeof(float));
		memcpy(block, block + (bsize2 / 2), (bsize2 / 2)*sizeof(float));
	}

}

static void dft8shrinkjump(
	float *dst,
	const float *src,
	float nv,
	int img_row, int img_col, int offest, int jump) {

	const int bsize = 8;
	int img_col_lim = img_col - bsize;

	for (int x = offest;x <= img_col_lim;x += jump) {
		dft8shrinkrow(dst, src, img_row, img_col, nv, x);
	}
}

static void dft16shrinkjump(
	float *dst,
	const float *src,
	float nv,
	int img_row, int img_col, int offest, int jump) {

	const int bsize = 16;
	int img_col_lim = img_col - bsize;

	for (int x = offest;x <= img_col_lim;x += jump) {
		dft16shrinkrow(dst, src, img_row, img_col, nv, x);
	}
}

void stftshrink(float *dst, const float *src, int img_row, int img_col, float nv, int th_num, int bsize)
{

	std::thread t[max_threads];
	
	memset(dst, 0, img_col*img_row*sizeof(float));

	for (int i = 0; i < th_num; ++i) {
		if (bsize == 8)
			t[i] = std::thread(dft8shrinkjump, dst, src, nv, img_row, img_col, i*bsize, th_num*bsize);
		else if (bsize == 16)
			t[i] = std::thread(dft16shrinkjump, dst, src, nv, img_row, img_col, i*bsize, th_num*bsize);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}

	for (int i = 0; i < th_num; ++i) {
		if (bsize == 8)
			t[i] = std::thread(dft8shrinkjump, dst, src, nv, img_row, img_col, i*bsize + bsize / 2, th_num*bsize);
		else if (bsize == 16)
			t[i] = std::thread(dft16shrinkjump, dst, src, nv, img_row, img_col, i*bsize + bsize / 2, th_num*bsize);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}	
	
	RescaleOverlap(dst, img_col, img_row, bsize);
}


void bilinear2Row(float *dst, const float *src, int img_row, int img_col, int x)
{
	int imc = img_col / 2;
	int imr = img_row / 2;

	int imcl = imc - 1;
	int imrl = imr - 1;

	
	int ifx = x / 2;
	float dx = (x % 2) / 2.f;
	int x_ceil = std::min(ifx + (x % 2), imcl)*imr;
	int x_floor = ifx*imr;

	dst = dst + x*img_row;

	for (int y = 0; y <img_row; ++y)
	{
		int ify = y / 2;
		float dy = (y % 2) / 2.f;
		int y_ceil = std::min(ify + (y % 2), imrl);
		int y_floor = ify;

		float Q11 = src[x_floor + y_floor];
		float Q21 = src[x_ceil + y_floor];
		float Q12 = src[x_floor + y_ceil];
		float Q22 = src[x_ceil + y_ceil];

		float R1 = (1 - dx)*Q11 + dx*Q21;
		float R2 = (1 - dx)*Q12 + dx*Q22;

		float v = (1 - dy)*R1 + dy*R2;

		*dst++ = v;
	}
	
}

void bilinearadd2Row(float *dst, const float *src, int img_row, int img_col, int x)
{
	int imc = img_col / 2;
	int imr = img_row / 2;

	int imcl = imc - 1;
	int imrl = imr - 1;


	int ifx = x / 2;
	float dx = (x % 2) / 2.f;
	int x_ceil = std::min(ifx + (x % 2), imcl)*imr;
	int x_floor = ifx*imr;

	dst = dst + x*img_row;

	for (int y = 0; y <img_row; ++y)
	{
		int ify = y / 2;
		float dy = (y % 2) / 2.f;
		int y_ceil = std::min(ify + (y % 2), imrl);
		int y_floor = ify;

		float Q11 = src[x_floor + y_floor];
		float Q21 = src[x_ceil + y_floor];
		float Q12 = src[x_floor + y_ceil];
		float Q22 = src[x_ceil + y_ceil];

		float R1 = (1 - dx)*Q11 + dx*Q21;
		float R2 = (1 - dx)*Q12 + dx*Q22;

		float v = (1 - dy)*R1 + dy*R2;

		(*dst++) += v;
	}

}

void bilinearadd2RowClip(float *dst, const float *src, int img_row, int img_col, int x)
{
	int imc = img_col / 2;
	int imr = img_row / 2;

	int imcl = imc - 1;
	int imrl = imr - 1;


	int ifx = x / 2;
	float dx = (x % 2) / 2.f;
	int x_ceil = std::min(ifx + (x % 2), imcl)*imr;
	int x_floor = ifx*imr;

	dst = dst + x*img_row;

	for (int y = 0; y <img_row; ++y)
	{
		int ify = y / 2;
		float dy = (y % 2) / 2.f;
		int y_ceil = std::min(ify + (y % 2), imrl);
		int y_floor = ify;

		float Q11 = src[x_floor + y_floor];
		float Q21 = src[x_ceil + y_floor];
		float Q12 = src[x_floor + y_ceil];
		float Q22 = src[x_ceil + y_ceil];

		float R1 = (1 - dx)*Q11 + dx*Q21;
		float R2 = (1 - dx)*Q12 + dx*Q22;

		float v = (1 - dy)*R1 + dy*R2;

		v += (*dst);
		v = fminf(fmaxf(v, 0), 255);
		(*dst++) = v;
	}

}

static void bilinear2jump(
	float *dst,
	const float *src,
	int img_row, int img_col, int offest, int jump) {

	for (int x = offest;x < img_col;x += jump) {

		bilinear2Row(
			dst,
			src,
			img_row, img_col, x);
	}
}
static void bilinear2addjump(
	float *dst,
	const float *src,
	int img_row, int img_col, int offest, int jump) {

	for (int x = offest;x < img_col;x += jump) {

		bilinearadd2Row(
			dst,
			src,
			img_row, img_col, x);
	}
}
static void bilinear2addjumpClip(
	float *dst,
	const float *src,
	int img_row, int img_col, int offest, int jump) {

	for (int x = offest; x < img_col; x += jump) {

		bilinearadd2RowClip(
			dst,
			src,
			img_row, img_col, x);
	}
}

static void bilinear2(float *dst,
	const float *src,
	int img_row, int img_col, int th_num)
{

	std::thread t[max_threads];

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(bilinear2jump, dst, src, img_row, img_col, i, th_num);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}
}

static void bilinear2add(float *dst,
	const float *src,
	int img_row, int img_col, int th_num)
{

	std::thread t[max_threads];

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(bilinear2addjump, dst, src, img_row, img_col, i, th_num);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}
}
static void bilinear2addClip(float *dst,
	const float *src,
	int img_row, int img_col, int th_num)
{

	std::thread t[max_threads];

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(bilinear2addjumpClip, dst, src, img_row, img_col, i, th_num);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}
}
static void imfilter3Row(
	float *dst, const float *src, const float *h,
	int img_rows, int img_cols, int x)
{


	const int rad = 1;
	float buf[3 * 3];	

	dst = dst + x*img_rows;
	const int br = (2 * rad + 1);
	int img_rows_minus_one = img_rows - 1;
	for (int dx = -rad; dx <= rad; dx++)
	{
		int xm = (dx + x);
		if (xm < 0) xm = 0;
		if (xm >= img_cols) xm = img_cols - 1;

		int bm = (dx + rad)*br;
		for (int y = 0; y < (br - 1); y++)
		{
			int ym = y - rad;
			if (ym < 0) ym = 0;
			if (ym >= img_rows) ym = img_rows - 1;
			buf[bm + y] = src[xm*img_rows + ym];
		}
	}

	int xm, ym;
	xm = (x - rad);	if (xm < 0) xm = 0;	int jx1 = xm*img_rows;
	xm = (x - 0);	int jx2 = xm*img_rows;
	xm = (x + rad);	if (xm >= img_cols) xm = img_cols - 1;int jx3 = xm*img_rows;

	for (int y = 0; y<img_rows; ++y)
	{
		ym = y + 1;	if (ym >= img_rows) ym = img_rows - 1;

		buf[2] = src[jx1 + ym];
		buf[br + 2] = src[jx2 + ym];
		buf[2 * br + 2] = src[jx3 + ym];

		float flt = 0;
		flt += buf[0] * h[0];flt += buf[1] * h[1];flt += buf[2] * h[2];
		flt += buf[3] * h[3];flt += buf[4] * h[4];flt += buf[5] * h[5];
		flt += buf[6] * h[6];flt += buf[7] * h[7];flt += buf[8] * h[8];

		buf[0] = buf[1];
		buf[1] = buf[2];
		buf[0 + 3] = buf[1 + 3];
		buf[1 + 3] = buf[2 + 3];
		buf[0 + 6] = buf[1 + 6];
		buf[1 + 6] = buf[2 + 6];
			
		dst[y] = flt;
	}
}

static void deblock3Row(
	float *dst, const float *src, const float *steer, const float *h, float nv,
	int img_rows, int img_cols, int x)
{

	const int rad = 1;
	float buf[3 * 3];
	float bufst[3 * 3];

	dst = dst + x*img_rows;
	const int br = (2 * rad + 1);
	int img_rows_minus_one = img_rows - 1;
	for (int dx = -rad; dx <= rad; dx++)
	{
		int xm = (dx + x);
		if (xm < 0) xm = 0;
		if (xm >= img_cols) xm = img_cols - 1;

		int bm = (dx + rad)*br;
		for (int y = 0; y < (br - 1); y++)
		{
			int ym = y - rad;
			if (ym < 0) ym = 0;
			if (ym >= img_rows) ym = img_rows - 1;
			bufst[bm + y] = steer[xm*img_rows + ym];
			buf[bm + y] = src[xm*img_rows + ym];
		}
	}

	int xm, ym;
	xm = (x - rad);	if (xm < 0) xm = 0;	int jx1 = xm*img_rows;
	xm = (x - 0);	int jx2 = xm*img_rows;
	xm = (x + rad);	if (xm >= img_cols) xm = img_cols - 1;int jx3 = xm*img_rows;

	for (int y = 0; y<img_rows; ++y)
	{
		ym = y + 1;	if (ym >= img_rows) ym = img_rows - 1;

		
		bufst[2] = steer[jx1 + ym];
		bufst[br + 2] = steer[jx2 + ym];
		bufst[2 * br + 2] = steer[jx3 + ym];

		float flt = 0;
		flt += bufst[0];flt += bufst[1];flt += bufst[2];
		flt += bufst[3];flt += bufst[4];flt += bufst[5];
		flt += bufst[6];flt += bufst[7];flt += bufst[8];

		float flt2 = 0;
		flt2 += bufst[0] * bufst[0];flt2 += bufst[1] * bufst[1];flt2 += bufst[2] * bufst[2];
		flt2 += bufst[3] * bufst[3];flt2 += bufst[4] * bufst[4];flt2 += bufst[5] * bufst[5];
		flt2 += bufst[6] * bufst[6];flt2 += bufst[7] * bufst[7];flt2 += bufst[8] * bufst[8];

		bufst[0] = bufst[1];
		bufst[1] = bufst[2];
		bufst[0 + 3] = bufst[1 + 3];
		bufst[1 + 3] = bufst[2 + 3];
		bufst[0 + 6] = bufst[1 + 6];
		bufst[1 + 6] = bufst[2 + 6];

		float var3x3 = fmaxf(flt2 / 9 - flt*flt / 81, 0);
		float p = expf(-var3x3 / nv);

		/////////////////////

		buf[2] = src[jx1 + ym];
		buf[br + 2] = src[jx2 + ym];
		buf[2 * br + 2] = src[jx3 + ym];

		flt = 0;
		flt += buf[0] * h[0]; flt += buf[1] * h[1]; flt += buf[2] * h[2];
		flt += buf[3] * h[3]; flt += buf[4] * h[4]; flt += buf[5] * h[5];
		flt += buf[6] * h[6]; flt += buf[7] * h[7]; flt += buf[8] * h[8];

		dst[y] = buf[4] * (1 - p) + flt*p;

		buf[0] = buf[1];
		buf[1] = buf[2];
		buf[0 + 3] = buf[1 + 3];
		buf[1 + 3] = buf[2 + 3];
		buf[0 + 6] = buf[1 + 6];
		buf[1 + 6] = buf[2 + 6];

		/////////////////////		

		
	}
}

static void downconv2Row(
	float *dst, const float *src,
	int img_rows, int img_cols, int x)
{
	
	
	const int rad = 1;
	float buf[3 * 3];

	if ((x & 1) == 1)
		return;	
	
	dst = dst + (x / 2)*(img_rows / 2);
	const int br = (2 * rad + 1);
	//int img_rows_minus_one = img_rows - 1;
	for (int dx = -rad; dx <= rad; dx++)
	{
		int xm = (dx + x);
		if (xm < 0) xm = 0;
		if (xm >= img_cols) xm = img_cols-1;
	
		int bm = (dx + rad)*br;
		buf[bm] = src[xm*img_rows];
		
	}
	
	const int center = (rad)*br + rad + 1;
	int xm, ym1, ym2;
	xm = (x - rad);	if (xm < 0) xm = 0;	int jx1 = xm*img_rows;
	xm = (x - 0);	int jx2 = xm*img_rows;
	xm = (x + rad);	if (xm >= img_cols) xm = img_cols - 1;int jx3 = xm*img_rows;

	for (int y = 0; y<img_rows; y += 2)
	{
		ym1 = y;	if (ym1 >= img_rows) ym1 = img_rows - 1;
		ym2 = y + 1;	if (ym2 >= img_rows) ym2 = img_rows - 1;

		buf[1] = src[jx1 + ym1];		
		buf[2] = src[jx1 + ym2];

		buf[br + 1] = src[jx2 + ym1];
		buf[br + 2] = src[jx2 + ym2];
	
		buf[2 * br + 1] = src[jx3 + ym1];		
		buf[2 * br + 2] = src[jx3 + ym2];
			
		float flt = 0;
		flt += buf[0];flt += buf[1];flt += buf[2];
		flt += buf[3];flt += buf[4];flt += buf[5];
		flt += buf[6];flt += buf[7];flt += buf[8];

		buf[0] = buf[2];
		buf[0 + 3] = buf[2 + 3];
		buf[0 + 6] = buf[2 + 6];

		dst[y / 2] = flt / 9;
	}
}

static void downconv2jump(
	float *dst,
	const float *src,
	int img_row, int img_col, int offest, int jump) {

	for (int x = offest;x < img_col;x += jump) {

		downconv2Row(
			dst,
			src,
			img_row, img_col, x);
	}
}

static void imfilter3jump(
	float *dst,
	const float *src, const float *h,
	int img_row, int img_col, int offest, int jump) {

	for (int x = offest;x < img_col;x += jump) {

		imfilter3Row(
			dst,
			src, h,
			img_row, img_col, x);
	}
}

static void deblock3jump(
	float *dst, const float *src, const float *steer, const float *h, 
	float nv, int img_row, int img_col, int offest, int jump) {

	for (int x = offest;x < img_col;x += jump) {

		deblock3Row(
			dst, src, steer, h, nv,
			img_row, img_col, x);
	}
}

static void downconv2(float *dst,
	const float *src,
	int img_row, int img_col, int th_num)
{

	std::thread t[max_threads];

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(downconv2jump, dst, src, img_row, img_col, i, th_num);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}
}

static void imfilter3(float *dst,
	const float *src,
	const float *h,
	int img_row, int img_col, int th_num)
{

	std::thread t[max_threads];

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(imfilter3jump, dst, src, h, img_row, img_col, i, th_num);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}
}

static void deblock3(float *dst,
	const float *src, const float *steer, const float *h,
	float nv,
	int img_row, int img_col, int th_num)
{

	std::thread t[max_threads];

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(deblock3jump, dst, src, steer, h, nv, img_row, img_col, i, th_num);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}
}

static void upconvCombine(float *dst,
	const float *src, float *srcflt,
	int img_row, int img_col, int th_num)
{
		
	int imsize = (img_row / 2)*(img_col / 2);
	for (int k = 0; k < imsize; ++k) {
		srcflt[k] -= src[k];
	}
	bilinear2add(dst, srcflt, img_row, img_col, th_num);
}

// ***********************************************************
static void NLProcess(const float *lblock, float *out, int bsize, float sc1, float sc2)
{
	fft2d m_fft2d;
	int os = (bsize - 8) / 2;
	float diff[64], imag[64], diffcpy[64], sum[64];
	const float *lb = lblock + os*bsize + os;
	float *sb = out;
	int jump = (2 * os);
	for (int x = 0; x < 8; x++)
	{		
		for (int y = 0; y < 8; y++)
		{
			*sb++ = *lb++;
		}
		lb += jump;
	}
	
	int blm = bsize - 8;
	float smin[4] = { 64 * 1024 * 1024,64 * 1024 * 1024 ,64 * 1024 * 1024 ,64 * 1024 * 1024};
	int xmin[4], ymin[4];
	int lbj = bsize - 8;

	for (int i = 0; i < 64; i++)
		sum[i] = 0;

	for (int x = 0; x <= blm; x++)
	{
		for (int y = 0; y <= blm; y++)
		{
			if (x == os && y == os)
				continue;
			
			sb = out;
			lb = lblock + x*bsize + y;
			float a;
			float s = 0;
			for (int xt = 0; xt < 8; xt++)
			{				
				a = (*sb++) - (*lb++); s += a*a;
				a = (*sb++) - (*lb++); s += a*a;
				a = (*sb++) - (*lb++); s += a*a;
				a = (*sb++) - (*lb++); s += a*a;
				a = (*sb++) - (*lb++); s += a*a;
				a = (*sb++) - (*lb++); s += a*a;
				a = (*sb++) - (*lb++); s += a*a;
				a = (*sb++) - (*lb++); s += a*a;
				lb += lbj;
			}

			if (smin[0] > s) {
				xmin[3] = xmin[2]; ymin[3] = ymin[2]; smin[3] = smin[2];
				xmin[2] = xmin[1]; ymin[2] = ymin[1]; smin[2] = smin[1];
				xmin[1] = xmin[0]; ymin[1] = ymin[0]; smin[1] = smin[0];
				smin[0] = s;xmin[0] = x;ymin[0] = y;
			}
			else if (smin[1] > s) {
				xmin[3] = xmin[2]; ymin[3] = ymin[2]; smin[3] = smin[2];
				xmin[2] = xmin[1]; ymin[2] = ymin[1]; smin[2] = smin[1];				
				smin[1] = s; xmin[1] = x; ymin[1] = y;
			}
			else if (smin[2] > s) {
				xmin[3] = xmin[2]; ymin[3] = ymin[2]; smin[3] = smin[2];
				smin[2] = s; xmin[2] = x; ymin[2] = y;				
			}
			else if (smin[3] > s) {
				smin[3] = s; xmin[3] = x; ymin[3] = y;				
			}
		}		
	}
	float mean = 0;
	for (int k = 0; k < 4; k++) {
		lb = lblock + xmin[k] * bsize + ymin[k];
		float bd; 
		sb = out;
		float *d1 = diff, *d2 = diffcpy;
		for (int x = 0; x < 8; x++)
		{
			bd = (*lb++) - (*sb++); bd *= expf(-bd*bd / sc1); *d1++ = bd; *d2++ = bd;
			bd = (*lb++) - (*sb++); bd *= expf(-bd*bd / sc1); *d1++ = bd; *d2++ = bd;
			bd = (*lb++) - (*sb++); bd *= expf(-bd*bd / sc1); *d1++ = bd; *d2++ = bd;
			bd = (*lb++) - (*sb++); bd *= expf(-bd*bd / sc1); *d1++ = bd; *d2++ = bd;
			bd = (*lb++) - (*sb++); bd *= expf(-bd*bd / sc1); *d1++ = bd; *d2++ = bd;
			bd = (*lb++) - (*sb++); bd *= expf(-bd*bd / sc1); *d1++ = bd; *d2++ = bd;
			bd = (*lb++) - (*sb++); bd *= expf(-bd*bd / sc1); *d1++ = bd; *d2++ = bd;
			bd = (*lb++) - (*sb++); bd *= expf(-bd*bd / sc1); *d1++ = bd; *d2++ = bd;
			lb += lbj;
		}

		m_fft2d.fftx8_2D(diff, imag);

		for (int i = 0; i < 64; i++)
		{
			float dr = diff[i];
			float di = imag[i];

			float pwr_fft = dr*dr+ di*di;
			float prb_err = (1 - expf(-pwr_fft / sc2))/ 64;
			
			diff[i] *= prb_err;
			imag[i] *= prb_err;
		}

		m_fft2d.ifftx8_2D(diff, imag);
		for (int i = 0; i < 64; i++)
		{
			float a = diffcpy[i] - diff[i];
			sum[i] += a;
			mean += a;
		}

	}
	mean /= (64 * 5);
	for (int i = 0; i < 64; i++)
	{		
		out[i] += sum[i] / 5 - mean;
		out[i] /= 4;
	}

}

void nl8shrinkrow(float *dst, const float *src, int img_row, int img_col, float sc1, float sc2, int col)
{

	fft2d m_fft2d;
	const int bsize = 8;
	const int os = 4;
	const int lbsize = bsize + 2 * os;
	
	const int bsize2 = bsize * bsize;	
	const int hsize = bsize / 2;
		
	float lblock[lbsize*lbsize], proc[bsize2];
	
	int bound_col = img_col - bsize;
	int bound_row = img_row - bsize;
	
	float *goutrblk = dst + col*img_row;
	if (col > bound_col)
		return;

	GetNLblockFirst(src, lblock, col, 0, img_row, img_col,os);

	for (int y = 0; y <= bound_row; y += hsize)
	{
		GetNLblockLast(src, lblock, col, y, img_row, img_col,os);

		NLProcess(lblock, proc, lbsize, sc1, sc2);
		Add2Dblk8(goutrblk, proc, bound_row);
		goutrblk += hsize;
		NLblockShift(lblock,os);
	}

}

static void nl8shrinkjump(
	float *dst,
	const float *src,
	float sc1, float sc2,
	int img_row, int img_col, int offest, int jump) {

	const int bsize = 8;
	int img_col_lim = img_col - bsize;

	for (int x = offest; x <= img_col_lim; x += jump) {
		nl8shrinkrow(dst, src, img_row, img_col, sc1, sc2, x);
	}
}

void nlshrink(float *dst, const float *src, int img_row, int img_col, float sc1, float sc2, int th_num)
{
	const int bsize = 8;
	std::thread t[max_threads];

	memset(dst, 0, img_col*img_row*sizeof(float));

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(nl8shrinkjump, dst, src, sc1, sc2, img_row, img_col, i*bsize, th_num*bsize);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}

	for (int i = 0; i < th_num; ++i) {
		t[i] = std::thread(nl8shrinkjump, dst, src, sc1, sc2, img_row, img_col, i*bsize + bsize / 2, th_num*bsize);
	}
	for (int i = 0; i < th_num; ++i) {
		t[i].join();
	}

	RescaleOverlap(dst, img_col, img_row, bsize);
}

int cmdf::filter1(float * img_out, const float * img, int img_row, int img_col, float nv) {

	int th_num = std::thread::hardware_concurrency();
	th_num = std::min(th_num, max_threads);

	if (img_row % 8 != 0 || img_col % 8 != 0 || th_num < 1 || th_num>100) {
		return EXIT_FAILURE;
	}

	if (img_row > 3840 || img_col > 3840 || img_row < 64 || img_col < 64) {
		return EXIT_FAILURE;
	}

	nlshrink(img_out, img, img_row, img_col, nv * 40, nv * 4 * 64, 4);

	return EXIT_SUCCESS;
}
// deblock ==0 no deblock
// deblock == 1 first deblock
// deblock == 2 second deblock

int cmdf::filter2(float * y2, float * y1, const float * img, int img_row, int img_col, float nv) {

	const int deblock = 2;
	int th_num = std::thread::hardware_concurrency();
	th_num = std::min(th_num, max_threads);

	if (img_row % 8 != 0 || img_col % 8 != 0 || th_num < 1 || th_num>100) {
		return EXIT_FAILURE;
	}

	if (img_row > 3840 || img_col > 3840 || img_row < 64 || img_col < 64) {
		return EXIT_FAILURE;
	}

	float * dummy = new float[img_row*img_col];
	float * dummy2 = new float[img_row*img_col];
	
	float * app = new float[img_row*img_col];
	float * imgd1 = new float[(img_row / 2)*(img_col / 2)];
	float * imgd2 = new float[(img_row / 4)*(img_col / 4)];

	float * detail = y2;

	const float H[25] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };

	downconv2(imgd1,
		img,
		img_row, img_col, th_num);

	downconv2(imgd2,
		imgd1,
		img_row / 2, img_col / 2, th_num);

	GenBilateral(dummy,
		imgd2,
		2 * nv,
		img_row / 4, img_col / 4, H, th_num);

	upconvCombine(imgd1,
		imgd2, dummy,
		img_row / 2, img_col / 2, th_num);

	GenBilateral(dummy,
		imgd1,
		8 * nv,
		img_row / 2, img_col / 2, H, th_num);

	bilinear2(app, dummy, img_row, img_col, th_num);


	int imsize = img_row*img_col;
	for (int k = 0; k < imsize; ++k) {
		detail[k] = img[k] - app[k];
	}



	stftshrink(dummy, detail, img_row, img_col, nv / 2.5f, th_num, 16);
	stftshrink(detail, dummy, img_row, img_col, nv * 4 / 9, th_num, 8);



	for (int k = 0; k < imsize; ++k) {
		y1[k] = app[k] + detail[k];
	}

	float nv_mod = nv / 3.3f;

	DCT3X3img(dummy, y1, nv_mod * 2, img_row, img_col, th_num);

	SubBilateral(y2, dummy, nv_mod, img_row, img_col, 2, 0.60653f, 0.3678f, th_num);
	SubBilateral(dummy, y2, nv_mod / 2, img_row, img_col, 3, 0.5134f, 0.2636f, th_num);
	SubBilateral(y2, dummy, nv_mod * 2 / 9, img_row, img_col, 4, 0.4029f, 0.1623f, th_num);



	for (int k = 0; k < imsize; ++k) {		
		dummy[k] = img[k] - y2[k];
	}

	float H2[25];
	double h_sig = 3;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y)
			H2[x * 5 + y + 12] = (float)expf(-(x*x + y*y) / (2 * h_sig *h_sig));


	SteerBilateral(dummy2, dummy, y1, nv_mod*3.5f, img_row, img_col, H2, th_num);

	stftshrink(dummy, dummy2, img_row, img_col, nv_mod * 2.75f, th_num, 16);

	for (int k = 0; k < imsize; ++k) {
		y2[k] += dummy[k];
	}

	if (deblock == 0) {
		return EXIT_SUCCESS;
	}
	/////////////// Fill H3
	float H3[9];
	h_sig = 0.72;
	double sum = 0;
	for (int x = -1; x <= 1; ++x) {
		for (int y = -1; y <= 1; ++y) {
			H3[x * 3 + y + 4] = (float)expf(-(x*x + y*y) / (h_sig));
			sum += H3[x * 3 + y + 4];
		}
	}
	for (int x = 0; x < 9; ++x)
		H3[x] /= (float)sum;
	///////////////

	deblock3(dummy2, y2, y1, H3, nv_mod, img_row, img_col, th_num);

	memcpy(y2, dummy2, img_row*img_col*sizeof(float));

	if (deblock == 1) {
		return EXIT_SUCCESS;
	}

	downconv2(imgd1,
		y2,
		img_row, img_col, th_num);

	downconv2(dummy,
		y1,
		img_row, img_col, th_num);

	deblock3(dummy2, imgd1, dummy, H3, nv_mod / 5, img_row / 2, img_col / 2, th_num);
	int imsize2 = (img_row / 2)*(img_col / 2);
	for (int k = 0; k < imsize2; ++k) {
		dummy2[k] -= imgd1[k];
	}


	bilinear2addClip(y2, dummy2, img_row, img_col, th_num);

	delete[] dummy;
	delete[] dummy2;
	delete[] app;
	delete[] imgd1;
	delete[] imgd2;

	return EXIT_SUCCESS;
}

int cmdf::filter2(float * img_out, const float * img, int img_row, int img_col, float nv, float poster) {

	int th_num = std::thread::hardware_concurrency();
	th_num = std::min(th_num, max_threads);

	if (img_row % 8 != 0 || img_col % 8 != 0 || poster > 1 || poster < 0 || th_num < 1 || th_num>100) {
		return EXIT_FAILURE;
	}

	if (img_row > 3840 || img_col > 3840 || img_row < 64 || img_col < 64) {
		return EXIT_FAILURE;
	}

	float * dummy = new float[img_row*img_col];
	float * dummy2 = new float[img_row*img_col];

	float * level1 = new float[img_row*img_col];
	float * app = new float[img_row*img_col];
	float * imgd1 = new float[(img_row / 2)*(img_col / 2)];
	float * imgd2 = new float[(img_row / 4)*(img_col / 4)];

	float * detail = img_out;




	const float H[25] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };

	downconv2(imgd1,
		img,
		img_row, img_col, th_num);

	downconv2(imgd2,
		imgd1,
		img_row / 2, img_col / 2, th_num);

	GenBilateral(dummy,
		imgd2,
		2 * nv,
		img_row / 4, img_col / 4, H, th_num);

	upconvCombine(imgd1,
		imgd2, dummy,
		img_row / 2, img_col / 2, th_num);

	GenBilateral(dummy,
		imgd1,
		8 * nv,
		img_row / 2, img_col / 2, H, th_num);

	bilinear2(app, dummy, img_row, img_col, th_num);


	int imsize = img_row*img_col;
	for (int k = 0; k < imsize; ++k) {
		detail[k] = img[k] - app[k];
	}



	stftshrink(dummy, detail, img_row, img_col, nv / 2.5f, th_num, 16);
	stftshrink(detail, dummy, img_row, img_col, nv * 4 / 9, th_num, 8);



	for (int k = 0; k < imsize; ++k) {
		level1[k] = app[k] + detail[k];
		detail[k] += (app[k] * poster);
	}

	float nv_mod = nv / 3.3f;

	DCT3X3img(dummy, detail, nv_mod * 2, img_row, img_col, th_num);

	SubBilateral(detail, dummy, nv_mod, img_row, img_col, 2, 0.60653f, 0.3678f, th_num);
	SubBilateral(dummy, detail, nv_mod / 2, img_row, img_col, 3, 0.5134f, 0.2636f, th_num);
	SubBilateral(detail, dummy, nv_mod * 2 / 9, img_row, img_col, 4, 0.4029f, 0.1623f, th_num);



	for (int k = 0; k < imsize; ++k) {
		float t1 = app[k];
		detail[k] -= (t1 * poster);
		dummy[k] = img[k] - (t1 + detail[k]);
	}

	float H2[25];
	double h_sig = 3;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y)
			H2[x * 5 + y + 12] = (float)expf(-(x*x + y*y) / (2 * h_sig *h_sig));


	SteerBilateral(dummy2, dummy, level1, nv_mod*3.5f, img_row, img_col, H2, th_num);

	stftshrink(dummy, dummy2, img_row, img_col, nv_mod * 2.75f, th_num, 16);

	for (int k = 0; k < imsize; ++k) {
		detail[k] += (dummy[k] + app[k]);
	}

	float H3[9];
	h_sig = 0.72;
	double sum = 0;
	for (int x = -1; x <= 1; ++x) {
		for (int y = -1; y <= 1; ++y) {
			H3[x * 3 + y + 4] = (float)expf(-(x*x + y*y) / (h_sig));
			sum += H3[x * 3 + y + 4];
		}
	}
	for (int x = 0; x < 9; ++x)
		H3[x] /= (float)sum;

	deblock3(dummy2, detail, level1, H3, nv_mod, img_row, img_col, th_num);

	memcpy(detail, dummy2, img_row*img_col*sizeof(float));

	downconv2(imgd1,
		detail,
		img_row, img_col, th_num);

	downconv2(dummy,
		level1,
		img_row, img_col, th_num);

	deblock3(dummy2, imgd1, dummy, H3, nv_mod / 5, img_row / 2, img_col / 2, th_num);

	bilinear2addClip(detail, dummy2, img_row, img_col, th_num);

	delete[] dummy;
	delete[] dummy2;
	delete[] level1;
	delete[] app;
	delete[] imgd1;
	delete[] imgd2;

	return EXIT_SUCCESS;
}



int cmdf::runmrSPConf1(float * y2, float * y1, const float * img, int img_row, int img_col, float nv) {

	int th_num = std::thread::hardware_concurrency();
	th_num = std::min(th_num, max_threads);

	if (img_row % 8 != 0 || img_col % 8 != 0 || th_num < 1 || th_num>100) {
		return EXIT_FAILURE;
	}

	if (img_row > 3840 || img_col > 3840 || img_row < 64 || img_col < 64) {
		return EXIT_FAILURE;
	}

	float * dummy = new float[img_row*img_col];
	float * dummy2 = new float[img_row*img_col];

	float * imgd1 = new float[(img_row / 2)*(img_col / 2)];
	float * imgd2 = new float[(img_row / 4)*(img_col / 4)];

	float * detail = y2;

	const float H[25] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };

	downconv2(imgd1,
		img,
		img_row, img_col, th_num);

	downconv2(imgd2,
		imgd1,
		img_row / 2, img_col / 2, th_num);

	GenBilateral(dummy,
		imgd2,
		2 * nv,
		img_row / 4, img_col / 4, H, th_num);

	upconvCombine(imgd1,
		imgd2, dummy,
		img_row / 2, img_col / 2, th_num);

	GenBilateral(dummy,
		imgd1,
		8 * nv,
		img_row / 2, img_col / 2, H, th_num);

	bilinear2(y1, dummy, img_row, img_col, th_num);


	int imsize = img_row*img_col;
	for (int k = 0; k < imsize; ++k) {
		detail[k] = img[k] - y1[k];
	}



	stftshrink(dummy, detail, img_row, img_col, nv / 5.0f, th_num, 16);
	stftshrink(detail, dummy, img_row, img_col, nv * 4.0f / 27.0f, th_num, 8);



	for (int k = 0; k < imsize; ++k) {
		y1[k] += detail[k];
	}

	float nv_mod = nv / 3.3f;

	DCT3X3img(y2, y1, nv_mod * 1.75f, img_row, img_col, th_num);	

	
	/////////////// Fill H3
	float H3[9];
	double h_sig = 0.72;
	double sum = 0;
	for (int x = -1; x <= 1; ++x) {
		for (int y = -1; y <= 1; ++y) {
			H3[x * 3 + y + 4] = (float)expf(-(x*x + y*y) / (h_sig));
			sum += H3[x * 3 + y + 4];
		}
	}
	for (int x = 0; x < 9; ++x)
		H3[x] /= (float)sum;
	///////////////

	deblock3(dummy2, y2, y1, H3, nv_mod, img_row, img_col, th_num);

	memcpy(y2, dummy2, img_row*img_col*sizeof(float));	

	downconv2(imgd1,
		y2,
		img_row, img_col, th_num);

	downconv2(dummy,
		y1,
		img_row, img_col, th_num);

	deblock3(dummy2, imgd1, dummy, H3, nv_mod / 5, img_row / 2, img_col / 2, th_num);
	int imsize2 = (img_row / 2)*(img_col / 2);
	for (int k = 0; k < imsize2; ++k) {
		dummy2[k] -= imgd1[k];
	}

	bilinear2addClip(y2, dummy2, img_row, img_col, th_num);

	delete[] dummy;
	delete[] dummy2;	
	delete[] imgd1;
	delete[] imgd2;

	return EXIT_SUCCESS;
}


int cmdf::runmrSPConf2(float * y2, const float * img, int img_row, int img_col, float nv) {

	int th_num = std::thread::hardware_concurrency();
	th_num = std::min(th_num, max_threads);

	if (img_row % 8 != 0 || img_col % 8 != 0 || th_num < 1 || th_num>100) {
		return EXIT_FAILURE;
	}

	if (img_row > 3840 || img_col > 3840 || img_row < 64 || img_col < 64) {
		return EXIT_FAILURE;
	}

	float * dummy = new float[img_row*img_col];
	float * dummy2 = new float[img_row*img_col];

	float * imgd1 = new float[(img_row / 2)*(img_col / 2)];
	float * imgd2 = new float[(img_row / 4)*(img_col / 4)];

	//float * detail = y2;

	float nv_mod = nv;

	DCT3X3img(dummy, img, nv_mod * 2, img_row, img_col, th_num);

	SubBilateral(y2, dummy, nv_mod, img_row, img_col, 2, 0.60653f, 0.3678f, th_num);
	SubBilateral(dummy, y2, nv_mod / 2, img_row, img_col, 3, 0.5134f, 0.2636f, th_num);
	SubBilateral(y2, dummy, nv_mod * 2 / 9, img_row, img_col, 4, 0.4029f, 0.1623f, th_num);


	int imsize = img_row*img_col;
	for (int k = 0; k < imsize; ++k) {
		dummy[k] = img[k] - y2[k];
	}

	float H2[25];
	double h_sig = 3;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y)
			H2[x * 5 + y + 12] = (float)expf(-(x*x + y*y) / (2 * h_sig *h_sig));


	SteerBilateral(dummy2, dummy, img, nv_mod*3.5f, img_row, img_col, H2, th_num);

	stftshrink(dummy, dummy2, img_row, img_col, nv_mod * 2.75f, th_num, 16);

	for (int k = 0; k < imsize; ++k) {
		y2[k] += dummy[k];
	}

	
	/////////////// Fill H3
	float H3[9];
	h_sig = 0.72;
	double sum = 0;
	for (int x = -1; x <= 1; ++x) {
		for (int y = -1; y <= 1; ++y) {
			H3[x * 3 + y + 4] = (float)expf(-(x*x + y*y) / (h_sig));
			sum += H3[x * 3 + y + 4];
		}
	}
	for (int x = 0; x < 9; ++x)
		H3[x] /= (float)sum;
	///////////////

	deblock3(dummy2, y2, img, H3, nv_mod * 2, img_row, img_col, th_num);

	memcpy(y2, dummy2, img_row*img_col*sizeof(float));
	

	downconv2(imgd1,
		y2,
		img_row, img_col, th_num);

	downconv2(dummy,
		img,
		img_row, img_col, th_num);

	deblock3(dummy2, imgd1, dummy, H3, nv_mod / 5, img_row / 2, img_col / 2, th_num);
	int imsize2 = (img_row / 2)*(img_col / 2);
	for (int k = 0; k < imsize2; ++k) {
		dummy2[k] -= imgd1[k];
	}


	bilinear2addClip(y2, dummy2, img_row, img_col, th_num);

	delete[] dummy;
	delete[] dummy2;
	delete[] imgd1;
	delete[] imgd2;

	return EXIT_SUCCESS;
}


int cmdf::runmrSPConf3(float * y1, const float * img, int img_row, int img_col, float nv) {

	int th_num = std::thread::hardware_concurrency();
	th_num = std::min(th_num, max_threads);

	if (img_row % 8 != 0 || img_col % 8 != 0 || th_num < 1 || th_num>100) {
		return EXIT_FAILURE;
	}

	if (img_row > 3840 || img_col > 3840 || img_row < 64 || img_col < 64) {
		return EXIT_FAILURE;
	}

	float * dummy = new float[img_row*img_col];
	float * dummy2 = new float[img_row*img_col];	

	//float * detail = y2;

	float nv_mod = nv;

	DCT3X3img(dummy, img, nv_mod * 2, img_row, img_col, th_num);

	SubBilateral(y1, dummy, nv_mod, img_row, img_col, 2, 0.60653f, 0.3678f, th_num);
	SubBilateral(dummy, y1, nv_mod / 2, img_row, img_col, 3, 0.5134f, 0.2636f, th_num);
	SubBilateral(y1, dummy, nv_mod * 2 / 9, img_row, img_col, 4, 0.4029f, 0.1623f, th_num);


	int imsize = img_row*img_col;
	for (int k = 0; k < imsize; ++k) {
		dummy[k] = img[k] - y1[k];
	}

	float H2[25];
	double h_sig = 3;
	for (int x = -2; x <= 2; ++x)
		for (int y = -2; y <= 2; ++y)
			H2[x * 5 + y + 12] = (float)expf(-(x*x + y*y) / (2 * h_sig *h_sig));


	SteerBilateral(dummy2, dummy, img, nv_mod *1.8f, img_row, img_col, H2, th_num);

	stftshrink(dummy, dummy2, img_row, img_col, nv_mod * 1.3f, th_num, 16);

	for (int k = 0; k < imsize; ++k) {
		y1[k] += dummy[k];
	}



	delete[] dummy;
	delete[] dummy2;	

	return EXIT_SUCCESS;
}


int cmdf::runmrSPConf4(float * y2, float * y1, const float * img, int img_row, int img_col, float nv) {

	int th_num = std::thread::hardware_concurrency();
	th_num = std::min(th_num, max_threads);

	if (img_row % 8 != 0 || img_col % 8 != 0 || th_num < 1 || th_num>100) {
		return EXIT_FAILURE;
	}

	if (img_row > 3840 || img_col > 3840 || img_row < 64 || img_col < 64) {
		return EXIT_FAILURE;
	}

	float * dummy = new float[img_row*img_col];
	float * imgd1 = new float[(img_row / 2)*(img_col / 2)];
	float * imgd2 = new float[(img_row / 4)*(img_col / 4)];

	float * detail = y2;

	const float H[25] = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };

	downconv2(imgd1,
		img,
		img_row, img_col, th_num);

	downconv2(imgd2,
		imgd1,
		img_row / 2, img_col / 2, th_num);

	GenBilateral(dummy,
		imgd2,
		2 * nv,
		img_row / 4, img_col / 4, H, th_num);

	upconvCombine(imgd1,
		imgd2, dummy,
		img_row / 2, img_col / 2, th_num);

	GenBilateral(dummy,
		imgd1,
		8 * nv,
		img_row / 2, img_col / 2, H, th_num);

	bilinear2(y1, dummy, img_row, img_col, th_num);


	int imsize = img_row*img_col;
	for (int k = 0; k < imsize; ++k) {
		detail[k] = img[k] - y1[k];
	}



	stftshrink(dummy, detail, img_row, img_col, nv / 2.5f, th_num, 16);
	stftshrink(detail, dummy, img_row, img_col, nv * 4.0f / 9.0f, th_num, 8);



	for (int k = 0; k < imsize; ++k) {
		y1[k] += detail[k];
	}

	float nv_mod = nv / 3.8f;

	DCT3X3img(y2, y1, nv_mod * 2.0f, img_row, img_col, th_num);


	delete[] dummy;
	delete[] imgd1;
	delete[] imgd2;

	return EXIT_SUCCESS;
}

