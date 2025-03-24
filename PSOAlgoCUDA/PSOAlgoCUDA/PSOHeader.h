#pragma once

#include<vector>
#include<iostream>
#include<thrust/device_vector.h>

#define BADDATA -999


struct SAmp {
	float PzPos_um = 0; // unit : um [2/14/2021 FSM]
};
std::vector<SAmp> ImgInfo;

struct SFrng {
	bool bZeroed = false;

};

int nSteps = 0; // total number of steps moved
float UStep_um = 0, zRange_um = 0;
int wd = 0, ht = 0, bpp = 0;

//Functions
void readZPosition(std::string csv_path);
void readImage(std::string img_path);
void GetDim(int& x, int& y, int& b);

enum FRP {
	// definition of fringe point
	REDA,  // Amplitude 1
	GRNA,  // Amplitude 2
	BLUA,  // Amplitude 3
	WHTA,  // Amplitude 4
	TMP1,  // temporary, intermediate
	ZAXS  // z value
};

float wlen_um[5] = { 0 }; //  non persist RGBWE

struct SROI {
	int idx = 0;
	int i1 = 0, i2 = 0;
};

float* Get(FRP Ch, int ist, int sz);
// This allows you to keep the data on the GPU
thrust::device_vector<float> zaxs_d, reda_d, grna_d, blua_d, whta_d,
phs1_d, phs2_d, phse_d, uph1_d, uph2_d,
uphe_d, ordr_d, vis1_d, vis2_d, rslt_d,
tmp1_d, tmp2_d, tmp3_d;

// Size tracker with corrected size
int Sz[5] = { 0 };
void MaxMin(FRP Ch, const SROI& R, int sz, bool bAve);

struct SStat {
	float fave = BADDATA, fmax = BADDATA, fmin = BADDATA;
	int imx = 0, imn = 0, inc = 1;
};

SStat St[5];
