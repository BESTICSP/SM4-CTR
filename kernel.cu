#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Windows.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>

typedef unsigned char muint8;  //一个字节
typedef unsigned long muint32; //4个字节 
//128bit是16字节

/**************S盒******************/
#ifdef USE_CONSTANT
__constant__
#endif
__constant__ muint8 Sbox[256] = {
	0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
	0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
	0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
	0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
	0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
	0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
	0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
	0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
	0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
	0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
	0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
	0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
	0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
	0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
	0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
	0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48
};
/*********************ck**********************/
const muint32 ck[32] = {
	0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269,
	0x70777e85, 0x8c939aa1, 0xa8afb6bd, 0xc4cbd2d9,
	0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249,
	0x50575e65, 0x6c737a81, 0x888f969d, 0xa4abb2b9,
	0xc0c7ced5, 0xdce3eaf1, 0xf8ff060d, 0x141b2229,
	0x30373e45, 0x4c535a61, 0x686f767d, 0x848b9299,
	0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209,
	0x10171e25, 0x2c333a41, 0x484f565d, 0x646b7279 };

/****************加解密时用到的计算**************/
#define Rotl(_x, _y) (((_x) << (_y)) | ((_x) >> (32 - (_y))))  
#define L1(_B) ((_B) ^ Rotl(_B, 2) ^ Rotl(_B, 10) ^ Rotl(_B, 18) ^ Rotl(_B, 24))  
#define L2(_B) ((_B) ^ Rotl(_B, 13) ^ Rotl(_B, 23))
#define S_box(_A) (Sbox[(_A) >> 24 & 0xFF] << 24 ^ Sbox[(_A) >> 16 & 0xFF] << 16 ^ Sbox[(_A) >>  8 & 0xFF] <<  8 ^ Sbox[(_A) & 0xFF])  
/*********************输入128bits明文/密文，加密/解密*************/
/**
*input 输入，
*output 输出
*rk 轮密钥
*flag CTR模式标志
*plaintext 明文，CTR时存在
*/
__host__ __device__ void SM4Crypt(const muint8* input, muint8* output, muint32* rk, int flag = 0, const muint8* plaintext = NULL) {
	//
	muint32 r, mid, x0, x1, x2, x3, * p;
	p = (muint32*)input;
	x0 = p[0]; x1 = p[1]; x2 = p[2]; x3 = p[3];

	//

	for (r = 0; r < 32; r += 4) {
		//利用轮密钥，加解密
		mid = x1 ^ x2 ^ x3 ^ rk[r + 0];
		mid = S_box(mid);
		x0 ^= L1(mid);
		mid = x2 ^ x3 ^ x0 ^ rk[r + 1];
		mid = S_box(mid);
		x1 ^= L1(mid);
		mid = x3 ^ x0 ^ x1 ^ rk[r + 2];
		mid = S_box(mid);
		x2 ^= L1(mid);
		mid = x0 ^ x1 ^ x2 ^ rk[r + 3];
		mid = S_box(mid);
		x3 ^= L1(mid);
	}

	p = (muint32*)output;
	p[0] = x3; p[1] = x2; p[2] = x1; p[3] = x0;

	//if mode="CTR"
	if (flag == 1) {
		muint32* ct;
		ct = (muint32*)plaintext;
		p[0] ^= ct[0]; p[1] ^= ct[1]; p[2] ^= ct[2]; p[3] ^= ct[3];
	}


}

/*********************输入128bits明文/密文，加密/解密，CBC、CFB、OFB*************/
/**
*input 输入，
*output 输出
*rk 轮密钥
*/

/****************SM4密钥拓展算法******************/
/**
*Key初始密钥
*rk 轮密钥
* CryptFlag 加密/解密标识
*/
void SM4KeyExt(muint8* Key, muint32* rk, muint32 CryptFlag) {

	muint32 r, mid, x0, x1, x2, x3, * p;
	p = (muint32*)Key;
	x0 = p[0]; x1 = p[1]; x2 = p[2]; x3 = p[3];
	x0 ^= 0xa3b1bac6;
	x1 ^= 0x56aa3350;
	x2 ^= 0x677d9197;
	x3 ^= 0xb27022dc;
	for (r = 0; r < 32; r += 4) {
		mid = x1 ^ x2 ^ x3 ^ ck[r];
		mid = S_box(mid);
		rk[r] = x0 ^= L2(mid);
		mid = x2 ^ x3 ^ x0 ^ ck[r + 1];
		mid = S_box(mid);
		rk[r + 1] = x1 ^= L2(mid);
		mid = x3 ^ x0 ^ x1 ^ ck[r + 2];
		mid = S_box(mid);
		rk[r + 2] = x2 ^= L2(mid);
		mid = x0 ^ x1 ^ x2 ^ ck[r + 3];
		mid = S_box(mid);
		rk[r + 3] = x3 ^= L2(mid);
	}
	if (CryptFlag == 1) {
		for (r = 0; r < 16; r++) {
			mid = rk[r], rk[r] = rk[31 - r], rk[31 - r] = mid;
		}
	}
}

//kernel
__global__  void SM4_encrypt_cuda_ecb(const muint8* p, muint8* c, muint32* rk, int num_cipher_blocks) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < num_cipher_blocks) {

		SM4Crypt(&(p[i * 16]), &(c[i * 16]), rk);
	}
	//
	//printf("num_cipher_blocks:%d\n",num_cipher_blocks);
	//printf("thread:i=%d\n",i);

}

__global__  void SM4_encrypt_cuda_ctr(muint8* p, muint8* c, muint32* rk, int num_cipher_blocks, muint8* counter) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < num_cipher_blocks) {
		muint8 ca[16];
		for (int w = 0; w < 16; w++) {
			ca[w] = counter[w];
		}
		ca[15] += i;
		SM4Crypt(ca, &(c[i * 16]), rk, 1, &(p[i * 16]));

	}
	//
	//printf("num_cipher_blocks:%d\n",num_cipher_blocks);
	//printf("thread:i=%d\n",i);

}

//ECB cuda
cudaError_t SM4_cuda_ecb(char* text, char* cipher, int size, muint32* key) {
	muint8* p, * c;
	muint32* rk;
	cudaError_t cudaStatus;
	//select GPU to run on
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda SetDevice failed\n");
		goto Error;
	}
	//分配内存
	cudaStatus = cudaMalloc((void**)&p, size * sizeof(muint8));//明文
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&c, size * sizeof(muint8));//密文
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&rk, 32 * sizeof(muint32));//轮密钥
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		goto Error;
	}
	//copy,from host to device
	cudaStatus = cudaMemcpy(p, text, size * sizeof(muint8), cudaMemcpyHostToDevice);//明文
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed");
		goto Error;
	}

	cudaStatus = cudaMemcpy(rk, key, 32 * sizeof(muint32), cudaMemcpyHostToDevice);//轮密钥
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed");
		goto Error;
	}

	//device properties
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	//计算开启GPU几个Block，多少线程
	int blocks, threads;
	int n = size >> 4;

	/*自动计算最好的线程数和块数*/
	if (n > 1) {
		threads = (n < prop.maxThreadsPerBlock) ? n : prop.maxThreadsPerBlock;
		blocks = n / threads;
	}
	else {
		threads = 1; blocks = 1;
	}
	printf("maxThreadsPerBlock=%d\n", prop.maxThreadsPerBlock);
	printf("n=%d; threads/block=%d; blocks=%d;\n", n, threads, blocks);

	//kernel
	_LARGE_INTEGER time_start;	//开始时间
	_LARGE_INTEGER time_over;	//结束时间
	double  CPU_totaltime;
	double dqFreq;		//计时器频率
	LARGE_INTEGER f;	//计时器频率
	QueryPerformanceFrequency(&f);
	dqFreq = (double)f.QuadPart;

	QueryPerformanceCounter(&time_start);	//开始计时

	//nn为加解密的次数
	for (int nn = 0; nn < 1; nn++) {
		SM4_encrypt_cuda_ecb << <blocks, threads >> > (p, c, rk, n);
	}

	//check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed:%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//waits for the kernel to finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel\n", cudaStatus);
		goto Error;
	}

	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("Encryption/Dencryption kernels in GPU alone takes %1f ms.\n", CPU_totaltime / 1000);

	//Copy Back to Host
	cudaStatus = cudaMemcpy(cipher, c, size * sizeof(muint8), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed\n");
		goto Error;
	}


Error:
	cudaFree(c);
	cudaFree(p);
	cudaFree(rk);
	return cudaStatus;
}

/*********************CTR模式*****************************/
//CTR cuda
cudaError_t SM4_cuda_ctr(char* text, char* cipher, int size, muint32* key, muint8* Counter) {
	cudaError_t cudaStatus;
	muint8* p, * c, * counter;
	muint32* rk;

	//select GPU to run on
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda SetDevice failed\n");
		goto Error;
	}

	//分配内存
	cudaStatus = cudaMalloc((void**)&counter, 16 * sizeof(muint8));//明文
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&p, size * sizeof(muint8));//明文
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&c, size * sizeof(muint8));//密文
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&rk, 32 * sizeof(muint32));//轮密钥
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed\n");
		goto Error;
	}

	//copy,from host to device
	cudaStatus = cudaMemcpy(counter, Counter, 16 * sizeof(muint8), cudaMemcpyHostToDevice);//明文
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed");
		goto Error;
	}

	cudaStatus = cudaMemcpy(p, text, size * sizeof(muint8), cudaMemcpyHostToDevice);//明文
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed");
		goto Error;
	}

	cudaStatus = cudaMemcpy(rk, key, 32 * sizeof(muint32), cudaMemcpyHostToDevice);//轮密钥
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed");
		goto Error;
	}

	//device properties
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	//计算开启GPU几个Block，多少线程
	//threads是块大小;blocks是块数量;n是总的线程数目
	int blocks, threads;
	int n = size >> 4;
	if (n > 1) {
		threads = (n < prop.maxThreadsPerBlock) ? n : prop.maxThreadsPerBlock;
		blocks = n / threads;
	}
	else {
		threads = 1; blocks = 1;
	}
	printf("maxThreadsPerBlock=%d\n", prop.maxThreadsPerBlock);
	printf("n=%d; threads/block=%d; blocks=%d;\n", n, threads, blocks);


	//kernel
	_LARGE_INTEGER time_start;	//开始时间
	_LARGE_INTEGER time_over;	//结束时间
	double  CPU_totaltime;
	double dqFreq;		//计时器频率
	LARGE_INTEGER f;	//计时器频率
	QueryPerformanceFrequency(&f);
	dqFreq = (double)f.QuadPart;

	QueryPerformanceCounter(&time_start);	//开始计时
	//nn为加解密的次数
	for (int nn = 0; nn < 1; nn++) {
		SM4_encrypt_cuda_ctr << <blocks, threads >> > (p, c, rk, n, counter);
	}
	//check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed:%s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//waits for the kernel to finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel\n", cudaStatus);
		goto Error;
	}

	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("Encryption/Dencryption kernels in GPU alone takes %1f ms.\n", CPU_totaltime / 1000);
	//Copy Back to Host
	cudaStatus = cudaMemcpy(cipher, c, size * sizeof(muint8), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed\n");
		goto Error;
	}


Error:
	cudaFree(c);
	cudaFree(p);
	cudaFree(rk);
	return cudaStatus;
	return cudaStatus;
}

//CPU串行部分
void SM4_cpu_ecb(char* text, char* cipher, int size, muint32* key) {

	muint8* p = (muint8*)malloc(size * sizeof(muint8));
	memcpy(p, text, size * sizeof(muint8));
	muint8* c = (muint8*)malloc(size * sizeof(muint8));

	muint32* rk = (muint32*)malloc(32 * sizeof(muint32));
	memcpy(rk, key, 32 * sizeof(muint32));

	int n = size >> 4;

	for (int i = 0; i < n; i++) {

		SM4Crypt(&(p[i * 16]), &(c[i * 16]), rk);

	}

	memcpy(cipher, c, size * sizeof(muint8));


}

/*********************CTR模式*****************************/
void SM4_cpu_ctr(char* text, char* cipher, int size, muint32* key, muint8* Counter) {

	muint8* p = (muint8*)malloc(size * sizeof(muint8));
	memcpy(p, text, size * sizeof(muint8));
	muint8* c = (muint8*)malloc(size * sizeof(muint8));
	muint8* counter = (muint8*)malloc(16 * sizeof(muint8));
	memcpy(counter, Counter, 16 * sizeof(muint8));

	muint32* rk = (muint32*)malloc(32 * sizeof(muint32));
	memcpy(rk, key, 32 * sizeof(muint32));

	int n = size >> 4;

	for (int i = 0; i < n; i++) {
		muint8 ca[16];
		for (int w = 0; w < 16; w++) {
			ca[w] = counter[w];
		}
		ca[15] += i;
		SM4Crypt(ca, &(c[i * 16]), rk, 1, &(p[i * 16]));

	}

	memcpy(cipher, c, size * sizeof(muint8));


}

int main() {

	///生成密钥
	unsigned char key[16] = { 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10 };
	muint32 rk[32];
	muint32 drk[32];
	SM4KeyExt(key, rk, 0);//加密轮密钥
	SM4KeyExt(key, drk, 1);//解密轮密钥


	////读取文件内容
	FILE* file;
	char filename[] = "    ";//此处填写需要加密的文件路径
	file = fopen(filename, "r");
	if (file == NULL) {
		printf("open file failed!\n");
		system("pause");
		return 0;
	}
	fseek(file, 0, SEEK_END);
	int size = ftell(file);
	fseek(file, 0, SEEK_SET);
	printf("size:%d\n", size);
	//补齐
	int add;
	add = (16 - size % 16) % 16;
	printf("add:%d\n", add);
	char* p = (char*)calloc(size + add, sizeof(char));
	fread(p, size, 1, file);
	for (int i = size; i < size + add; i++)
	{
		p[i] = (char)add;
	}
	fclose(file);
	size = size + add;

	printf("now size:%d:\n", size);

	printf("明文大小:%dMB\n", size / (1024 * 1024));

	//输出要加密的明文：
	//printf("要加密的明文:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",p[i]);
	//}
	printf("\n");

	//m存密文
	char* m = (char*)calloc(size, sizeof(char));
	//pt存解密后的明文
	char* pt = (char*)calloc(size, sizeof(char));

	_LARGE_INTEGER time_start;	//开始时间
	_LARGE_INTEGER time_over;	//结束时间
	double  CPU_totaltime;
	double dqFreq;		//计时器频率
	LARGE_INTEGER f;	//计时器频率
	QueryPerformanceFrequency(&f);
	dqFreq = (double)f.QuadPart;

	int n;	//循环次数

	cudaError_t cudaStatus;

	//ECB预
#if 1
	///////////////////ECB串行 //////////////////////////////
	//加密
	printf("NOW ECB加密串行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时

	SM4_cpu_ecb(p, m, size, rk);

	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("ECB Encryption takes %lf ms.\n", CPU_totaltime / 1000);

	//输出密文
	//printf("ECB模式加密后的密文:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",m[i]);
	//}
	printf("\n");


	//解密
	printf("ECB解密串行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时

	SM4_cpu_ecb(m, pt, size, drk);

	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("ECB Dencryption takes %1f ms.\n", CPU_totaltime / 1000);
	//输出还原的明文
	//printf("ECB模式解密后:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",pt[i]);
	//}
	printf("\n");


	///////////////////ECB并行 //////////////////////////////
	printf("NOW ECB加密并行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时
	cudaStatus = SM4_cuda_ecb(p, m, size, rk);
	if (cudaStatus != cudaSuccess) {
		printf("SM4_cuda_ecb failed\n");

	}
	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("ECB Enncryption takes %1f ms.\n", CPU_totaltime / 1000);

	//输出密文
	//printf("ECB模式加密后的密文:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",m[i]);
	//}
	printf("\n");


	//解密
	printf("ECB解密并行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时
	cudaStatus = SM4_cuda_ecb(m, pt, size, drk);
	if (cudaStatus != cudaSuccess) {
		printf("SM4_cuda_ecb_dencryption failed\n");
	}
	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("ECB Dencryption takes %1f ms.\n", CPU_totaltime / 1000);
	//输出还原的明文
	//printf("ECB模式解密后:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",pt[i]);
	//}
	printf("\n");
#endif

	//ECB
#if 1
	///////////////////ECB串行 //////////////////////////////
	//加密
	printf("NOW ECB加密串行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时
	//n为串行加密进行的次数
	for (n = 0; n < 1; n++) {
		SM4_cpu_ecb(p, m, size, rk);
	}
	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("ECB Encryption takes %lf ms.\n", CPU_totaltime / 1000);

	//输出密文
	//printf("ECB模式加密后的密文:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",m[i]);
	//}
	printf("\n");


	//解密
	printf("ECB解密串行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时
	//n为串行解密进行的次数
	for (n = 0; n < 1; n++) {
		SM4_cpu_ecb(m, pt, size, drk);
	}

	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("ECB Dencryption takes %1f ms.\n", CPU_totaltime / 1000);
	//输出还原的明文
	//printf("ECB模式解密后:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",pt[i]);
	//}
	printf("\n");


	///////////////////ECB并行 //////////////////////////////
	printf("NOW ECB加密并行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时
	cudaStatus = SM4_cuda_ecb(p, m, size, rk);
	if (cudaStatus != cudaSuccess) {
		printf("SM4_cuda_ecb failed\n");

	}
	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("ECB Enncryption takes %1f ms.\n", CPU_totaltime / 1000);

	//输出密文
	//printf("ECB模式加密后的密文:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",m[i]);
	//}
	printf("\n");


	//解密
	printf("ECB解密并行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时
	cudaStatus = SM4_cuda_ecb(m, pt, size, drk);
	if (cudaStatus != cudaSuccess) {
		printf("SM4_cuda_ecb_dencryption failed\n");
	}
	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("ECB Dencryption takes %1f ms.\n", CPU_totaltime / 1000);
	//输出还原的明文
	//printf("ECB模式解密后:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",pt[i]);
	//}
	printf("\n");
#endif

	//CTR
#if 1
	/////////////////////////CTR串行//////////////////////
	//产生一个随机数Counter,
	muint8 Counter[16] = "abftnbtfreskjuy";
	printf("NOW CTR加密串行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时
	for (n = 0; n < 1; n++) {
		SM4_cpu_ctr(p, m, size, rk, Counter);
	}
	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("CTR Encryption takes %1f ms.\n", CPU_totaltime / 1000);

	//输出密文
	//printf("CTR模式加密后的密文:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",m[i]);
	//}
	printf("\n");

	//解密
	printf("CTR解密串行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时
	for (n = 0; n < 1; n++) {
		SM4_cpu_ctr(m, pt, size, rk, Counter);
	}
	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("CTR Decryption takes %1f ms.\n", CPU_totaltime / 1000);

	//输出还原的明文
	//printf("CTR模式解密后:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",pt[i]);
	//}
	printf("\n");

	/////////////////////////CTR并行//////////////////////
	printf("NOW CTR加密并行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时
	cudaStatus = SM4_cuda_ctr(p, m, size, rk, Counter);
	if (cudaStatus != cudaSuccess) {
		printf("SM4_cuda_ctr failed\n");
	}
	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("CTR Encryption takes %1f ms.\n", CPU_totaltime / 1000);

	/*输出密文*/
	/*
		printf("CTR模式加密后的密文:\n");
		for(int i=0;i<size;i++){
		printf("%c",m[i]);
	}
	printf("\n");
	*/

	//解密
	printf("CTR解密并行...\n");
	QueryPerformanceCounter(&time_start);	//开始计时
	cudaStatus = SM4_cuda_ctr(m, pt, size, rk, Counter);
	if (cudaStatus != cudaSuccess) {
		printf("SM4_cuda_CTR_dencryption failed\n");
	}
	QueryPerformanceCounter(&time_over);	//计时结束
	CPU_totaltime = 1000000 * (time_over.QuadPart - time_start.QuadPart) / dqFreq;
	printf("CTR Decryption takes %1f ms.\n", CPU_totaltime / 1000);


	//输出还原的明文
	//printf("CTR模式解密后:\n");
	//for(int i=0;i<size;i++){
	//	printf("%c",pt[i]);
	//}
	printf("\n");

#endif
}