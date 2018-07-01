//OpenMP version.  Edit and submit only this file.
/* Enter your details below
 * Name : Chris Lam
 * UCLA ID: 304995748
 * Email id: christopherwlam@gmail.com
 * New input
 */

/*
Optimizations:
1) [Deblur] Common subexpression in first nested-for loop (u[Index(x,y,z)])
	lnxsrv07 6/5/18. Many cores at 100% use by wesleyh. Made with GPROF
	Sequential took 16.669774 time units
	OpenMP took 13.571952 time units
	This resulted in a 1.228252x speed-up
2) [Gaussian Deblur] Rearranging loops to increase spatial locality (all following possible loops have been rearranged)
	lnxsrv07 6/5/18. Many cores at 100% use by wesleyh. Made with GPROF
	Sequential took 14.283527 time units
	OpenMP took 5.101178 time units
	This resulted in a 2.800045x speed-up
3)	[Deblur] Rearrange g[] initialization loop since iterations are independent
	lnxsrv07 6/5/18. Many cores at 100% use by wesleyh. Made with GPROF
	Sequential took 14.364249 time units
	OpenMP took 4.695986 time units
	This resulted in a 3.058836x speed-up
4 & 5) [Deblur] Common subexpression and rearranging data-independent for loop
	lnxsrv08 6/5/18. ~5 people at 100% cpu. No GPROF
	Sequential took 13.259767 time units
	OpenMP took 2.013507 time units
	This resulted in a 6.585409x speed-up
6) [Gaussian Deblur] Combine loops to take advantage of temporal locality (don't run same loops again when we can do it in 1)
	[Only the first 4 loops, since my rearrangement of the rest makes it a bit more complicated]
	lnxsrv08 6/8/18. I'm the only one. No GPROF
	Sequential took 9.795922 time units
	OpenMP took 1.930378 time units
	This resulted in a 5.074614x speed-up
	lnxsrv06 6/8/18. I'm the only one
	Sequential took 23.759488 time units
	OpenMP took 3.143649 time units
	This resulted in a 7.557933x speed-up
7) [OMP_Index] Define Macro
	lnxsrv08 6/8/18. I'm the only one
	Sequential took 9.782432 time units
	OpenMP took 2.020398 time units
	This resulted in a 4.841834x speed-up
8) [Gaussian Deblur] Reducing loops by grouping parallel code (e.g. we can boundary scale multiple places)
	lnxsrv02 6/8/18. I'm the only one
	Sequential took 19.717005 time units
	OpenMP took 2.588466 time units
	This resulted in a 7.617255x speed-up
9) [Gaussian Deblur] Loop unrolling + code motion for boundary scaling and post scaling
	lnxsrv09 6/8/18. 5 other people at 100% CPU.
	Sequential took 14.283315 time units
	OpenMP took 3.244445 time units
	This resulted in a 4.402391x speed-up
	Ending the deblur parallelization test
10) [Gaussian Blur & Deblur] Strength reduction/common sub expression.
	(Instead of calling Index() everytime, we can use the increment operator since we're accessing the array in row-major order)
	lnxsrv02 6/8/18. I'm the only one
	Sequential took 19.558082 time units
	OpenMP took 2.579507 time units
	This resulted in a 7.582101x speed-up
11) [Everywhere] Parallelism! Using 16 threads to compute faster
	lnxsrv08 6/8/18. 3 other's at 100%
	Sequential took 13.550239 time units
	OpenMP took 1.782179 time units
	This resulted in a 7.603186x speed-up

	lnxsrv03 6/8/18. I'm the only one.
	Sequential took 24.920047 time units
	OpenMP took 2.495305 time units
	This resulted in a 9.986773x speed-up

After parallelizing some loops in Gaussian Deblur
	lnxsrv03 6/8/18. I'm the only one.
	Sequential took 27.572652 time units
	OpenMP took 1.421386 time units
	This resulted in a 19.398426x speed-up
	
	lnxsrv08 6/8/18. 5 others at 100%
	Sequential took 12.062968 time units
	OpenMP took 1.254058 time units
	This resulted in a 9.619146x speed-up

	lnxsrv08 6/8/18. 1 other at 100%
	Sequential took 19.694374 time units
	OpenMP took 1.051001 time units
	This resulted in a 18.738682x speed-up

	lnxsrv08 6/8/18. 1 other. GRPOF on.
	Sequential took 11.033012 time units
	OpenMP took 1.029900 time units
	This resulted in a 10.712702x speed-up

	lnxsrv08 6/8/18. 6 others at 100%. GPROF off.
	Sequential took 13.515435 time units
	OpenMP took 1.112025 time units
	This resulted in a 12.153895x speed-up

	lnxsrv08 6/8/18. 2 others at 100%.
	Sequential took 18.104332 time units
	OpenMP took 1.288325 time units
	This resulted in a 14.052612x speed-up

	lnxsrv08 6/8/18. 3 others at 100%.
	Sequential took 16.978894 time units
	OpenMP took 1.093171 time units
	This resulted in a 15.531782x speed-up

*/

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int OMP_xMax;
#define xMax OMP_xMax
int OMP_yMax;
#define yMax OMP_yMax
int OMP_zMax;
#define zMax OMP_zMax

// Index(x, y, z) = (z*Y_MAX + y)*X_MAX + x
			//	  = z*Y_MAX*X_MAX + y*X_MAX + x
			//	  = (z * 2d_size) + row_offset + col_offset
int OMP_Index(int x, int y, int z)
{
	return ((z * yMax + y) * xMax + x);
}
// Opt 7) Define macro
#define Index(x, y, z) (((z) * yMax + (y)) * xMax + (x))//OMP_Index(x, y, z)
#define zLen xMax*yMax 	// the length of 1 z "movement"

double OMP_SQR(double x)
{
	return pow(x, 2.0);
}
#define SQR(x) OMP_SQR(x)

double* OMP_conv;
double* OMP_g;

void OMP_Initialize(int xM, int yM, int zM)
{
	xMax = xM;
	yMax = yM;
	zMax = zM;
	assert(OMP_conv = (double*)malloc(sizeof(double) * xMax * yMax * zMax));
	assert(OMP_g = (double*)malloc(sizeof(double) * xMax * yMax * zMax));

	// Opt 11) Parallelism! OpenMP
	omp_set_num_threads(16);
}
void OMP_Finish()
{
	free(OMP_conv);
	free(OMP_g);
}
void OMP_GaussianBlur(double *u, double Ksigma, int stepCount)
{
	double lambda = (Ksigma * Ksigma) / (double)(2 * stepCount);
	double nu = (1.0 + 2.0*lambda - sqrt(1.0 + 4.0*lambda))/(2.0*lambda);
	int x, y, z, step;
	double boundryScale = 1.0 / (1.0 - nu);
	double postScale = pow(nu / lambda, (double)(3 * stepCount));

	//int bX, bZ, bY, blockSize = 16;
	int index = 0, indexA = 0;
	/*
		   ________________
		  /	  /	  /	  /	  /|
		 / z /	 /	 /	 / |
		/___/___/___/___/  |
		| x	| - | -	| >	|  |
		| y	|	|	|	|  |
		| |	|	|	|	|  /
		| v	|	|	|	| /
		|___|___|___|___|/
	
	Memory is laid out (x, y, z):
				/	002, 102, 202,
			/	001, 101, 201,
		|	000, 100, 200,
		|	010, 110, 210,
		|	020, 120, 220,

		x continguous, each y jumps xMax, each z jumps yMax (which jumps xMax times)

	Best order is for(z) for(y) for(x)
	Worst order is for(x) for(y) for(z)
	*/

	// TODO 3D tiling whoa

	for(step = 0; step < stepCount; step++)
	{
		#pragma omp parallel
		{
		
		/*
			Opt 2) Rearranging loops to increase spatial locality (all following possible loops have been rearranged)
		*/

		/*
			Opt 6) Combine loops to take advantage of temporal locality (don't run same loops again when we can do it in 1)
					[Only the first 4 loops, since my rearrangement of the rest makes it a bit more complicated]
		*/

		// for(z = 0; z < zMax; z++)
		// {
		// 	for(y = 0; y < yMax; y++)
		// 	{
		// 		// Traverses whole z-axis for every element in the 1st column
				
		// 	}
		// }
		// This goes from x=1 to x=xMax, taking the prev yz cross-section and adding it (after *nu) to curr cross-section

		/*
			Opt 10) Strength reduction/common sub expression. Instead of calling Index() everytime, we can use the increment operator
		*/
		#pragma omp for private(x,y,z,index)
		for( z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				index = Index(0, y, z);
				u[index++] *= boundryScale;
				
				for(x=1; x<xMax; x++)
				{
					u[index] += u[index-1]*nu;
					index++;
				}
				
				u[Index(0, y, z)] *= boundryScale;
				index = Index(xMax-2, y, z);
				for(x = xMax - 2; x >= 0; x --)
				{
					u[index] += u[index+1] * nu;
					index--;
				}
				
			}
		}
		
		// This double-for scales the left face (yz cross-section at x=0)
		// TODO-embarrassingly parallel
		// for(z = 0; z < zMax; z++)
		// {
		// 	for(y = 0; y < yMax; y++)
		// 	{
				
		// 	}
		// }
		// This goes from x=xMax to x=1 (right to left), taking the prev yz cross-section and adding it (after *nu) to curr cross-section
		// for(z = 0; z < zMax; z++)
		// {
		// 	// TODO-embarrassingly parallel
		// 	for(y = 0; y < yMax; y++)
		// 	{
		// 		for(x = xMax - 2; x >= 0; x--)
		// 		{
		// 			// Traverses each element in each col from xMax-1~0, and foreach, go into z-axis
		// 			u[Index(x, y, z)] += u[Index(x + 1, y, z)] * nu;
		// 		}
		// 	}
		// }



		//This scales the top face (xz cross-section at y=0)

		/*
			Opt 9) Loop unrolling
		*/
		// #pragma omp single
		// {
		#pragma omp for private(x,y,z,index)
		for(z = 0; z < zMax; z++)
		{
			index = Index(0, 0, z);
			for(x = 0; x < xMax; x += 4)
			{
				// Traverses z-axis for every element in the first row
				u[index++] *= boundryScale;
				u[index++] *= boundryScale;
				u[index++] *= boundryScale;
				u[index++] *= boundryScale;
			}
		}
		//}
		// This goes from y=1 to y=yMax (rows top to bottom), left (x=0) to right so that prev xz cross-section * nu added to curr cross-section
		// #pragma omp single
		// {
		#pragma omp for private(x,y,z,index)
		for(z = 0; z < zMax; z++)
		{
			for(y = 1; y < yMax; y++)
			{
				index = Index(0, y, z);
				for(x = 0; x < xMax; x++)
				{
					u[index] += u[index-xMax] * nu;
					index++;
				}
			}
		}
		//}
		// for(bZ = 0; bZ < zMax; bZ += blockSize)
		// {
		// 	for(bX = 0; bX < xMax; bX += blockSize)
		// 	{
		// 		for(bY = 1; bY < yMax; bY += blockSize)
		// 		{
		// 			for(z = bZ; z < (bZ + blockSize) && z < zMax; z++)
		// 			{
		// 				for(x = bX; x < (bX + blockSize) && x < xMax; x++)
		// 				{
		// 					u[Index(x, 0, z)] *= boundryScale;
		// 					for(y = bY; y < (bY + blockSize) && y < yMax; y++)
		// 					{
		// 						u[Index(x, y, z)] += u[Index(x, y - 1, z)] * nu;
		// 					}
		// 				}
		// 			}
		// 		}
		// 	}
		// }

		
		// This scales the bottom face, moving from x,z=0,0 to x,z=0,zMax-1 to x,z=1,0 (bottom left to top right from top view)
		// #pragma omp single
		// {
		#pragma omp for private(x,y,z,index)
		for(z = 0; z < zMax; z++)
		{
			index = Index(0, yMax-1, z);
			for(x = 0; x < xMax; x+=2)
			{
				u[index++] *= boundryScale;
				u[index++] *= boundryScale;
			}
		}
		//}
		// This goes from y=yMax-2 to y=0, taking prev xz cross-sec ()
		// #pragma omp single
		// {
		#pragma omp for private(x,y,z,index)
		for(z = 0; z < zMax; z++)
		{
			for(y = yMax - 2; y >= 0; y--)
			{
				index = Index(0,y,z);
				for(x = 0; x < xMax; x++)
				{
					u[index] += u[index+xMax] * nu;
					index++;
				}
			}
		}
		// }

		/*
			Opt 8) Group parallel motions (e.g. moving down the cube and up the cube, can be done in parallel since operations are commutative/associative)
		*/
		// scale front face
		// #pragma omp single
		// {
		#pragma omp for private(x,y,z,index,indexA)
		for(y = 0; y < yMax; y++)
		{
			index = Index(0,y,0);
			indexA = Index(0,y,zMax-1);
			for(x = 0; x < xMax; x++)
			{
				u[index++] *= boundryScale;
				u[indexA--] *= boundryScale;
			}
		}
		// }
		// from front face (z=1) to back face
		#pragma omp single
		{
		//#pragma omp for private(x,y,z,index,indexA)
		for(z = 1; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				index = Index(0,y,z);
				indexA = Index(0,y,zMax-z-1);
				for(x = 0; x < xMax; x++)
				{
					u[index] = u[index-zLen] * nu;
					u[indexA] += u[indexA+zLen] * nu;
					index++;
					indexA++;
				}
			}
		}
		}
	}
	}

	// Scale the whole cube, going in z-dir (inward), then y-dir (down row), then x-dir (right col)
	#pragma omp parallel for private(x,y,z,index,indexA)
	for(z = 0; z < zMax; z++)
	{
		for(y = 0; y < yMax; y++)
		{
			index = Index(0, y, z);
			for(x = 0; x < xMax; x+=2)
			{
				u[index++] *= postScale;
				u[index++] *= postScale;
			}
		}
	}
}


void OMP_Deblur(double* u, const double* f, int maxIterations, double dt, double gamma, double sigma, double Ksigma)
{
	double epsilon = 1.0e-7;
	double sigma2 = SQR(sigma);
	int x, y, z, iteration;
	int converged = 0;
	int lastConverged = 0;
	int fullyConverged = (xMax - 1) * (yMax - 1) * (zMax - 1);
	double* conv = OMP_conv;
	double* g = OMP_g;

	for(iteration = 0; iteration < maxIterations && converged != fullyConverged; iteration++)
	{
		// Block this, how to handle overlap? No data dependence since we're reading from
		// u and writing to g!
		
		// Opt 3) Rearrange this loop since iterations are independent
		// TODO: Unroll, tile, and parallelize
		#pragma omp parallel for private(x,y,z)
		for(z = 1; z < zMax - 1; z++)
		{
			for(y = 1; y < yMax - 1; y++)
			{
				int index = Index(1,y,z);
				for(x = 1; x < xMax - 1; x++)
				{
					// Opt 1) Temporal Locality [SCRATCHED because of optimization #10]
					double pivot = u[index];
					g[index] = 1.0 / sqrt(epsilon + 
						SQR(pivot - u[index+1]) + 
						SQR(pivot - u[index-1]) + 
						SQR(pivot - u[index+xMax]) + 
						SQR(pivot - u[index-xMax]) + 
						SQR(pivot - u[index+zLen]) + 
						SQR(pivot - u[index-zLen]));
					index++;
				}
			}
		}
		memcpy(conv, u, sizeof(double) * xMax * yMax * zMax);
		OMP_GaussianBlur(conv, Ksigma, 3);
		// Opt 4) Common subexpression and rearranging data-independent for loop
		#pragma omp parallel for private(x,y,z)
		for(z = 0; z < zMax; z++)
		{
			for(y = 0; y < yMax; y++)
			{
				int index = Index(0,y,z);
				for(x = 0; x < xMax; x++)
				{
					double fN = f[index];
					double r = conv[index] * fN / sigma2;
					r = (r * (2.38944 + r * (0.950037 + r))) / (4.65314 + r * (2.57541 + r * (1.48937 + r)));
					conv[index] -= fN * r;
					index++;
				}
			}
		}
		OMP_GaussianBlur(conv, Ksigma, 3);
		converged = 0;
		// Opt 5) Common subexpression and rearranging data-independent for loop
		for(z = 1; z < zMax - 1; z++)
		{
			for(y = 1; y < yMax - 1; y++)
			{
				int index = Index(1,y,z);
				for(x = 1; x < xMax - 1; x++)
				{
					double oldVal = u[index];
					double a = g[index+1], b=g[index-1], c=g[index+xMax],
							d= g[index-xMax], e=g[index+zLen], h=g[index-zLen];
					double newVal = (oldVal + dt * ( 
						u[index-1] * b + 
						u[index+1] * a + 
						u[index-xMax] * d + 
						u[index+xMax] * c + 
						u[index-zLen] * h + 
						u[index+zLen] * e - gamma * conv[index])) /
						(1.0 + dt * (a+b+c+d+e+h));
					if(fabs(oldVal - newVal) < epsilon)
					{
						converged++;
					}
					u[index] = newVal;
					index++;
				}
			}
		}
		if(converged > lastConverged)
		{
			printf("%d pixels have converged on iteration %d\n", converged, iteration);
			lastConverged = converged;
		}
	}
}