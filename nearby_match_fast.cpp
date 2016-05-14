#include "mex.h"
#include <stdlib.h>
#include <math.h>
#include <vector>

#define NUM_FEATURES 128

typedef struct {
	int queryIdx;
	int dbIdx;
} Pair;

using namespace std;

/*
paras:
query_keypoints, db_keypoints, query_descriptors, db_descriptors
types:
matrix: 2 * N, 2 * N, 128 * N, 128 * N
*/
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	double thresDist = 32;
	double thresRatio = 1.5;
    if (nrhs >= 5) {
        thresDist = mxGetScalar(prhs[4]);
    }
    if (nrhs >= 6) {
        thresRatio = mxGetScalar(prhs[5]);
    }
    
	double* queryKeypoints = mxGetPr(prhs[0]);
	double* dbKeypoints = mxGetPr(prhs[1]);
	char* queryDescriptors = (char *)mxGetPr(prhs[2]);
	char* dbDescriptors = (char *)mxGetPr(prhs[3]);

	int nQueryKeypoints = mxGetN(prhs[0]);
	int nDbKeypoints = mxGetN(prhs[1]);

	vector<Pair> matches;
	vector<int> scores;

	double* pQueryKeypoints = queryKeypoints;
	char* pQueryDescriptors = queryDescriptors;

	double tmp = 0;
	int tmp2 = 0;
	double dist = 0;
	for (int i = 0; i < nQueryKeypoints; i++) {
		int bestIdx = -1;
		int bestScore = INT_MAX;
		int secondBestScore = INT_MAX;

		double *pDbKeypoints = dbKeypoints;
		char *pDbDescriptors = dbDescriptors;
		for (int j = 0; j < nDbKeypoints; j++) {
			dist = 0;
			tmp = pDbKeypoints[0] - pQueryKeypoints[0];
			dist += tmp * tmp;
			tmp = pDbKeypoints[1] - pQueryKeypoints[1];
			dist += tmp * tmp;

			if (dist < thresDist * thresDist) {
				int score = 0;
				for (int k = 0; k < NUM_FEATURES; k++) {
					tmp2 = (int)pDbDescriptors[k] - (int)pQueryDescriptors[k];
					score += tmp2 * tmp2;
					if (score >= secondBestScore) {
						break;
					}
				}
				if (score < bestScore) {
					bestIdx = j;
					secondBestScore = bestScore;
					bestScore = score;
				} else if (score < secondBestScore) {
					secondBestScore = score;
				}
			}

			pDbKeypoints += 2;
			pDbDescriptors += NUM_FEATURES;
		}

		if (bestIdx != -1 && (double)bestScore * thresRatio < (double)secondBestScore) {
			Pair match;
			match.queryIdx = i;
			match.dbIdx = bestIdx;
			matches.push_back(match);
			scores.push_back(bestScore);
		}

		pQueryKeypoints += 2;
		pQueryDescriptors += NUM_FEATURES;
	}

	int nMatches = matches.size();
	plhs[0] = mxCreateNumericMatrix(2, nMatches, mxINT32_CLASS, mxREAL);
	plhs[1] = mxCreateNumericMatrix(1, nMatches, mxINT32_CLASS, mxREAL);
	int *pMatches = (int *)mxGetPr(plhs[0]);
	int *pScores = (int *)mxGetPr(plhs[1]);

	for (int i = 0; i < nMatches; i++) {
		pMatches[0] = matches[i].queryIdx + 1;
		pMatches[1] = matches[i].dbIdx + 1;
		pScores[0] = scores[i];
		pMatches += 2;
		pScores++;
	}

}




