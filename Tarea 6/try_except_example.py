import numpy as np


def cholesky_plus_identity(A: np.ndarray, beta=1e-3):
	try:
		L = np.linalg.cholesky(A)
	except np.linalg.LinAlgError:
		#print('\'A\' matrix is not positive definite.')
		I = np.eye(len(A))*beta
		A=cholesky_plus_identity(A+I, 1e-3)
		return A
	else:
		return A
    


if __name__ == '__main__':
	
	# Positive definite example:
	A = np.eye(5)
	A = cholesky_plus_identity(A, 1e-3)


	
	# A not positive definite matrix
	B = -np.eye(5)
	B = cholesky_plus_identity(B, 1e-3)

#%%
C = -np.eye(5)
D=cholesky_plus_identity(C, beta=1e-3)