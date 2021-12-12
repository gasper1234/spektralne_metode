import numpy as np
import matplotlib.pyplot as plt
from numpy import fft

N = 1000
tN = 1000
t = 10
t_step = t/tN


a = 1
D = 0.05
sigma = 0.5

N_step = a/N

def T_0(x):
	return np.exp(-x**2/sigma**2)

x = np.linspace(-a, a, N)
T = np.array([x for _ in range(tN)])

T[0] = T_0(T[0])
#T[0][0] = T[0][-1] = 0

#fourie test
'''
fig, ax = plt.subplots(3, 1)
x_freq = np.linspace(1, 1/2*N_step, N)
ax[0].plot(x, T[0])
ax[1].plot(x_freq, fft.fftshift(fft.fft(T[0]))/N)
ax[2].plot(x, fft.ifft(fft.fft(T[0])))
plt.show()
'''
def f_PRP():
	for i in range(tN-1):
		F_koef = fft.fft(T[i])
		#F_k = fft.fftshift(F_koef)
		#check fourie
		#print('real', sum(np.abs(np.real(F_koef))))
		#print('iamg', sum(np.abs(np.imag(F_koef))))

		#calculate k_frequenyc!!!

		F_koef_next = np.copy(F_koef)
		for j in range(N):
			k_freq = j/a/N/2
			F_koef_next[j] = F_koef[j]*(1-t_step*D*4*np.pi**2*k_freq**2)

		T[i+1] = fft.ifft(F_koef_next)

		#RP periodicni
		#RP = T[i+1][-1] / 2 + T[i+1][0] / 2
		#T[i+1][0] = T[i+1][-1] = RP

		#check energy
		if i % 20 == 0:
			print(sum(T[i]))


	plt.imshow(T, cmap='inferno')
	plt.show()

f_PRP()
