import numpy as np 
import matplotlib.pyplot as plt
import statistics as st
import math


def gaussian(A,x, mu, sig):
    return A*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

n=1000   # number of pixels
#m=[1]
#m=[1,10,35,50]
m=[1,2,3,5,25,100,200,300,500,1000]    #number of stacks
A=200     #signal strength
sigma=3 #noise strength


noise=np.random.normal(0,sigma,n)
signal=[]
SNR=[]


for i in range(len(noise)):

    signal.append(gaussian(A,float(i),n/2,n/10)+noise[i])

#init x
plt.plot(range(0,n),signal)
plt.title("single frame", fontsize=16)
plt.xlabel("pixel number", fontsize=16)
plt.ylabel("Intensity", fontsize=16)
plt.show()

##############################################################################
i2=1 #for subplots
for i in m: 
        signal=[0]*n
        averaged_signal=[0]*n

                    # i number of total stacks

    
        for ii in range(i):
                noise=np.random.normal(0,sigma,n)
                for iii in range(len(noise)):
                        signal[iii]+=((gaussian(A, float(iii),n/2,n/10)+noise[iii])) # stacking iii pixels ii times
      


        for iiii in range(len(signal)):
                averaged_signal[iiii]=signal[iiii]/i

        plt.subplot(math.ceil(len(m)/2),2,i2)
        plt.plot(range(0,n),averaged_signal)
        max_signal=max(averaged_signal)
        noise_level=st.stdev(averaged_signal[: int(n/10)])
        #print(averaged_signal[: int(n/10)])
        #noise_level=st.pstdev(averaged_signal[: int(n/8)], averaged_signal[int(7*n/8) :])
        print("Max",max_signal)
        print("noise_level",noise_level)
        print("SNR",max_signal/noise_level)
        plt.ylim(-3*sigma,max(1.2*A,3*sigma))
        #plt.title(i,"Frames")

        SNR.append(max_signal/noise_level)
        plt.text(0.7*n,0.9*max(averaged_signal),"SNR="+str(round(max_signal/noise_level,2))+"",fontsize=16)
        plt.xlabel("pixel number", fontsize=16)
        plt.ylabel("Intensity", fontsize=16)
        plt.title("Number of stacks: "+str(i)+"", fontsize=16)
        i2+=1

plt.savefig("output.pdf")
plt.show()




plt.plot(m,SNR)
plt.xlabel("Number of stacks", fontsize=16)
plt.ylabel("Signal to Noise ratio (SNR)", fontsize=16)
plt.savefig("SNR.pdf")
plt.show()