import numpy as np#This code takes about 15 seconds to run.thanks for your patience! 
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa
def toBinary(myNum):
 sample=[0]*8
 if(myNum==0):
    return sample
 sgn=myNum/abs(myNum)
 u=abs(myNum)
 for j in range (0,8):
    if(u%2==0):
     sample[j]=0
    else:
     sample[j]=1
    u=u//2
 if(sgn!=1):
    t=0
    while(sample[t]==0):
        t=t+1
    for r in range (t+1,8):
        sample[r]=1-sample[r]
 return sample

def binaryToDecimal(x): #takes an an 8 bit array and returns the decimal value of that base_2 presentation
 y=[0]*8
 for w in range(0,8):
   y[w]=x[7-w]#reversing the order of x
 t=1
 ans=0
 for i in range (0,8):
  ans=ans+t*y[i]
  t=t*2
 return ans

def adaptive_lms(d,x,mu,M):
    # x is refrence signal noise
    # d is the input of signal that is noisy beacuse of power line noise
    #mu is the learning rate 
    #M is the memory of filter
    # y is the out put of the filter which is the input with less noise
    # you can find more details and explantion in the report file
    N = len(d)
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    for i in range(M, N):
        x_1 = np.flip(x[i - M:i])
        y_n = 0.0
        for j in range(M):
            y_n += w[j] * x_1[j]
        e[i] = d[i] - y_n
        y[i] = e[i]
        for j in range(M):
            w[j] += 2* mu*e[i]*x_1[j]
    return y 
def ASCIIDecoder(temp):
  str=""
  for i in range(0,len(temp)):
    str=str+chr(temp[i])
  return str
signal_with_noise,sample_rate1=librosa.load("C://Users//LENOVO//Desktop//Daneshgah//Signal//Group2//Recordings//100-2625-noisy.wav", sr=None) #librosa library loads the signal normalized
'''
fft_data = np.fft.fft(signal_with_noise)
N=len(signal_with_noise)
freqs = np.fft.fftfreq(N, 1/sample_rate1)
for i in range(0,len(freqs)):
   if(abs(fft_data[i])>7000):
      print(freqs[i])
plt.plot(freqs[0:N//2],fft_data[0:N//2])
plt.show() #this code is to find out the frequency of the noise.after running it we see that the noise is in 1500 and 2500 Hz frequencies

'''

t = np.arange(len(signal_with_noise))/sample_rate1
ref = np.sin(2*np.pi * 1500*t)
ref2 = np.sin(2*np.pi * 2500*t)
# filtering process
output = adaptive_lms(signal_with_noise, ref,0.00005,32) #here we remove 1500Hz noise
output=adaptive_lms(output,ref2,0.00005,32)#here we remove 2500Hz noise
attenuated_sig=100*output #attenuated signal has float elements between -100,100.
for i in range (0,len(attenuated_sig)):
  attenuated_sig[i]=np.round(attenuated_sig[i]) #here we round elements so they be integers.
mylen=len(attenuated_sig)/2625
mylen=int(mylen)

decoded_signal=[0]*mylen
for i in range (0,len(decoded_signal)):
  decoded_signal[i]=attenuated_sig[i*2625+2624] #Decoded_signal picks the elements we care about,that is 2625*n_th elements of the attenuated signal. 
for i in range (0,len(decoded_signal)):
  if(decoded_signal[i]%2==0):
   decoded_signal[i]=0
  else:
   decoded_signal[i]=1 #We only need the last digit of each element,which is 0 if the number is even and 1 if it is odd.

mylen=len(decoded_signal)//8
extractedChars=[0]*mylen
for i in range (0,mylen):
  temp=decoded_signal[8*i:8*i+8] #We split the bits into 8tuple categories;so each category is the base_2 representation of a number in range 0-255;
  extractedChars[i]=binaryToDecimal(temp) #which is the ASCII code of a character.

#Finally,we use our function to convert ASCII codes into human language
print(ASCIIDecoder(extractedChars))
