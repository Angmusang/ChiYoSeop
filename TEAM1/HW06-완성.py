import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from lmfit import Model
from sklearn.metrics import r2_score
import time

tree = ET.parse('C:/Users/gunwo/PycharmProjects/pythonProject4/TEAM1/HW05.xml')
root = tree.getroot()

# I-IL Graph

def spfl(a):    # spfl 함수 정의
    sp = a.text.split(',')  # ,를 기준으로 나누고 값 가져오기
    fl = list(map(float, sp))   # 가져온 값을 실수로 바꾸고 리스트에 넣기
    return fl   # fl 반환

wvl = []
itst = []

for data in root.iter('L'):
    L = spfl(data)
    wvl.append(L)

for data in root.iter('IL'):
    IL = spfl(data)
    itst.append(IL)

name = []
for data in root.iter("WavelengthSweep"):
    name.append(data.get("DCBias"))

plt.figure(figsize=(12,8))
plt.subplot(2, 3, 1)

for n in range(len(wvl)):

    plt.title("Transmission spectra-as measured")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Measured transmission [Bm]")
    plt.rc("legend", fontsize = 7)

    if n == 6 :
     plt.plot(wvl[6], itst[6], label='DCBias = REF')
    else :
     plt.plot(wvl[n], itst[n], label="DCBias ={}V".format(name[n]))
    plt.plot(loc = 'best' , ncol = 3)

# I-IL Graph fitting 값

plt.subplot(2, 3, 2)
for n in range(len(wvl)):
    if n == 6 :
        plt.plot(wvl[n], itst[n], label = "REF")
    else :
        continue

dp1 = np.polyfit(wvl[6], itst[6], 3)
f1 = np.poly1d(dp1)
plt.plot(wvl[6], f1(wvl[6]), 'r--', label = 'Fit ref polynomial')
plt.xlabel('Wavelength[nm]')
plt.ylabel('Transmissions[dB]')
plt.title('Transmission spectra - fitted')
plt.legend(loc='best')
plt.rc("legend", fontsize=7)

print(r2_score((itst[6]),f1(wvl[6]))) # 라벨로 무언가 띄워 놓기!


# 피팅후 뺸값
plt.subplot(2, 3, 3)

for k in range(len(wvl)-1):
    plt.title("Fitting Function")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Measured transmission [dB]")
    plt.plot(wvl[k], itst[k] - f1(wvl[k]), label = f'DCBias = {name[k]}V')
    plt.legend(loc = 'best', ncol = 3)

plt.tight_layout()
plt.show()

# I-V 그래프 피팅
plt.subplot(2, 3, 4)
for data in root.iter('Voltage'):
    vlt = spfl(data)                                                # 'Voltage'안에 있는 값을 spfl함수를 사용해 v에 저장
    v = np.array(vlt)
for data in root.iter('Current'):
    crt = list(map(abs, spfl(data)))                                #'Current'안에 있는 값을 spfl함수를 사용하고, 절댓값을 사용해 리스트 안에 넣어
    i = np.array(crt)

def IV(x, Is, q, n, k):
    return Is * (exp((q * x) / (n * k)) - 1)

def eq(x, a, b, c, d, e):
    return a * (x**4) + b * (x**3) + c * (x**2) + d * x + e

v1 = v[:10]
v2 = v[9:]

i1 = i[:10]
i2 = i[9:]

lmodel = Model(eq)
rmodel = Model(IV)
params1 = lmodel.make_params(a=1, b=1, c=1, d=1, e=1)
params2 = rmodel.make_params(Is=1, q=1, n=1, k=1)

result1 = lmodel.fit(i1, params1, x = v1)
result2 = rmodel.fit(i2, params2, x = v2)

plt.plot(v, i, 'o', label = 'I-V curve')
plt.title("IV analysis")
plt.xlabel("Voltage [V]")
plt.ylabel("Current [A]")
plt.yscale('logit')
plt.legend(loc='best')

start_time1 = time.time()
plt.plot(v1, result1.best_fit, '--', label = 'Fit-l')
end_time1 = time.time()
print(f'left fitting time : {end_time1 - start_time1}')

start_time2 = time.time()
plt.plot(v2, result2.best_fit, '--', label = 'Fit-r')
end_time2 = time.time()
print(f'right fitting time : {end_time2 - start_time2}')
print(f'left : {r2_score(i1, result1.best_fit)}')
print(f'right : {r2_score(i2, result2.best_fit)}')


plt.show()




