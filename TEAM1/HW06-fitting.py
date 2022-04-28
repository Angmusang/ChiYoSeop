import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from lmfit import Model
from sklearn.metrics import r2_score


tree = ET.parse('C:/Users/gunwo/PycharmProjects/pythonProject4/TEAM1/HW05.xml')
root = tree.getroot()

# I-V Graph
for data in root.iter('Voltage'):
    print("{}: {}".format(data.tag,data.text))
    a = data.text.split(',')
    b = list(map(float,a))
    # Vd의 값

for data in root.iter('Current'):
    print(f'{data.tag}: {data.text}')
    c = data.text.split(',')
    d = list(map(float,c))
    d = list(map(abs,d))
    # Is의 값

plt.figure(figsize = (10,5))
plt.subplot(1,2,1) #그래프를 그리는 위치 알려주는것
plt.title("IV analysis")
plt.plot(b, d, 'bo-',label = 'I-V curve')
plt.xlabel("Voltage [V]")
plt.ylabel("Current [A]")
plt.legend(loc = ('best'))
plt.yscale('log')

first = np.polyfit(b, d, 12)
second = np.polyval(first, b)
plt.plot(b, second)
plt.show()



def Fitting(locs, info, callback=None):
    bin_centers, dnfl_ = next_frame_neighbor_distance_histogram(locs, callback)

    def func(d, a, s, ac, dc, sc):
        f = a * (d / s ** 2) * _np.exp(-0.5 * d ** 2 / s ** 2)
        fc = (ac * (d / sc ** 2)* _np.exp(-0.5 * (d ** 2 + dc ** 2) / sc ** 2)* _iv(0, d * dc / sc))
        return f + fc

    pdf_model = _lmfit.Model(func)
    params = _lmfit.Parameters()
    area = _np.trapz(dnfl_, bin_centers)
    median_lp = _np.mean([_np.median(locs.lpx), _np.median(locs.lpy)])
    params.add("a", value=area / 2, min=0)
    params.add("s", value=median_lp, min=0)
    params.add("ac", value=area / 2, min=0)
    params.add("dc", value=2 * median_lp, min=0)
    params.add("sc", value=median_lp, min=0)
    result = pdf_model.fit(dnfl_, params, d=bin_centers)
    return result, result.best_values["s"]

plt.show()

## 2번째 시도

import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ExpressionModel

x = np.linspace(-10, 10, 201)
amp, cen, wid = 3.4, 1.8, 0.5

y = amp * np.exp(-(x-cen)**2 / (2*wid**2)) / (np.sqrt(2*np.pi)*wid)
np.random.seed(2021)
y = y + np.random.normal(size=x.size, scale=0.01)

gmod = ExpressionModel("amp * exp(-(x-cen)**2 /(2*wid**2))/(sqrt(2*pi)*wid)")
result = gmod.fit(y, x=x, amp=5, cen=5, wid=1)

print(result.fit_report())
