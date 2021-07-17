import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
import Linear
x = np.arange(21)
d1,a1 = Linear.mod1()
y1 = a1*x+d1
d2,a2 = Linear.mod2()
y2 = a2*x+d2

plt.plot(x,y1,label="Supply")
plt.plot(x,y2,label="Demand")

plt.title("Econmoy Model",fontsize=15)
plt.xlabel("Quantity",fontsize=13)
plt.ylabel("Price",fontsize=13)
plt.xlim(-1,20)
plt.ylim(-1,20)
plt.legend()
plt.show()