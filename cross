import numpy as np
import matplotlib.pyplot as plt
y_true = np.array([0, 1, 0, 1])      
y_pred = np.array([0.2, 0.8, 0.4, 0.6])  
y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
plt.plot(range(len(y_true)), loss)
plt.show()






