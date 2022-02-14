import numpy as np
from sklearn.model_selection import train_test_split


x_train_bg = np.load("x_background_train.npy")
x_train_fg = np.load("x_foreground_train.npy")

m_bg = np.mean(x_train_bg,axis=0)

x_test_bg = np.load("x_background_eval.npy")
x_test_fg = np.load("x_foreground_eval.npy")

y_train_bg = np.load("y_background_train.npy")
y_train_fg = np.load("y_foreground_train.npy")

y_test_bg = np.load("y_background_eval.npy")
y_test_fg = np.load("y_foreground_eval.npy")

m_bg = np.mean(x_train_bg,axis=0)
x_train_bg = x_train_bg - m_bg
x_test_bg = x_test_bg - m_bg

m_fg = np.mean(x_train_fg,axis=0)
x_train_fg = x_train_fg - m_fg
x_test_fg = x_test_fg - m_fg

x_train = []
y_train = []

x_train.append(x_train_bg)
x_train.append(x_train_fg)
y_train.append(y_train_bg)
y_train.append(y_train_fg)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.swapaxes(x_train, 0, 1)
y_train = np.swapaxes(y_train, 0, 1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=5)

x_train = np.swapaxes(x_train, 0, 1)
y_train = np.swapaxes(y_train, 0, 1)
x_val = np.swapaxes(x_val, 0, 1)
y_val = np.swapaxes(y_val, 0, 1)


x_test = []
y_test = []

x_test.append(x_test_bg)
x_test.append(x_test_fg)
y_test.append(y_test_bg)
y_test.append(y_test_fg)

x_test = np.array(x_test)
y_test = np.array(y_test)


np.save("x_train_rpca.npy", x_train)
np.save("y_train_rpca.npy", y_train)
np.save("x_val_rpca.npy", x_val)
np.save("y_val_rpca.npy", y_val)
np.save("x_test_rpca.npy", x_test)
np.save("y_test_rpca.npy", y_test)
