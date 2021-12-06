import numpy as np
import matplotlib.pyplot as plt

def moving_average(loss):
    avged_loss = np.zeros((len(loss) - 50))
    for i in range(len(avged_loss)):
        avged_loss[i] = np.mean(loss[i:i+50])
    return avged_loss

def avg_val(loss):
    avged_loss = np.zeros((82))
    for i in range(len(avged_loss)):
        avged_loss[i] = np.mean(loss[i*178:(i+1)*178])
    return avged_loss
loss = np.load("./train_result/total_train.npy")
avged_loss = moving_average(loss)
plt.plot(avged_loss)
plt.title("Moving Averaged Total Loss of Train")
plt.xlabel("Training Step")
plt.ylabel("Total Loss")
plt.savefig("training_total.png")
plt.close()

loss = np.load("./train_result/c_train.npy")
avged_loss = moving_average(loss)
plt.plot(avged_loss)
plt.title("Moving Averaged Category Loss of Train")
plt.xlabel("Training Step")
plt.ylabel("Category Loss")
plt.savefig("training_c.png")
plt.close()

loss = np.load("./train_result/r_train.npy")
avged_loss = moving_average(loss)
plt.plot(avged_loss)
plt.title("Moving Averaged Regressor Loss of Train")
plt.xlabel("Training Step")
plt.ylabel("Regressor Loss")
plt.savefig("training_r.png")
plt.close()

loss = np.load("./train_result/total_val.npy")
avged_loss = avg_val(loss)
plt.plot(avged_loss)
plt.title("Moving Averaged Total Loss of Validation")
plt.xlabel("Validation Step")
plt.ylabel("Total Loss")
plt.savefig("val_total.png")
plt.close()

loss = np.load("./train_result/c_val.npy")
avged_loss = avg_val(loss)
plt.plot(avged_loss)
plt.title("Moving Averaged Category Loss of Validation")
plt.xlabel("Validation Step")
plt.ylabel("Category Loss")
plt.savefig("val_c.png")
plt.close()

loss = np.load("./train_result/r_val.npy")
print(loss.shape)
avged_loss = avg_val(loss)
plt.plot(avged_loss)
plt.title("Moving Averaged Regressor Loss of Validation")
plt.xlabel("Validation Step")
plt.ylabel("Regressor Loss")
plt.savefig("val_r.png")
plt.close()
