"""Plot the training and validation loss from pickle file.

   train_model.py saves trainHistory.p with the training and validationloss.
   This is then loaded and visualized in a plot.
"""
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

training_history = pickle.load(open("trainHistory.p", "rb"))
loss = training_history['loss']
val_loss = training_history['val_loss']

# sns.set()
sns.set_context("paper")  # paper, notebook, talk, and poster
sns.set_style("dark")

plt.plot(training_history['loss'])
plt.plot(training_history['val_loss'])
plt.title('Model MSE loss')
plt.ylabel('MSE loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
