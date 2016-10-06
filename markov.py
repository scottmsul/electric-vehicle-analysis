import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import util

class EVMarkovModel:

  def __init__(self, usages, when_charging, normed=True):
    self.clf0 = QuadraticDiscriminantAnalysis(store_covariances=True)
    self.clf1 = QuadraticDiscriminantAnalysis(store_covariances=True)
    self.when_charging = when_charging
    if normed:
      usages = util.normalize_usages(usages)
    self.usages = usages
    derivatives = util.derivatives(usages)

    # since the derivatives act as a "4D" vector, normalize each dimension so they have the same weights
    self.mean = [np.average(d.values) for d in derivatives]
    self.std = [np.std(d.values) for d in derivatives]
    self.derivatives = [(derivatives[i] - self.mean[i]) / self.std[i] for i in range(len(derivatives))]

#     for i,d in enumerate(derivatives):
#         mean = np.average(d.values)
#         std = np.std(d.values)
#         d = (d - mean) / std
#         derivatives[i] = d
#     self.derivatives = derivatives


  def train(self, train_houses, normed=True):
    train_derivs = [d[train_houses] for d in self.derivatives]
    train_when_charging = self.when_charging[train_houses]

    # get transitions at each timestep
    back = train_when_charging[:-1]
    back.index = range(1, len(back.index)+1)
    front = train_when_charging[1:]
    
    n00 = (back == 0) & (front == 0)
    n01 = (back == 0) & (front == 1)
    n10 = (back == 1) & (front == 0)
    n11 = (back == 1) & (front == 1)
    transition_times = [n00, n01, n10, n11]
    
    # put derivatives in terms of X (a 4D array at each timestep) and y (label at each timestep)
    T00 = 0
    T01 = 1
    T10 = 2
    T11 = 3
    transition_labels = [T00, T01, T10, T11]

    train_X = []
    train_y = []

    for (label, times) in zip(transition_labels, transition_times):    
      curr_train_X = np.array([util.get_all(d, times[train_houses]) for d in train_derivs]).T
      curr_train_y = np.array([label] * len(curr_train_X))
      train_X.extend(curr_train_X)
      train_y.extend(curr_train_y)
    
    train_X = np.array(train_X)
    train_y = np.array(train_y)

    X0 = train_X[(train_y==0) | (train_y==1)]
    y0 = train_y[(train_y==0) | (train_y==1)]
    
    X1 = train_X[(train_y==2) | (train_y==3)]
    y1 = train_y[(train_y==2) | (train_y==3)]

    self.clf0.fit(X0, y0)
    self.clf1.fit(X1, y1)

  def transition_prob(self, x, initial_y, final_y):
      x = [x]
      if (initial_y, final_y) == (0,0):
          p = self.clf0.predict_proba(x)[0][0]
      elif (initial_y, final_y) == (0,1):
          p = self.clf0.predict_proba(x)[0][1]
      elif (initial_y, final_y) == (1,0):
          p = self.clf1.predict_proba(x)[0][0]
      else:
          p = self.clf1.predict_proba(x)[0][1]
      return p

  def run(self, house):
  
      p0s = [1.0]
      p1s = [0.0]
  
      p0 = 1.0
      p1 = 0.0

      for x in np.array([d[house] for d in self.derivatives]).T[1:]:
          p0 = p0 * self.transition_prob(x, 0, 0) + p1 * self.transition_prob(x, 1, 0)
          p1 = p0 * self.transition_prob(x, 0, 1) + p1 * self.transition_prob(x, 1, 1)
          # normalize (shouldn't drift, but in case it does for some reason)
          s = p0 + p1
          p0 = p0 / s
          p1 = p1 / s
          
          p0s.append(p0)
          p1s.append(p1)
      return [np.array(p0s), np.array(p1s)]

  def run_usage(self, usage):
    start = np.argwhere(~np.isnan(usage))[0][0]
    end = np.argwhere(~np.isnan(usage))[-1][0]
    curr_usage = usage[start:(end+1)]
    normed_usage = util.normalize_single_usage(curr_usage)
    derivatives = util.single_derivatives(normed_usage)
    derivatives = [(derivatives[i] - self.mean[i]) / self.std[i] for i in range(len(derivatives))]
    derivatives = [d.values for d in derivatives]

    p0s = [1.0]
    p1s = [0.0]

    p0 = 1.0
    p1 = 0.0

    for x in np.array([d for d in derivatives]).T[1:]:
      p0 = p0 * self.transition_prob(x, 0, 0) + p1 * self.transition_prob(x, 1, 0)
      p1 = p0 * self.transition_prob(x, 0, 1) + p1 * self.transition_prob(x, 1, 1)
      # normalize (shouldn't drift, but in case it does for some reason)
      s = p0 + p1
      p0 = p0 / s
      p1 = p1 / s
      
      p0s.append(p0)
      p1s.append(p1)
    preds = np.array([np.nan]*len(usage))

    for i,t in enumerate(range(start, end+1)):
      preds[t] = p1s[i]
    
    return preds
