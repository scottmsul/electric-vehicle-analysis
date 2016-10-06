import pandas as pd
import numpy as np
import random
import scipy.interpolate as interpolate

def split_data(data, ratios=[0.8, 0.2]):
    ind_sets = []
    num_points = len(data)
    remaining_inds = range(num_points)
    for r in ratios:
        curr_num = int(num_points * r)
        new_inds = random.sample(remaining_inds, curr_num)
        ind_sets.append(new_inds)
        remaining_inds = list(set(remaining_inds) - set(new_inds))
    split_data = [data[inds] for inds in ind_sets]
    return split_data

def normalize_usages(usages):
  mean = usages.apply(np.average)
  std = usages.apply(np.std)
  usages = (usages - mean) / std
  return usages

def normalize_single_usage(usage):
  mean = np.average(usage)
  std = np.std(usage)
  usage = (usage - mean) / std
  return usage

def derivatives(usages):
  houses = usages.columns
  raw_splines = {}
  xs = np.linspace(0, len(usages.index)-1, len(usages.index)).astype(float)
  for house in houses:
      raw_splines[house] = interpolate.UnivariateSpline(xs, usages[house], k=3, s=0)
  
  # get the derivatives at each timestep
  d0 = pd.DataFrame(index=usages.index)
  d1 = pd.DataFrame(index=usages.index)
  d2 = pd.DataFrame(index=usages.index)
  d3 = pd.DataFrame(index=usages.index)
  for house in houses:
    d0[house] = [raw_splines[house].derivatives(x)[0] for x in xs]
    d1[house] = [raw_splines[house].derivatives(x)[1] for x in xs]
    d2[house] = [raw_splines[house].derivatives(x)[2] for x in xs]
    d3[house] = [raw_splines[house].derivatives(x)[3] for x in xs]
  derivatives = [d0, d1, d2, d3]
  return derivatives

def single_derivatives(usage):
  xs = np.linspace(0, len(usage.index)-1, len(usage.index)).astype('float')
  raw_splines = interpolate.UnivariateSpline(xs, usage, k=3, s=0)
  d0 = pd.Series([raw_splines.derivatives(x)[0] for x in xs], index=usage.index)
  d1 = pd.Series([raw_splines.derivatives(x)[1] for x in xs], index=usage.index)
  d2 = pd.Series([raw_splines.derivatives(x)[2] for x in xs], index=usage.index)
  d3 = pd.Series([raw_splines.derivatives(x)[3] for x in xs], index=usage.index)
  derivatives = [d0, d1, d2, d3]
  return derivatives


def transitions_times(when_charging):
  back = when_charging[:-1]
  back.index = range(1, len(back.index)+1)
  front = when_charging[1:]
  
  n00 = (back == 0) & (front == 0)
  n01 = (back == 0) & (front == 1)
  n10 = (back == 1) & (front == 0)
  n11 = (back == 1) & (front == 1)

  transition_times = [n00, n01, n10, n11]
  return transition_times

def remove_nan(x):
    return x[~np.isnan(x)]
  
def get_all(x, index):
  return remove_nan(x[index].values.flatten())

def acc_info(pred, true):

    tn = float(sum((pred == true) & (pred == 0)))
    tp = float(sum((pred == true) & (pred == 1)))
    fn = float(sum((pred != true) & (pred == 0)))
    fp = float(sum((pred != true) & (pred == 1)))
    tn_unc = max(1, np.sqrt(tn))
    tp_unc = max(1, np.sqrt(tp))
    fn_unc = max(1, np.sqrt(fn))
    fp_unc = max(1, np.sqrt(fp))
    
    pos_acc = tp / (tp + fn)
    pos_acc_unc = np.sqrt( ((fn/(tp+fn)**2) * tp_unc)**2 + ((tp/(tp+fn)**2) * fn_unc)**2 )
    
    neg_acc = tn / (tn + fp)
    neg_acc_unc = np.sqrt( ((fp/(tn+fp)**2) * tn_unc)**2 + ((tn/(tn+fp)**2) * fp_unc)**2 )
    
    t = tp + tn
    t_unc = np.sqrt( tp_unc**2 + tn_unc**2 )
    
    f = fn + fp
    f_unc = np.sqrt( fn_unc**2 + fp_unc**2 )
    
    acc = t / (t + f)
    acc_unc = np.sqrt( ((f/(t+f)**2) * t_unc)**2 + ((t/(t+f)**2) * f_unc)**2 )
    
    info = {}
    info['tn'] = tn
    info['tp'] = tp
    info['fn'] = fn
    info['fp'] = fp
    info['tn_unc'] = tn_unc
    info['tp_unc'] = tp_unc
    info['fn_unc'] = fn_unc
    info['fp_unc'] = fp_unc
    info['pos_acc'] = pos_acc
    info['pos_acc_unc'] = pos_acc_unc
    info['neg_acc'] = neg_acc
    info['neg_acc_unc'] = neg_acc_unc
    info['t'] = t
    info['t_unc'] = t_unc
    info['f'] = f
    info['f_unc'] = f_unc
    info['acc'] = acc
    info['acc_unc'] = acc_unc
    return info

def acc_str(acc_info, num_digits=4):
  s1 = "acc: %%.%if (%%.%if), pos acc: %%.%if (%%.%if), neg acc: %%.%if (%%.%if)" % (num_digits, num_digits, num_digits, num_digits, num_digits, num_digits)
  s2 = s1 % (acc_info['acc'], acc_info['acc_unc'], acc_info['pos_acc'], acc_info['pos_acc_unc'], acc_info['neg_acc'], acc_info['neg_acc_unc'])
  return s2
