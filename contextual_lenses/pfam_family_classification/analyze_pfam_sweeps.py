"""Analyze and plot pfam train samples sweeps."""


import pandas as pd

import matplotlib.pyplot as plt


def sweep_plot(df, families):

  df = df[df['families']==families]
  df_rand = df[df['pretrain']==0]
  df_pretrain = df[df['pretrain']==1]

  plt.figure(figsize=(8, 6))

  plt.plot(df_rand['train_samples'].values[:-1], df_rand['accuracy'].values[:-1], label='Random')
  plt.scatter(df_rand['train_samples'].values[:-1], df_rand['accuracy'].values[:-1])
  plt.plot([1, 100], [df_rand['accuracy'].values[-1]]*2, label='Random Threshold', linestyle='dashed', color='b')

  plt.plot(df_pretrain['train_samples'].values[:-1], df_pretrain['accuracy'].values[:-1], label='Pretrain')
  plt.scatter(df_pretrain['train_samples'].values[:-1], df_pretrain['accuracy'].values[:-1])
  plt.plot([1, 100], [df_pretrain['accuracy'].values[-1]]*2, label='Pretrain Threshold', linestyle='dashed', color='orange')

  plt.xlabel('Train Samples')
  plt.ylabel('Accuracy')
  plt.title('Accuracy vs. Train Samples for Families ' + families)
  plt.xticks([1, 5, 10, 25, 50, 75, 100])
  plt.yticks([0.1*i for i in range(11)])
  plt.legend()
  plt.savefig('pfam_figures/accuracy_on_families_' + families)
  plt.show()


def main():

  df = pd.read_csv('samples_sweep.csv')

  family_names = ['1-100', '101-200', '1-200']
  for families in family_names:
    sweep_plot(df, families)


if __name__ == "__main__":
  main()
