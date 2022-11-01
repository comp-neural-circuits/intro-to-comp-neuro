# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: intro-to-comp-neuro
#     language: python
#     name: intro-to-comp-neuro
# ---

# # Neural encoding 1

TODO - include figure here -> spikes - neuron - LIF - spikes

# +
import io
import requests
import numpy as np
import matplotlib.pyplot as plt

# Settings for the figures
plt.style.use(plt.style.available[20])
plt.style.use("https://github.com/comp-neural-circuits/intro-to-comp-neuro/raw/dev/plots_style.txt")
# -

# #### Data Retrieval - Exploring the Steinmetz dataset
# In this tutorial we will explore the structure of a neuroscience dataset.
#
# We consider a subset of data from a study of [Steinmetz et al. (2019)](https://www.nature.com/articles/s41586-019-1787-x). In this study, Neuropixels probes were implanted in the brains of mice. Electrical potentials were measured by hundreds of electrodes along the length of each probe. Each electrode’s measurements captured local variations in the electric field due to nearby spiking neurons. A spike sorting algorithm was used to infer spike times and cluster spikes according to common origin: a single cluster of sorted spikes is causally attributed to a single neuron.
#
# In particular, a single recording session of spike times and neuron assignments was loaded and assigned to spike_times in the preceding setup.
#
# Typically a dataset comes with some information about its structure. However, this information may be incomplete. You might also apply some transformations or “pre-processing” to create a working representation of the data of interest, which might go partly undocumented depending on the circumstances. In any case it is important to be able to use the available tools to investigate unfamiliar aspects of a data structure.
#
# Let’s see what our data looks like…

r = requests.get('https://osf.io/sy5xt/download')
if r.status_code != 200:
  print('Failed to download data')
else:
  spike_times = np.load(io.BytesIO(r.content), allow_pickle=True)['spike_times']

type(spike_times)




