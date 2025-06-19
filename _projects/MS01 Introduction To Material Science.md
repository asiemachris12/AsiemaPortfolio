---
layout: page
title: 1. Introduction To Material Science
description: a project with a background image
img: assets/img/1.jpg
importance: 1
category: Materials Science
---

## Table of Contents:
- [Table of Contents:](#table-of-contents)
- [Introduction ](#introduction-)
- [Featurizers ](#featurizers-)
  - [Material Structure Featurizers ](#material-structure-featurizers-)
    - [Example: Featurizing a crystal](#example-featurizing-a-crystal)
- [|  pbc   | True True True   |](#--pbc----true-true-true---)

---

## Introduction <a class="anchor" id="introduction"></a>

One of the most exciting applications of machine learning in the recent time is it's application to material science domain. DeepChem helps in development and application of machine learning to solid-state systems. As a starting point of applying machine learning to material science domain, DeepChem provides material science datasets as part of the MoleculeNet suite of datasets, data featurizers and implementation of popular machine learning algorithms specific to material science domain. This tutorial serves as an introduction of using DeepChem for machine learning related tasks in material science domain.

Traditionally, experimental research were used to find and characterize new materials. But traditional methods have high limitations by constraints of required resources and equipments. Material science is one of the booming areas where machine learning is making new in-roads. The discovery of new material properties holds key to lot of problems like climate change, development of new semi-conducting materials etc. DeepChem acts as a toolbox for using machine learning in material science.

---

```python
!pip install --pre deepchem
```

---

DeepChem for material science will also require the additional libraries [`pymatgen`](https://pymatgen.org/) and [`matminer`](https://hackingmaterials.lbl.gov/matminer/). These two libraries assist machine learning in material science. For graph neural network models which we will be used in the backend, DeepChem requires [`dgl`](https://www.dgl.ai/) library. All these can be installed using `pip`. **Note** when using locally, install a higher version of the jupyter notebook (>6.5.5, here on colab).

---

```python
!pip install -q pymatgen==2023.12.18
!pip install -q matminer==0.9.0
!pip install -q dgl
!pip install -q tqdm
```

---

```python
import deepchem as dc
dc.__version__
```

---

```python
from tqdm import tqdm
import matplotlib.pyplot as plt

import pymatgen as mg
from pymatgen import core as core

import os
os.environ['DEEPCHEM_DATA_DIR'] = os.getcwd()
```

---

## Featurizers <a class="anchor" id="featurizers"></a>

### Material Structure Featurizers <a class="anchor" id="crystal-featurizers"></a>

Crystal are geometric structures which has to be featurized for using in machine learning algorithms.  The following featurizers provided by DeepChem helps in featurizing crystals:

 The [SineCoulombMatrix](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#sinecoulombmatrix) featurizer a crystal by calculating sine coulomb matrix for the crystals. It can be called using `dc.featurizers.SineCoulombMatrix` function. [1]

 The [CGCNNFeaturizer](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#cgcnnfeaturizer) calculates structure graph features of crystals. It can be called using `dc.featurizers.CGCNNFeaturizer` function. [2]

 The [LCNNFeaturizer](https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html#lcnnfeaturizer) calculates the 2-D Surface graph features in 6 different permutations. It can be used using the utility `dc.feat.LCNNFeaturizer`. [3]

[1] Faber et al. “Crystal Structure Representations for Machine Learning Models of Formation Energies”, Inter. J. Quantum Chem. 115, 16, 2015. [https://arxiv.org/abs/1503.07406](https://arxiv.org/abs/1503.07406)

[2] T. Xie and J. C. Grossman, “Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties”, Phys. Rev. Lett. 120, 2018, https://arxiv.org/abs/1710.10324

[3] Jonathan Lym, Geun Ho Gu, Yousung Jung, and Dionisios G. Vlachos, Lattice Convolutional Neural Network Modeling of Adsorbate Coverage Effects, J. Phys. Chem. C 2019

---

#### Example: Featurizing a crystal

In this part, we will be using `pymatgen` for representing the crystal structure of Caesium Chloride and calculate structure graph features using `CGCNNFeaturizer`.

The `CsCl` crystal is a cubic lattice with the chloride atoms lying upon the lattice points at the edges of the cube, while the caesium atoms lie in the holes in the center of the cubes. The green colored atoms are the caesium atoms in this crystal structure and chloride atoms are the grey ones.

<img src="https://github.com/deepchem/deepchem/blob/master/examples/tutorials/assets/CsCl_crystal_structure.png?raw=1">

Source: [Wikipedia](https://en.wikipedia.org/wiki/Caesium_chloride)

---

```python
from tqdm import tqdm
import matplotlib.pyplot as plt

import pymatgen as mg
from pymatgen import core as core

import os
os.environ['DEEPCHEM_DATA_DIR'] = os.getcwd()
```

---

In above code sample, we first defined a cubic lattice using the cubic lattice parameter `a`. Then, we created a structure with atoms in the crystal and their coordinates as features. A nice introduction to crystallographic coordinates can be found [here](https://www.youtube.com/watch?v=dP3LjWtoeMU). Once a structure is defined, it can be featurized using CGCNN Featurizer. Featurization of a crystal using `CGCNNFeaturizer` returns a DeepChem GraphData object which can be used for machine learning tasks.

---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/1.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/3.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Caption photos easily. On the left, a road goes through a tunnel. Middle, leaves artistically fall in a hipster photoshoot. Right, in another hipster photoshoot, a lumberjack grasps a handful of pine needles.
</div>
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/5.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    This image can also have a caption. It's like magic.
</div>

You can also put regular text between your rows of images.
Say you wanted to write a little bit about your project before you posted the rest of the images.
You describe how you toiled, sweated, _bled_ for your project, and then... you reveal its glory in the next row of images.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-4 mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    You can also have artistically styled 2/3 + 1/3 images, like these.
</div>

The code is simple.
Just wrap your images with `<div class="col-sm">` and place them inside `<div class="row">` (read more about the <a href="https://getbootstrap.com/docs/4.4/layout/grid/">Bootstrap Grid</a> system).
To make images responsive, add `img-fluid` class to each; for rounded corners and shadows use `rounded` and `z-depth-1` classes.
Here's the code for the last row of images above:

{% raw %}

```html
<div class="row justify-content-sm-center">
  <div class="col-sm-8 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/6.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/11.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
```

{% endraw %}
