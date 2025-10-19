
==============================================
gensim -- Topic Modelling in Python
==============================================

|GA|_
|Wheel|_

.. |GA| image:: https://github.com/RaRe-Technologies/gensim/actions/workflows/tests.yml/badge.svg?branch=develop
.. |Wheel| image:: https://img.shields.io/pypi/wheel/gensim.svg

.. _GA: https://github.com/RaRe-Technologies/gensim/actions
.. _Downloads: https://pypi.org/project/gensim/
.. _License: https://radimrehurek.com/gensim/intro.html#licensing
.. _Wheel: https://pypi.org/project/gensim/

Gensim is a Python library for *topic modelling*, *document indexing* and *similarity retrieval* with large corpora.
Target audience is the *natural language processing* (NLP) and *information retrieval* (IR) community.

Features
---------

* All algorithms are **memory-independent** w.r.t. the corpus size (can process input larger than RAM, streamed, out-of-core)
* **Intuitive interfaces**

  * easy to plug in your own input corpus/datastream (simple streaming API)
  * easy to extend with other Vector Space algorithms (simple transformation API)

* Efficient multicore implementations of popular algorithms, such as online **Latent Semantic Analysis (LSA/LSI/SVD)**,
  **Latent Dirichlet Allocation (LDA)**, **Random Projections (RP)**, **Hierarchical Dirichlet Process (HDP)** or **word2vec deep learning**.
* **Distributed computing**: can run *Latent Semantic Analysis* and *Latent Dirichlet Allocation* on a cluster of computers.
* Extensive `documentation and Jupyter Notebook tutorials <https://github.com/RaRe-Technologies/gensim/#documentation>`_.


If this feature list left you scratching your head, you can first read more about the `Vector
Space Model <https://en.wikipedia.org/wiki/Vector_space_model>`_ and `unsupervised
document analysis <https://en.wikipedia.org/wiki/Latent_semantic_indexing>`_ on Wikipedia.

Installation
------------

This software depends on `NumPy and Scipy <https://scipy.org/install/>`_, two Python packages for scientific computing.
You must have them installed prior to installing `gensim`.

It is also recommended you install a fast BLAS library before installing NumPy. This is optional, but using an optimized BLAS such as MKL, `ATLAS <https://math-atlas.sourceforge.net/>`_ or `OpenBLAS <https://xianyi.github.io/OpenBLAS/>`_ is known to improve performance by as much as an order of magnitude. On OSX, NumPy picks up its vecLib BLAS automatically, so you don't need to do anything special.

Install the latest version of gensim::

    pip install --upgrade gensim

Or, if you have instead downloaded and unzipped the `source tar.gz <https://pypi.org/project/gensim/>`_ package::

    python setup.py install


For alternative modes of installation, see the `documentation <https://radimrehurek.com/gensim/#install>`_.

Gensim is being `continuously tested <https://radimrehurek.com/gensim/#testing>`_ under all `supported Python versions <https://github.com/RaRe-Technologies/gensim/wiki/Gensim-And-Compatibility>`_.
Support for Python 2.7 was dropped in gensim 4.0.0 – install gensim 3.8.3 if you must use Python 2.7.


How come gensim is so fast and memory efficient? Isn't it pure Python, and isn't Python slow and greedy?
--------------------------------------------------------------------------------------------------------

Many scientific algorithms can be expressed in terms of large matrix operations (see the BLAS note above). Gensim taps into these low-level BLAS libraries, by means of its dependency on NumPy. So while gensim-the-top-level-code is pure Python, it actually executes highly optimized Fortran/C under the hood, including multithreading (if your BLAS is so configured).

Memory-wise, gensim makes heavy use of Python's built-in generators and iterators for streamed data processing. Memory efficiency was one of gensim's `design goals <https://radimrehurek.com/gensim/intro.html#design-principles>`_, and is a central feature of gensim, rather than something bolted on as an afterthought.

Documentation
-------------
* `QuickStart`_
* `Tutorials`_
* `Tutorial Videos`_
* `Official Documentation and Walkthrough`_

Citing gensim
-------------

When `citing gensim in academic papers and theses <https://scholar.google.cz/citations?view_op=view_citation&hl=en&user=9vG_kV0AAAAJ&citation_for_view=9vG_kV0AAAAJ:u-x6o8ySG0sC>`_, please use this BibTeX entry::

  @inproceedings{rehurek_lrec,
        title = {{Software Framework for Topic Modelling with Large Corpora}},
        author = {Radim {\v R}eh{\r u}{\v r}ek and Petr Sojka},
        booktitle = {{Proceedings of the LREC 2010 Workshop on New
             Challenges for NLP Frameworks}},
        pages = {45--50},
        year = 2010,
        month = May,
        day = 22,
        publisher = {ELRA},
        address = {Valletta, Malta},
        language={English}
  }

----------------

Gensim is open source software released under the `GNU LGPLv2.1 license <https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>`_.
Copyright (c) 2009-now Radim Rehurek

.. _Official Documentation and Walkthrough: https://radimrehurek.com/gensim/
.. _Tutorials: https://github.com/RaRe-Technologies/gensim/blob/develop/tutorials.md#tutorials
.. _Tutorial Videos: https://github.com/RaRe-Technologies/gensim/blob/develop/tutorials.md#videos
.. _QuickStart: https://radimrehurek.com/gensim/gensim_numfocus/auto_examples/core/run_core_concepts.html

