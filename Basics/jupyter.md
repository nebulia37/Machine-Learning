[Home](../sequence.md) - Running a Jupyter notebook

# Running a Jupyter notebook

Once your Python environment is set up, you can run a [Jupyter](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html) notebook (say, `demo.ipynb`) from a terminal window as follows.
First, navigate to the directory containing `demo.ipynb`.  Then type...
~~~bash
    > jupyter notebook
~~~
This will open up the Jupyter Notebook app in a web browser, which will show the files in the current directory.
Then...

* Clicking on `demo.ipynb` in the browser will open that notebook in a new browser tab, and then clicking the `Run` button in new tab will run one section of code at a time.

* On some platforms, you may need to first "trust" the notebook by clicking on the `not trusted` button.

* Clicking Control-C in the terminal window will terminate the Jupyter Notebook app.  Note that closing the browser tab _will not_ terminate the app; the app will keep running in the background.

* By default, Jupyter does not include a "variable inspector," which shows the current variables in the workspace, their types, values, etc.  But this can be a very useful feature!  You can add one by following the instructions 
[here](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/varInspector/README.html)
or 
[here](https://stackoverflow.com/questions/37718907/variable-explorer-in-jupyter-notebook).

A similar (and perhaps nicer) way to run Jupyter notebooks is by typing
~~~bash
    > jupyter lab
~~~

This will open the [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) app in a web browser. 

* One of the main [advantages](https://towardsdatascience.com/jupyter-lab-evolution-of-the-jupyter-notebook-5297cacde6b)  of JupyterLab is that it has a file browser and shows multiple tabs

* Also, it works with other programming languages (Julia, R) and filetypes (HTML, text, Markdown).

* JupyterLab will eventually replace the simpler Jupyter Notebook app.

* You can find a variable inspector for JupyterLab [here](https://github.com/lckr/jupyterlab-variableInspector).

<!--
Would be nice to get Jupyter working as a full-fledged IDE: 

https://towardsdatascience.com/jupyter-is-now-a-full-fledged-ide-c99218d33095

but this requires the xeus-python kernel, which seems incompatible with matplotlib.  Here is an evolving bug report...

https://github.com/jupyter-xeus/xeus-python/issues/224
-->
