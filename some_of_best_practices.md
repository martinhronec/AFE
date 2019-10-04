# Best practices

This serves (non-complete) list of best practices to be used for easier collaboration.
Feel free to add other ideas.

### GIT

- Do not push junk like:
	- ipynb_checkpoints (jupyter notebooks create these)
	- __pycache__ (created by python when running scripts)
	- outputs from tests like .coverage .cache
	- much more
- Use **.gitignore** files. Files added to this file will be ignored by git. Have a look at 
  [this projects .gitignore file](/.gitignore)*
- Generally when commiting do not commit more then one file and one feature in the file per commit,
  this make your life easier when going through the history or when some problems appear.
- Keep you git history clean.
- When starting a new project use either linear (always rebasing before merging)
  or non-linear (never rebase before merging) history. Both have (different) benefits.
  Combination still works, but looses the benefits.
- Use one branch for one thing. Name your branches as master, dev or development, fix/??? and feature/????.
  The ??? describe a specific purpose of the branch.
- Leverage the tree structure of branches - it is very powerful when used properly.
- Merge only branches that work so that you never have master branch that fails.
- If you work in a team, let someone else to merge your code. If you work alone,
  merge it the next day with a fresh look at the code.
- Never ever commit to master, only merge to it, even when working alone. This will save you a lot of trouble.
- Use tags for situations that are worth tagging, like *working model with score = 122345*.
- Use LFS when pushing data.
- Data versioning systems are still young and they have different benefits.
  But are different from GIT - it versions the code. Have a look at for example [DVC](https://dvc.org/).
- Same as code, your machine learning models can be versioned too. Have a look at for example [DVC](https://dvc.org/).
- Name your commits properly, it will save you some troubles.


### General

- Every repo has to have README and requirements file (used packages INCLUDING versions).
  Using code that does not have this is painful for others and future yourself.
- When using any library/package in the code you push, append the name of the library/package
  to requirements.txt with its **version**!!! Imagine in few years trying to guess the correct version
  of some library you used, to replicate your results.
- Comment your code. It is easier to work with it, correct it, test it,...
- Write documentation. *The code is the best documentation* is usually stated by people who never worked
  with someone else on the same code/project.
- Get rid of unused code. Its just junk.
- There is easy way how to run tests automatically with every push to gitlab/github/bitbucket.
- Test your code, even data science and machine learning code should be automatically tested.


### Jupyter notebooks

- Somewhat discussable, but try not to commit jupyter notebooks with output, its really hard to compare
  (in pull request) and read what is a difference between two versions of one notebook with output.
  There might be situations when the output is required, but those are very specific and very irregular events,
  but generally it is better to stop, reset the notebook and then push it.


### Python

- Always declare types (like 'def foo(number: float) -> pd.DataFrame:'), there are even tools
  that help to check that the types are correct (mypy, pylint,...). This is not required by python
  and the code will run even when the types are not correct, but imagine how much easier it is to read
  someone elses code or even your own when you know how the argument should look like.
- When importing try not to use aliases. Some are ok and well understood like *import pandas as pd*
  and *import numpy as np*, but imagine you have two settings files */settings/dev.py*
  and */settings/production.py* and you import it as *import settings.dev as settings* its very easy
  to forget to change this and very hard to read the code properly (you have to always check what was imported).
