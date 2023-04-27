[Home](../sequence.md) - Submitting labs via GitHub 


# Downloading and submitted your labs using GitHub

In this course, labs will be distributed and submitted using 
[GitHub Classroom](https://classroom.github.com/). 

For each lab, you will find a URL link posted under the `Assignments` tab in Carmen.
Follow that link and accept the "assignment".
A private GitHub repository will be created for you, specifically for that lab. 
For example, for lab1, the repo will be titled something like `ece5307sp23/lab1-yourgitname`.
That repo will contain a Jupyter notebook with many blank sections that need to be completed by you.

Next we will describe how to clone the first GitHub lab repo on your local machine. 
The directions below assume that you have already cloned the main `ece5307` repo.  If you have not yet done so, please follow the directions [here](./github.md) before proceeding.

The first step is to find a good place to clone your lab repo.
*Do not clone it inside another git repo*, such as your local copy of the `ece5307` repo!
If you are unsure whether a given directory lies within a git repo, you can check it as follows using the terminal window: 
```
    > cd [location where you want to create the new repo]
    > git status
    fatal: not a git repository (or any of the parent directories): .git
```
If you see the `'not a git repository'` message, that means you are not in a git repo and thus it is safe to clone a new git repo in that location.

## Cloning and submitting lab1 using GitHub Desktop 

* In the [GitHub Desktop]() app, you should see `ece5307sp23/lab1-yourgitname`.  Click on it, and then, under `Local Path`, enter the location on your computer where you would like the repository to be cloned.  (As described above, make sure it's not inside another git repository!)  Finally, click the `Clone` button and wait for the process to finish.

* Next, you will complete the lab.  To do this, fill in the `TODO` sections of the Jupyter notebook and run the code until the results are satisfactory.  *Do not create a new notebook!*

* When you are finished, it's time to locally commit your changes.  In GitHub Desktop, enter a short summary of your changes in the `Summary` field, and then click `Commit to main` to locally commit those changes.  

* You can now push your local commits up to GitHub.  In GitHub Desktop, you do this by clicking `Push origin`.  

* To verify that your push worked, navigate to `http://github.com/ece5307sp23/lab1-yourgitname`.  You should see your latest commit message, as well as the updated version of your `lab1.ipynb`. 

* You can revise your lab submission at any time before the deadline by modifying `lab1.ipynb` and repeating the steps above.  GitHub Desktop will show the changes you made relative to your last commit.

* TIP: If you are working with others on a *collaborative* project (e.g., the ece5307 final project), then you'll want to follow these steps instead:
    * In GitHub Desktop, click `Fetch changes` to get the most recent version 
      of the repository.
    * Modify your file(s).
    * In GitHub Desktop, enter a summary of your changes in the `Summary` box and 
      click `Commit to main` to locally commit them.
    * Now click `Push Origin`.  If there were no changes at the remote, 
      the push will be successful, and you'll be done. 
    * If there were changes at the remote, you'll need to continue with
      the steps below...
    * You'll first be asked to `Fetch` the changes. 
      At this point, you might want to check the commit messages
      on GitHub to see what the changes were.
    * When it seems safe to proceed, click the `Pull origin` button to 
      merge those changes into your local repository.
      There are now two possibilities:
        1) The remote changes do not conflict with your local changes.
           In this case, you'll see the `Push Origin` button.
           Click it and you'll be done.
        2) The remote changes *do* conflict with your local changes. 
           In this case, you'll see a warning about conflicts.
           You'll need to resolve these conflicts before proceeding. 
           To do this...
            * Click on `View conflicts`. 
              You'll see a list of files with conflicts.
            * For each file, click on `Open in editor` and 
              take one of the following 3 options:
              1) Use the "modified file from main" (i.e., the remote version),
                 which will overwrite the local one.
              2) Use the "modified file" (i.e., the local version),
                 which will overwrite the remote one.
              3) Edit the conflicting file. 
                 You can do this by selecting "Default Program" or 
                 by manually invoking your favorite editor.
                 In any case, look for lines like this:
                ```
                <<<<<<< HEAD
                    [the remote version of the conflicting line(s)]
                =======
                    [the local version of the conflicting line(s)]
                >>>>>>>
                ```
                and replace those lines with the desired content.
            * Once this is done, GitHub Desktop should report 
              `No conflicts remaining`.
              At this point, you can click the `Continue Merge` or 
              `Commit to main` button (whichever appears) 
              to commit your changes, and finally click `Push origin`.


## Cloning and submitting lab1 using `git` 

* We can clone the lab1 repo using the `git` client on the command line as follows:
```
    > git clone https://github.com/ece5307sp23/lab1-yourgitname
    > cd lab1-yourgitname  # change directory 
    > ls  # list contents of directory
```
After running the `ls` command, you should see the lab assignment `lab1.ipybn`, which is a Jupyter notebook, and possibly some other files.
(For example, there is a hidden `.git/` subdirectory that identifies `lab1-yourgitname` as a git repo, but make sure to *never* modify any files in `.git/` directly!)

* Now, if you have never run `git` before on this machine, you will need to configure it as follows:
```
    > git config --global user.name "FIRST_NAME LAST_NAME"
    > git config --global user.name "MY_NAME@example.com"
```
If you don't do this, `git` may block some future actions.

* Next, you will complete the lab.  To do this, fill in the `TODO` sections of the Jupyter notebook and run the code until the results are satisfactory.  *Do not create a new notebook!*

* When you are finished, you can add/commit/push your completed lab up to GitHub as follows:
```
    > git add lab1.ipynb  # this adds the file to the "staging area"
    > git commit -m "revised lab1.ipynb"  # this commits the change to your local repo, with a message
    > git push  # this pushes your commit to GitHub
```
At `http://github.com/ece5307sp23/lab1-yourgitname`, you should see your latest commit message, as well as the updated version of your `lab1.ipynb`. 

* You can revise your lab submission at any time before the deadline by modifying `lab1.ipynb` and repeating the add/commit/push steps above.

* TIP #1: Type [`git status`](https://www.cs.swarthmore.edu/git/git-status.php) before you add, commit, and push.  It will tell you the current state of your repo, e.g., whether files have been modified, staged (i.e., added), committed, and whether your local repo is in sync with GitHub or not.

* TIP #2: If you are working with others on a *collaborative* project (e.g., the ece5307 final project), then you want to use these add/commit/pull/push steps instead of add/commit/push:
```
    > git add lab1.ipynb  # this adds the file to the "staging area"
    > git commit -m "revised lab1.ipynb"  # this commits the change to your local repo, with a message
    > git pull  # this pulls down recent changes from GitHub
    > git push  # this pushes your commit to GitHub
```
Important: If your changes conflict with the changes that someone else has committed to GitHub, you'll see a `CONFLICT` message after the pull step, and you won't be able to push.  You need resolve the conflict first.  To do this...
    * Open the conflicting file(s) in your favorite editor.
    * Look for lines like this
      ```
      <<<<<<< HEAD
        [the remote version of the conflicting line(s)]
      =======
        [the local version of the conflicting line(s)]
      >>>>>>>
      ```
      and replace them with the desired content.
    * Do an git add/commit/pull/push.
    * For more on resolving conflicts, see [here](https://docs.github.com/en/github/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line).

If you'd like to learn more, see the `GitHub` documentation [here](https://guides.github.com/) and the `git` documentation [here](https://git-scm.com/doc) and [here](https://docs.github.com/en/get-started/using-git).  

