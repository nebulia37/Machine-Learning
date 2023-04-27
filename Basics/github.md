[Home](../sequence.md) - GitHub 

# Using GitHub to access the course materials

Much of the material for this course will be hosted on [GitHub](https://github.com/). 
GitHub is a powerful platform often used by software development teams to share files when collaborating on large projects. 
It has become the standard way to share machine-learning code.
A complete guide to GitHub is available on the [GitHub guide site](https://guides.github.com/). 

To access materials and submit your lab solutions, you only need to understand a few basic ideas. 
First, the materials for this course will live in the *repository* [https://github.com/ece5307sp23/ece5307](https://github.com/ece5307sp23/ece5307).
Once you have access, you can view the contents of that repository using a web browser. 

## Getting access

To get access, you can do the following.

* If you do not already have a GitHub account, please sign up at [GitHub](https://github.com/).

* Click on the link [https://classroom.github.com/a/0p7ImrdT](https://classroom.github.com/a/0p7ImrdT).

* Accept the authorization request from GitHub Classroom.

* Find your name in the list and click on it. [Email the professor](mailto:schniter.1@osu.edu) if you can't find your name!

* Join the team called `everyone`.

* You will then be taken to an empty repository called `ece5307sp23/hello-everyone`, which you can ignore.

* Click on the blue `ece5307sp23` and you should see another repository called `ece5307`. This is where most of the course materials will live. You should also see the `ece5307sp23` organization now listed on your GitHub home page.

* Click on `ece5307` and you will see the README.md file. This README.md file contains lots of useful information and hyperlinks to course materials. Throughout the semester, this README file will grow as more information is added.  

## Creating and updating your local copy 

Now that you have access to the main course repo, you will want to download it to your local machine, and update your local copy throughout the semester, as materials are added. 
There are two ways to do this: 1) use the GitHub Desktop app, or 2) use a command-line `git` client. 

### GitHub Desktop app

We first describe how to use a the [GitHub Desktop](https://docs.github.com/en/desktop) app to clone and update your local copy of the main course repository.

* First, download the app and connect it to your GitHub account by following [these directions](https://docs.github.com/en/desktop/installing-and-configuring-github-desktop/overview/getting-started-with-github-desktop).  (You shouldn't need to do any customization of the app.)

* Among your repositories, you should see `ece5307sp23/ece5307`. Click on it, and then, under the `Local Path`, enter the location on your computer where you would like the repository to be cloned.  Click the `Clone` button and wait for the process to finish.

* At any time, you can click on the `Fetch Origin` button to update your local copy of the repository.  Sometimes, when changes have been made to the remote repo, the button will appear as `Pull Origin` and it will show how many commits (i.e., changes) were made.  Click the `Pull Origin` button to merge those changes into your local repository.

* If you'd like to clone a different repository, select File -> Clone Repository, and choose from your GitHub repositories or enter a URL.

### Command-line `git` client

We now describe how to use a command-line `git` client to clone (i.e., download) and update your local copy of the main course repository.

* First, you will need to [create a Personal Access Token](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token) on GitHub.

* Next, check to see whether a `git` client is installed on your local machine. If not, install one. 

* We're now ready to clone the repository.  To do this...
    * Open a terminal window or command shell. 
      (For Windows, you might use the [powershell](https://docs.microsoft.com/en-us/powershell)). 
      Then in the terminal window, type:
    ```
        > cd [directory where you want the files on your local machine]
        > git clone https://github.com/ece5307sp23/ece5307
    ```
    * You will now be prompted for your username and password.
      *IMPORTANT: Enter your GitHub Personal Access Token instead of your github.com password!*
    * You should now have a local directory called `ece5307` with all the files from the GitHub repo. 
* The course repository will change over the course of the semester, as material gets added or modified. To update your local copy at any time, open the terminal window and type:
```
    > cd [directory containing ece5307 git repo; should end in "ece5307"]
    > git pull
```

## Now that you are set up...

You will also use GitHub to [submit your lab solutions](./github_labs.md). You can use either the `git` client or the GitHub Desktop app.

GitHub is a great tool for your personal projects, even if you are working individually. Using GitHub will allow you to do version control, develop in an organized manner, and release your work to a broader audience.  That's why GitHub is the main way that people develop and share machine-learning code. 
