# freezer-ml-pipeline

This repo contains example code for 
* setting up dependecies (environment + compute target)
* a *traditional* long python script, excuting an experiment via Azure ML
* a modular pipeline, executing the experiment in several steps
* creating a triggered schedule for rerunning the pipeline.

# Instructions
Make sure to add your config.json (from your Azure ML Workspace) to your root or src directory.

The code will not run, as you do not have access to the original dataset. 

Replace code and datasat with your own.
