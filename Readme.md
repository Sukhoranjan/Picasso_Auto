1. Make an environment named Picasso_Auto from the environment.yaml
2. Picasso 0.10.0 comes with the suite however if you want any specific version, delete the the picasso folder and reinstall in the Picasso_Auto folder following the instructions from Picasso github.
3. Install in editable mode (Keep in mind this automated script is built on top of picasso 0.10.0, changing versions may crash the script)
4. if you want gpu accelaration, download pygpufit zip file, then go to the python folder, copy the path of the whl file, do a pip install pygpufit.whl in the Picasso_Auto environment, then youre are good to go with lq-gpu, otherwise it will throw error, if no gpu use lq only.
5. Change the variables in the config.toml file.
6. Make sure to update the root directory for new datasets
7. Be mindful of the suffix, so that every folder has the same suffix.
8. Go to environment conda then type python main.py run-all to execute everystep serially
9. All data generated including the pipeline will be saved in the root directory.
10. Developer Tip: Copy the data to be processed in a ssd and then run, for fast processing, running from hdd is possible but a bit slower."# Picasso_Auto" 
