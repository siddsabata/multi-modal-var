### A Quick Note on Disk Space

Your home directory has a limited storage quota. The Conda environment and the packages you install, especially large ones for deep learning, can consume several gigabytes. For instance the HW1 conda env (`progen`) will be 6.8GB, and the evo2 environment will be 18.2GB. So most certainly you might run into space issues.


The `polyfill-glibc` patching step in the next section can fail with a misleading `Error writing to file` message if your disk is full.

Before proceeding, it's a good idea to check your available space on your home folder in PSC. You can see how much space you have left on the filesystem with:

```bash
my_quotas
# example output:
The quota for home directory /jet/home/rhettiar
Storage quota: 25.00GiB
 Storage used: 22.82GiB
  Inode quota: 0
  Inodes used: 120,115

The quota for project directory /ocean/projects/cis250160p
Storage quota: 1.95TiB
 Storage used: 608.20GiB
  Inode quota: 12,140,000
  Inodes used: 1,293,748
```

If you are low on space, here are three ways to fix it:

1.  **Remove Old Conda Environments**
You may have environments from previous assignments that you no longer need. First, list all your environments to see what you have:

```bash
conda env list
```

If you find an old environment you don't need anymore (e.g., one named `progen` from a previous project), you can remove it completely to free up a lot of space.

```bash
# Replace <env_name> with the actual name of the environment to delete
conda env remove -n <env_name>

# Example:
conda env remove -n progen
```
2.  **Clean Conda Caches:** Conda keeps a cache of downloaded packages that you can safely clear to free up a significant amount of space.
    ```bash
    conda clean --all
    ```
3.  **Move Other Projects:** If you have large files or old projects in your home directory, move them to your dedicated project scratch space, which has much more room.
    ```bash
    # Example: moving progen env
    mv /jet/home/<your_username>/.conda/envs/progen /ocean/projects/cis250160p/<your_username>/
    ```