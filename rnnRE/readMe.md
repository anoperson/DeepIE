This is the source code for the paper:

Combining Neural Networks and Log-linear Models to Improve Relation Extraction

Thien Huu Nguyen, and Ralph Grishman (2016)

If you use this code, please cite the following paper:

```
@inproceedings{Nguyen:16,
  author={Nguyen, Thien Huu and Grishman, Ralph}, 
  title={Combining Neural Networks and Log-linear Models to Improve Relation Extraction},
  booktitle={Proceedings of IJCAI Workshop on Deep Learning for Artificial Intelligence (DLAI)},
  year={2016}
}
```

There are two steps to run this code:

**Preprocessing: using file ```rnn_process_onlyExtra.py```**

You will need to have the ACE 2005 data set in the format required by this file. The sample files are provided in the ```data``` folder. The names of the files in this folder should be kept. See the readMe.md in this folder to see the format.

```python rnn_process_onlyExtra.py full word2vec path-to-the-bin-file-of-word2vec path-to-the-data-folder-above```

**Train and test the model: using file ```evaluate.py```**
    
```python evaluate.py```

This step takes the output file in step 1.

There are various parameters of the model you can change directly in the ```evaluate.py``` file.

Note that this release includes some development code, so it is more suitable for research purpose. You can use GPU to speed up.

THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

