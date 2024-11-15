# Task: G-FACE

<img src="assets/G-FACE.jpg" width="400"/>

## DIG-FACE: De-biased Learning for Generalized Facial Expression Category Discovery


We introduce a novel task, Generalized FAcial expressionCategory discovEry (G-FACE), 
that discovers new, unseen facial expressions while recognizing known categories effectively. 
Even though there are generalized category discovery methods for natural images, they show compromised performance on G-FACE. We identified two biases that affect the learning: 
implicit bias, coming from an underlying distributional gap between new categories in unlabeled data and known categories in labeled data, and explicit bias, coming from shifted preference on explicit visual facial change characteristics from known expressions to unknown expressions. By addressing the challenges caused by both biases, we propose a Debiased G-FACE method, namely DIG-FACE, that facilitates the de-biasing of both implicit and explicit biases. In the implicit debiasing process of DIG-FACE, we devise a novel learning strategy that aims at estimating and minimizing the upper bound of implicit bias. In the explicit debiasing process, we optimize the model’s ability to handle nuanced visual facial expression data by introducing a hierarchical  category-discrimination refinement strategy: sample-level, triplet-level, and distribution-level optimizations. Extensive experiments demonstrate that our DIG-FACE significantly enhances recognition accuracy for both known and new categories, setting a first-of-its-kind standard for the task.  

<img src="assets/frame_v2.jpg" width="800"/>


## Running

### Dependencies

```
pip install -r requirements.txt
```

### Scripts

**Train the model**:

```
bash scripts/run_${DATASET_NAME}.sh
```

### Datasets

We use FER datasets  in this paper, including:

RAF-DB, FerPlus and AffectNet.

Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset), put the RAF-DB-Basic folder under the dataset folder:
```key
-FERdata/
  raf-basic/
	   Image/aligned/
	     train_00001_aligned.jpg
	     test_0001_aligned.jpg
	     ...

```
Before RAF-DB-Compound training, it is first necessary to ensure that both basic expressions and compound expressions are available in the dataset, so that we can do the discovery of composite classes based on the basic classes.
put the  RAF-DB-compound folder under the dataset folder:
```key
-FERdata/
  raf-comp/
	   compound-aligned/
	     train_00001_aligned.jpg
	     test_0001_aligned.jpg
	     ...

AffectNet is a large-scale FER dataset, access to which is subject to licensing in link [AffectNet](http://mohammadmahoor.com/affectnet/),
