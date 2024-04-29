metadata_compiled.csv：原始数据
waveinfo.csv：取出了filename、cough_detected、duration、status、status_SSL、anomaly列。
waveinfo_labeled.csv：取出了有status, status_SSL至少一个标注的行。
waveinfo_unlabeled.csv：没有任何标注的行。
waveinfo_labeled_fine.csv：经过数据清洗，符合条件的带标注的行。
waveinfo_unlabeled_fine.csv：经过数据清洗的无标注行。

waveinfo_labedfine_staaSSL.csv：在“waveinfo_labeled_fine.csv”的基础上，合并status, status_SSL两行，以
status为主，没有的就按照SSL的标注。需要注意的是，SSL标注准确率只有83.47%。
统计得到0，1，2标签的数目分别为：13709 3288 939。
于是：
```
heal_indices = random.choices(heal_indices, k=(3288+939)//2+1)
cnt_list = [(3288+939)//2+1, 3288, 939]
resdf = metadf.iloc[heal_indices, :]
resdf = pd.concat([resdf, metadf.iloc[symp_indices, :]], axis=0)
resdf = pd.concat([resdf, metadf.iloc[cov19_indices, :]], axis=0)
resdf.to_csv("C:/Program Files (zk)/PythonFiles/AClassification/SoundDL-CoughVID/datasets/waveinfo_labedfine_forcls.csv", sep=',')
```