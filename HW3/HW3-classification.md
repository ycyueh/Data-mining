## HW3-classification

**音頻特徵處理：**

|       | danceability |       energy |          key |     loudness |         mode |  speechiness | acousticness | instrumentalness |     liveness |      valence |        tempo |   duration_ms | time_signature |
| ----: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | -----------: | ---------------: | -----------: | -----------: | -----------: | ------------: | -------------: |
| count | 42305.000000 | 42305.000000 | 42305.000000 | 42305.000000 | 42305.000000 | 42305.000000 | 42305.000000 |     42305.000000 | 42305.000000 | 42305.000000 | 42305.000000 |  42305.000000 |   42305.000000 |
|  mean |     0.639364 |     0.762516 |     5.370240 |    -6.465442 |     0.549462 |     0.136561 |     0.096160 |         0.283048 |     0.214079 |     0.357101 |   147.474056 | 250865.846685 |       3.972580 |
|   std |     0.156617 |     0.183823 |     3.666145 |     2.941165 |     0.497553 |     0.126168 |     0.170827 |         0.370791 |     0.175576 |     0.233200 |    23.844623 | 102957.713571 |       0.268342 |
|   min |     0.065100 |     0.000243 |     0.000000 |   -33.357000 |     0.000000 |     0.022700 |     0.000001 |         0.000000 |     0.010700 |     0.018700 |    57.967000 |  25600.000000 |       1.000000 |
|   25% |     0.524000 |     0.632000 |     1.000000 |    -8.161000 |     0.000000 |     0.049100 |     0.001730 |         0.000000 |     0.099600 |     0.161000 |   129.931000 | 179840.000000 |       4.000000 |
|   50% |     0.646000 |     0.803000 |     6.000000 |    -6.234000 |     1.000000 |     0.075500 |     0.016400 |         0.005940 |     0.135000 |     0.322000 |   144.973000 | 224760.000000 |       4.000000 |
|   75% |     0.766000 |     0.923000 |     9.000000 |    -4.513000 |     1.000000 |     0.193000 |     0.107000 |         0.722000 |     0.294000 |     0.522000 |   161.464000 | 301133.000000 |       4.000000 |
|   max |     0.988000 |     1.000000 |    11.000000 |     3.148000 |     1.000000 |     0.946000 |     0.988000 |         0.989000 |     0.988000 |     0.988000 |   220.290000 | 913052.000000 |       5.000000 |

'danceability'：適合跳舞的， A value of 0.0 is least danceable and 1.0 is most danceable

'energy'：a perceptual measure of intensity and activity.

'key'：音調， If no key was detected, the value is -1.

'loudness'：音量，The overall loudness of a track in decibels (dB)

'mode'：modality (major or minor) of a track. Major is represented by 1 and minor is 0.

'speechiness'： the presence of spoken words in a track.(the closer to 1.0 the attribute value)

'acousticness'：1.0 represents high confidence the track is acoustic.

'instrumentalness'： whether a track contains no vocals.The closer the instrumentalness value is to 1.0

'liveness'：Detects the presence of an audience in the recording

'valence'： 1.0 describing the musical positiveness conveyed by a track.=>sound more positive

'tempo'：The overall estimated tempo of a track in beats per minute (BPM).

'duration_ms'：歌曲的持續時間

'time_signature'：這種拍號表示以四分音符為一拍，每小節有四拍

遺失值：
song_name：20786、Unnamed: 0 ：21525、title：21525
移除資料中重複值後，剩下的資料不存在遺失值。
### EDA

![Figure 2025-01-31 143454 (0)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (0).png)

![Figure 2025-01-31 143454 (1)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (1).png)

![Figure 2025-01-31 143454 (2)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (2).png)

![Figure 2025-01-31 143454 (3)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (3).png)

![Figure 2025-01-31 143454 (4)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (4).png)

![Figure 2025-01-31 143454 (5)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (5).png)

![Figure 2025-01-31 143454 (6)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (6).png)

![Figure 2025-01-31 143454 (7)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (7).png)

![Figure 2025-01-31 143454 (8)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (8).png)

![Figure 2025-01-31 143454 (9)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (9).png)

![Figure 2025-01-31 143454 (10)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (10).png)

![Figure 2025-01-31 143454 (11)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (11).png)

![Figure 2025-01-31 143454 (12)](D:\essay\OneDrive - National ChengChi University\桌面\113-01\資料採掘\hw3\Figure 2025-01-31 143454 (12).png)

**數據預處理：**

`One-Hot Encoding` (將`time_signature,mode`轉成數值化)

`genre`轉成 Label

`smote` 處理資料不平衡

| 類別             | 原始數據數量 | SMOTE 後數量 |
| ---------------- | ------------ | ------------ |
| Underground  Rap | 5771         | 5771         |
| Dark Trap        | 4518         | 5771         |
| Hiphop           | 2960         | 5771         |
| trance           | 2734         | 5771         |
| techno           | 2632         | 5771         |
| psytrance        | 2610         | 5771         |
| dnb              | 2407         | 5771         |
| hardstyle        | 2277         | 5771         |
| trap             | 2226         | 5771         |
| techhouse        | 2192         | 5771         |
| RnB              | 2039         | 5771         |
| Trap  Metal      | 1897         | 5771         |
| Rap              | 1809         | 5771         |
| Emo              | 1601         | 5771         |
| Pop              | 452          | 5771         |

**1. Random Forest分析：**

- Cross-Validation：use k = 5
  - Cross-Validation average Accuracy: 0.7567
  
- Random Forest  
  - parameters：n_estimators=100、max_depth=30
  
  - Train Accuracy: 0.9700
  
  - Cross-Validation Accuracy: [0.75720165 0.76398816 0.76086643 0.76375451 0.76245487]
    Cross-Validation Accuracy: 0.7804
  
  - classification report
    | 類別             | Precision | Recall | F1-score | Support |
    | ---------------- | --------- | ------ | -------- | ------- |
    | Dark Trap        | 0.57      | 0.47   | 0.52     | 1139    |
    | Emo              | 0.87      | 0.89   | 0.88     | 1180    |
    | Hiphop           | 0.62      | 0.6    | 0.61     | 1154    |
    | Pop              | 0.84      | 0.95   | 0.89     | 1121    |
    | Rap              | 0.71      | 0.72   | 0.72     | 1191    |
    | RnB              | 0.7       | 0.7    | 0.7      | 1174    |
    | Trap  Metal      | 0.65      | 0.72   | 0.68     | 1129    |
    | Underground  Rap | 0.35      | 0.3    | 0.32     | 1142    |
    | dnb              | 0.98      | 0.98   | 0.98     | 1219    |
    | hardstyle        | 0.93      | 0.95   | 0.94     | 1202    |
    | psytrance        | 0.95      | 0.95   | 0.95     | 1140    |
    | techhouse        | 0.91      | 0.94   | 0.93     | 1100    |
    | techno           | 0.91      | 0.91   | 0.91     | 1166    |
    | trance           | 0.91      | 0.92   | 0.91     | 1111    |
    | trap             | 0.91      | 0.92   | 0.92     | 1145    |
    | Accuracy         |           |        | 0.8      | 17313   |
    | Macro avg        | 0.79      | 0.8    | 0.79     | 17313   |
    | Weighted  avg    | 0.79      | 0.8    | 0.79     | 17313   |
  
    - 可看出 Underground  Rap  的分類結果最差，f1 score 僅有 32 %
      - Underground Rap只一種饒舌音樂事業的面象，它不是一種style，它是聽不出來的( from ptt )
    - dnb 的分類結果最好，f1 score 為 98 %
  
  - confusion matrix：
  
    - 附圖可看出 Dark Trap、Trap Metal 風格相近；Hiphop、RnB 風格較相近，易被分錯
    - Underground Rap 的風格較不明確
    - 其餘類別音樂特徵明確，準確率高
  
  
  ![image-20250113172448746](D:\essay\OneDrive - National ChengChi University\桌面\工作\vizuro\js_visualize\image-20250113172448746.png)
  
  - feature selection：
    - 以 tempo、duration_min、instrumentalness、danceability、loudness 為前五重要特徵
  
   <img src="C:\Users\USER\AppData\Roaming\Typora\typora-user-images\image-20250113172429802.png" alt="image-20250113172429802" style="zoom: 50%;" />
  

## **2.SVC**
- Cross_val Scores:  [0.6374 0.6346 0.6357]

- Train Accuracy(average): 0.6359
- Test Accuracy: 0.6372

<img src="C:\Users\USER\AppData\Roaming\Typora\typora-user-images\image-20250113171819268.png" alt="image-20250113171819268" style="zoom: 50%;" />

<img src="C:\Users\USER\AppData\Roaming\Typora\typora-user-images\image-20250113171834134.png" alt="image-20250113171834134" style="zoom: 50%;" />

- classification report
  - psytrance 為分類結果最佳之類別，F1 score 達0.93；Underground Rap 仍為結果最差之音樂類別，F1 score 僅有0.32。
  - 平均而言，SVM 方法分類結果較隨機森林差。

```
                  precision    recall  f1-score   support

      Dark Trap       0.44      0.38      0.41      1139
            Emo       0.50      0.67      0.58      1180
         Hiphop       0.47      0.39      0.42      1154
            Pop       0.47      0.54      0.50      1121
            Rap       0.47      0.53      0.50      1191
            RnB       0.40      0.36      0.38      1174
     Trap Metal       0.49      0.58      0.53      1129
Underground Rap       0.34      0.29      0.32      1142
            dnb       0.90      0.91      0.90      1219
      hardstyle       0.81      0.73      0.77      1202
      psytrance       0.93      0.93      0.93      1140
      techhouse       0.83      0.86      0.84      1100
         techno       0.88      0.87      0.88      1166
         trance       0.81      0.79      0.80      1111
           trap       0.85      0.73      0.79      1145

       accuracy                           0.64     17313
      macro avg       0.64      0.64      0.64     17313
   weighted avg       0.64      0.64      0.64     17313
```






