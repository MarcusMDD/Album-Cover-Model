# Does the genre of an album influence the design of its cover?
## Intro

Does the music genre of an album influence its design, or is there a wider underlying design language ?  To address this question, I created a CNN to classify an album's genre based on an image of its cover, making use of the PyTorch library. 

Note: This writeup contains explanations for the decisions made during the process of addressing. The raw code can be found in the attached notebook.
## Dataset
### Overview
The base dataset used for training, evaluating and testing  the model is the '20k Album Covers within 20 Genres' by Michael Kerr, which can be found [here](https://www.kaggle.com/datasets/michaeljkerr/20k-album-covers-within-20-genres).  It contains 20 genres with 1000 album covers per genre. 
### Dataset cleaning 
  Looking through the dataset, it was apparent that album covers were duplicated both within a genre and across genres. Though  duplicates within genres were removed (i.e. 2 of the same album in the Rock genre), those between genres (i.e. the same album being in both rock and pop) were left in. This is so that  albums which straddle genres would be factored in to the design language of each genre. 

Many of the genres contained images of vinyl sleeves. These are not images of album covers, and look the same across genres. This would impact the model's ability to discriminate between different genres. A count of the amount of record sleeves found in each genre is listed in the table below. 

| genre            | number of record labels or sleeves |
| ---------------- | ---------------------------------- |
| Blues            | 8                                  |
| Classical        | 2                                  |
| Country          | 16                                 |
| DeathMetal       | 0                                  |
| Doom Metal       | 0                                  |
| Drum N bass      | 50<                                |
| Electronic       | 35                                 |
| Folk             | 7                                  |
| Grime            | 50<                                |
| Heavy Metal      | 4                                  |
| Hip Hop          | 22                                 |
| Jazz             | 6                                  |
| LoFi             | 3                                  |
| Pop              | 9                                  |
| Psychedelic rock | 12                                 |
| Punk             | 2                                  |
| Reggae           | <50                                |
| Soul             | 50                                 |
| Techno           | 50<                                |


The genres of Techno, Soul, Grime and DnB had over 5% of their dataset comprised of vinyl sleeve images. This is to be expected, as many tracks from these genres were typically club mixes and never saw full commercial releases. Therefore, these genres were removed, leaving a final data set of 16 genres.

### Dataset pipeline 

All images in the dataset of 14,000 plus were  300x300. In order to ensure that *all* image sizes were standardised, all images were sized to 300x300 before being converted into tensors. 

After splitting the dataset into the 16 categories, the length of each category was checked and the shortest category was 884 images long.  Each category was limited to 880 images, as the standard 80-10-10 split for the train, evaluation and test datasets was to be used,. This was the number closest to 884 which cleanly divided into the split. This ensured that no  single genre had greater influence over the model. 

Each split of the dataset: train ,eval and test was passed into its own loader . A batch size of 32 was used across all loaders to decrease noise ,as opposed to loading each image 1 at a time. Additionally, each batch loaded in to the training loop was shuffled so that a mix of genres would be loaded in each batch.
## Model

### Model Overview and Structure 

After initial testing confirmed that solely storing the images and labels on the CPU would cause the training and evaluation time to be impractical, GPU acceleration was used. 

The original  model  began by feeding each batch through  4 convolutional  blocks , each containing: a convolutional layer, a batch normalisation layer, and a ReLU activation
unit. 

The progression of channels  in the convolutional layers were as follows: 3 ,32 ,64 ,128 ,256. This progression was chosen as a baseline because it would be able to drill into many characteristics of the image, without overfitting.  A batch normalisation layer was included in each block to provide stability and increase the ability for the model to generalise.(what does one do?)

After these layers, the batch was fed through a global average pooling layer to make the network less prone to overfitting. (explain)

Finally the batch was fed through a classification layer: Each tensor was flattened, a dropout layer applied and finally a linear layer decreasing the 256 outputs to 16, one for each genre.


### Training loop

In each training loop, the prediction for each image was defined as the genre which was assigned the highest value. In addition to the number of correct predictions, the running loss and accuracy of each loop were also calculated and returned. This was done to aid in finetuning the model.

### Tuning changelog

The table below notes what was changed between each run 

| Run | Epoch | Learning Rate                 | Train Acc | Validation Acc | Changes made from prior run                                                                                                                                             |
| --- | ----- | ----------------------------- | --------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 20    | 0.001                         | 21        | 21             |                                                                                                                                                                         |
| 2   | 20    | 0.002                         | 21        | 20             | changed Learning rate to 0.002                                                                                                                                          |
| 3   | 20    | 0.001                         | 26        | 23             | added another convolutional block : 256 to 512 and changed learning rate back to 0.001( no  performance gain of loss when compared to prior)                            |
| 4   | 20    | 0.001                         | 25        | 24             | increased dropout in classifier layer to 0.3                                                                                                                            |
| 5   | 20    | variable (see changes column) | 28        | 26             | Used reduce LR on plateau as scheduler: base value of 0.001, factor of 0.25 and patience of 2 epochs.                                                                   |
| 6   | 20    | variable                      | 20        | 24             | added dropout layers to blocks 2 through 5 starting at 0.1 and increasing by 0.05 each time                                                                             |
| 7   | 20    | variable                      | 43        | 29             | removed dropouts specified above, added another block: 512 to 1024                                                                                                      |
| 8   | 50    | variable                      | 42        | 28             | added in dropouts: layers 2,4,6 with values 0.1,0.15 and 0.2 respectively                                                                                               |
| 9   | 50    | variable                      | 33        | 27             | Added weight decay of 0.0001 to Adam optimiser                                                                                                                          |
| 10  | 50    | variable                      | 36        | 27             | increased dropout to 0.15,0.2, 0.25 and 0.3 respectively                                                                                                                |
| 11  | 50    | variable                      | 37        | 29             | decreased dropout probabilities back to the level defined in the 8th run  and changed optimiser to AdamW                                                                |
| 12  | 50    | variable                      | 39        | 28             | decreased initial learning rate in scheduler to 0.0003                                                                                                                  |
| 13  | 50    | variable                      | 24        | 25             | removed 512 -1024 block, stacked convolutional blocks for 32,64,128 and 256                                                                                             |
| 14  | 50    | variable                      | 39        | 27             | reset model structure to 6 blocks as defined  in the 12th run                                                                                                           |
| 15  | 50    | variable                      | 49        | 28             | used random horizontal flip on training data set p=0.5 and colour jiggle with all parameters ( brightness, contrast, saturation, hue) set to 0.2 and probability to 0.5 |
| 16  | 50    | variable                      | 39        | 28             | removed data augmentation defined above in 15th run                                                                                                                     |
| 17  | 100   | variable                      | 55        | 30             | increased train and eval run to 100 epochs ( current best model)                                                                                                        |
| 18  | 100   | variable                      | 80        | 27             | change base learning rate to 0.0015 and change the factor for reduce LR on plateau to 0.75 ( mass overfitting)                                                          |
| 19  | 100   | variable                      | 65        | 27             | reduce base learning rate to 0.00125 and reduce factor to 0.5                                                                                                           |
| 20  | 100   | variable                      | 36        | 27             | reset learning rate to 0.001 and factor to 0.25                                                                                                                         |
| 21  | 100   | variable                      | 36        | 26             | increased weight decay to 0.000101<br>                                                                                                                                  |
| 22  | 100   | variable                      | 31        | 26             | decreased factor to 0.125, decreased patience to 1 in scheduler                                                                                                         |
| 23  | 100   | variable                      | 31        | 27             | increased base learning rate to 0.00105 in scheduler                                                                                                                    |
| 24  | 100   | variable                      | 38        | 27             | decreased factor to 0.1 and increased patience to 2 in scheduler                                                                                                        |
| 25  | 100   | variable                      | 92        | 29             | increased factor to 0.5, patience to 5 and added a cooldown of 1 epoch to the scheduler ( Mass overfitting)                                                             |
## Results 

### Model specification for final test run 
The model saved at the end of the 25th run was used. 
6 convolutional blocks  each containing  a convolutional layer, batch normalisation layer , and a ReLU activation unit.  The channel progression is :  3 ,32,64,128,256,512,1024. This is then fed through the global average pooling layer as above.

The AdamW optimiser was used, with a learning rate of 0.001 and a weight decay of 0.0001

For the learning rate, the reduce LR on plateau scheduler was used. The factor was set to 0.5, patience was set to 5 and a cooldown of 1 

### Confusion matrix  of final run 






| Actual \ Predicted  | Blues | Classical | Country | DeathMetal | DoomMetal | Electronic | Folk | HeavyMetal | HipHop | Jazz | Lofi | Pop | Psychedelic Rock | Punk | Reggae | Rock |
| ------------------- | ----- | --------- | ------- | ---------- | --------- | ---------- | ---- | ---------- | ------ | ---- | ---- | --- | ---------------- | ---- | ------ | ---- |
| **Blues**           | 10    | 5         | 7       | 2          | 2         | 6          | 4    | 6          | 1      | 11   | 0    | 8   | 7                | 5    | 7      | 7    |
| **Classical**       | 5     | 52        | 2       | 0          | 2         | 4          | 6    | 1          | 0      | 8    | 0    | 3   | 1                | 2    | 1      | 1    |
| **Country**         | 2     | 5         | 18      | 1          | 2         | 6          | 10   | 3          | 2      | 7    | 4    | 7   | 4                | 7    | 2      | 8    |
| **DeathMetal**      | 0     | 0         | 0       | 42         | 14        | 2          | 0    | 7          | 3      | 1    | 3    | 0   | 3                | 8    | 0      | 5    |
| **DoomMetal**       | 2     | 2         | 2       | 17         | 29        | 5          | 1    | 7          | 2      | 3    | 5    | 1   | 3                | 5    | 1      | 3    |
| **Electronic**      | 3     | 3         | 3       | 1          | 7         | 12         | 4    | 5          | 5      | 8    | 4    | 7   | 5                | 6    | 7      | 8    |
| **Folk**            | 9     | 8         | 3       | 2          | 4         | 8          | 14   | 2          | 3      | 9    | 3    | 5   | 2                | 3    | 5      | 8    |
| **HeavyMetal**      | 2     | 2         | 1       | 14         | 8         | 2          | 3    | 33         | 4      | 1    | 0    | 2   | 2                | 2    | 6      | 6    |
| **HipHop**          | 0     | 1         | 2       | 2          | 4         | 9          | 3    | 4          | 29     | 4    | 3    | 3   | 6                | 6    | 6      | 6    |
| **Jazz**            | 11    | 6         | 4       | 0          | 3         | 5          | 8    | 3          | 2      | 17   | 4    | 2   | 3                | 4    | 3      | 13   |
| **Lofi**            | 2     | 5         | 1       | 2          | 3         | 6          | 3    | 2          | 6      | 4    | 23   | 5   | 6                | 8    | 5      | 7    |
| **Pop**             | 6     | 5         | 5       | 2          | 3         | 8          | 7    | 2          | 5      | 10   | 8    | 16  | 1                | 3    | 1      | 6    |
| **PsychedelicRock** | 10    | 3         | 1       | 1          | 6         | 7          | 5    | 4          | 0      | 4    | 6    | 5   | 15               | 6    | 3      | 12   |
| **Punk**            | 4     | 0         | 4       | 2          | 1         | 3          | 4    | 3          | 8      | 2    | 5    | 4   | 5                | 30   | 9      | 4    |
| **Reggae**          | 4     | 3         | 1       | 1          | 0         | 4          | 5    | 4          | 5      | 3    | 6    | 5   | 3                | 12   | 22     | 10   |
| **Rock**            | 2     | 2         | 4       | 5          | 5         | 11         | 7    | 6          | 2      | 8    | 6    | 5   | 4                | 4    | 5      | 12   |

Comments: 

The model was most accurate at predicting the genre of classical album covers, correctly predicting 55 out of 88.

The model was least accurate at predicting blues, with only 10 correctly predicted. 

The model would often confuse doom and heavy metal with death metal, with 14 to 17 incorrect predictions . This could be due to the similarities of the subgenres leading to similar album design patterns. 

A similar level of confusion between specific genres also occurred between blues, rock and psychedelic rock
### Classification report

[Figure1](images/Figure1.png)
   
## Summary 

A completely  random guess  in a situation with 16 classes results in an accuracy of 6.25%. The final test run accuracy of 26.5% ended up being over 4x as accurate as if the genre was chosen  at random.  This could  show that there is at least some overall design cues which are indicative to specific groups of genres, if not specific genres themselves. The confusion matrix supports this assertion with certain  groups of genres having similar incorrect predictions numbers. 


## Limitations / Next Steps 

The dataset used for the model was made up of around 14,000 images. In order to improve accuracy, more data could have been sourced and used.  In regards to next steps, unlabelled images could be fed into the data set. In order to improve accuracy, apply transfer learning from a prebuilt model could help

