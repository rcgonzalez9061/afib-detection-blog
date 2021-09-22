---
layout: page
title: AFib Detection
permalink: /
---

# Detecting Atrial Fibrilation with 1-D CNNs

So I recently bought an Apple Watch. It was on sale for a pretty good deal and I've been wanting one for quite a while, so I just had to get it. Like I normally do with all of my recently purchased gadgets, I played around with it in the middle of the night to figure out all of its features. This led to me discovering the ECG feature which intrigued me, so I set out to recreate a similar system to detect Atrial Fibrillation (AFib) like Apple's implementation does. 

The thing is, I had no prior experience with ECGs. Most of what I know is focused around computers, so the biomedical domain isn't really my specialty. However, I thought this project would serve as a good exploration into the domain.

**TLDR;** I got an Apple Watch. It can do ECGs. I wanted to replicate that and learn more about ECGs.

## Data

The first thing I had to do was to source a good ECG dataset. Luckily, I quickly found the [MIT-BIH Atrial Fibrillation Database](https://physionet.org/content/afdb/1.0.0/). It contains annotated ECG data from 23 unique patients each with two simultaneous 10 hour ECG signals. The annotations detailing the rythym at a given time are given in a record's `.atr` files. There are 4 different types of annotations for hearth rythyms: Atrial Fibrilation (AFIB), Atrial Flutter (AFL), AV Junction (J), and "all other rythyms" (N). The dataset mostly captures AFIB and N rythyms while AFL and J only make up around 1% of the data.


<style>
#data-overview {
    margin: auto !important;
    display: flex;
    justify-content: center;
}
</style>


<div id='#data-overview'><style  type="text/css" >
</style><table id="T_ccb7b_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Total Duration<br>(Minutes)</th>        <th class="col_heading level0 col1" >Total<br>Duration (%)</th>        <th class="col_heading level0 col2" >Unique<br>Occasions</th>        <th class="col_heading level0 col3" >Min<br>Duration</th>        <th class="col_heading level0 col4" >Avg Duration<br>(Samples)</th>        <th class="col_heading level0 col5" >Long Samples<br>(>30s)</th>    </tr>    <tr>        <th class="index_name level0" >Label</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_ccb7b_level0_row0" class="row_heading level0 row0" >AFIB</th>
                        <td id="T_ccb7b_row0_col0" class="data row0 col0" >5,603.85</td>
                        <td id="T_ccb7b_row0_col1" class="data row0 col1" >39.87%</td>
                        <td id="T_ccb7b_row0_col2" class="data row0 col2" >291</td>
                        <td id="T_ccb7b_row0_col3" class="data row0 col3" >420</td>
                        <td id="T_ccb7b_row0_col4" class="data row0 col4" >288,858</td>
                        <td id="T_ccb7b_row0_col5" class="data row0 col5" >226</td>
            </tr>
            <tr>
                        <th id="T_ccb7b_level0_row1" class="row_heading level0 row1" >AFL</th>
                        <td id="T_ccb7b_row1_col0" class="data row1 col0" >97.95</td>
                        <td id="T_ccb7b_row1_col1" class="data row1 col1" >0.70%</td>
                        <td id="T_ccb7b_row1_col2" class="data row1 col2" >14</td>
                        <td id="T_ccb7b_row1_col3" class="data row1 col3" >882</td>
                        <td id="T_ccb7b_row1_col4" class="data row1 col4" >104,947</td>
                        <td id="T_ccb7b_row1_col5" class="data row1 col5" >7</td>
            </tr>
            <tr>
                        <th id="T_ccb7b_level0_row2" class="row_heading level0 row2" >J</th>
                        <td id="T_ccb7b_row2_col0" class="data row2 col0" >5.52</td>
                        <td id="T_ccb7b_row2_col1" class="data row2 col1" >0.04%</td>
                        <td id="T_ccb7b_row2_col2" class="data row2 col2" >12</td>
                        <td id="T_ccb7b_row2_col3" class="data row2 col3" >380</td>
                        <td id="T_ccb7b_row2_col4" class="data row2 col4" >6,894</td>
                        <td id="T_ccb7b_row2_col5" class="data row2 col5" >3</td>
            </tr>
            <tr>
                        <th id="T_ccb7b_level0_row3" class="row_heading level0 row3" >N</th>
                        <td id="T_ccb7b_row3_col0" class="data row3 col0" >8,349.30</td>
                        <td id="T_ccb7b_row3_col1" class="data row3 col1" >59.40%</td>
                        <td id="T_ccb7b_row3_col2" class="data row3 col2" >288</td>
                        <td id="T_ccb7b_row3_col3" class="data row3 col3" >1062</td>
                        <td id="T_ccb7b_row3_col4" class="data row3 col4" >434,859</td>
                        <td id="T_ccb7b_row3_col5" class="data row3 col5" >263</td>
            </tr>
    </tbody></table></div>

Again, I am going into this project with little to no background knowledge on how ECGs work and what characteristics each rhythm tends to display. So using the data, I'm going to hypothesize the characteristics of each rhythm then compare my hypotheses with existing information.

## Time Domain Analysis

FFirst, I've curated a small random sample that has decent variation which I have plotted below for a quick visual comparison. While I don't really notice any obvious differences between the different rhythms, I did notice that they see to follow the same pattern of having some sort of lead-up activity, a giant spike in activity and then some follow-up activity. This is extremely prevalent in record 08455.

{% include 1x4.png %}

I later learned that these were the P, R, and T waves of the heart rhythm. However, I missed a couple other waves. According to this diagram, it should appear like this for normal sinus rhythm (Wikipedia).

{% include SinusRhythmLabels.png %}

## Frequency Domain Analysis

Since nothing really stood out between the different rythms, perhaps another view of the data might prove more informative. Using the same sample, I plotted their Discrete Fourier Transforms and noted some observations below.

<table style='font-size:14px'>
<thead>
  <tr>
    <th>AFIB</th>
    <th>AFL</th>
    <th>J</th>
    <th>N</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>• Hard to notice distinct harmonic banding<br>• Noisy †<br></td>
    <td>• Clear harmonic banding<br>• High frequency fundamental</td>
    <td>• Sometimes strong harmonic banding<br>• Low frequency fundamental<br>• Slightly noisy †</td>
    <td>• Occasionally strong harmonic banding<br>• Records 05091 and 04936 are a little noisy †<br>• Record 08455 has 60 Hz noise<br>(probably a product of the data capture process)</td>
  </tr>
</tbody>
</table>

† I should be careful when I say "noisy". If you actually look at the signals they derived from, they aren't so noisy per say. However, in the frequency domain, it is hard distinguished harmonic spikes like we see in the other signals.

{% include 1x4_dft.png %}

## Aggregated FDA

To do a more general comparison, I took 200 samples for each class, applied a Discrete Fourier Transform, and averaged their values to generate the plot below.

{% include mean_ecg_dft.png %}

The only real distinction I can make is sharpness in the banding for each rythym. However, I believe this doesn't really inform us much about the characteristics of each class. We can expect banding as the result of the harmonics produced by the beat rythym. The sharpness of this banding is likely a result of the variance in each label. AFL and J will have low variance due to the limited number of samples available for them while AFIB and N have higher variance since they have a larger sample pool. Thus, AFL and J appear "sharper" while AFIB and N are more "fuzzy".

There is a slight contradiction though. AFIB and N do not follow this pattern. N has slightly more sample availability than AFIB, having about 20% more data available by duration and appearing in more unique occasions that last longer than 5 seconds. Still, AFIB appears to have more variance making it hard to distinguish any clear harmonics. Meanwhile, N has some pretty evident harmonic banding. 

One explanation could be due to AFIB appearing in 23 records while N only appears in 21. If this is the case, then I theorize that N's appearance in this plot will become more fuzzy as I increase the sample size for each class. I tested this at two other sample sizes. At 1,000, I observed slightly more variance in both N and AFIB, but harmonic banding remained distinguishable in N. At 10,000 there was no noticable difference than that at the 1,000 level. From this experiment, I can't conclusively say that this isn't the case.

Another explanation could be that AFIB rythyms simply have more variation in BPM resulting in this plot appearing less sharp.

One final explanation might be more clear if we reference the previous frequency domain plot.  Recall how I noted that the AFIB plots had indistinguishable spikes. This may indicate the AFIB signals are more eratic resulting in their aggregates appearing as they do.

## Detection Models

Now, my main goal: Creating machine learning models to detect AFib. I actually had a lot of trouble with this initially. I thought that by having such a large dataset, I could just generate random samples during training and validation. However, this system yielded unreliable results, no matter the sample sizes I used. After struggling for a while, I referred to [Detection of Atrial Fibrillation Using 1D Convolutional Neural Network (Hsieh, 2020)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7180882/) for some guidance which lead to my final data loading system. After making these changes, I immediately saw better results. Here's the details:

### ETL Pipeline

To split the data into 10-second labeled samples with 3-fold cross-validation, I extracted each unique occurance of an annotation and noted the record it came from, when the occurance began, and when it ended. Then I discarded any occurances less than 30s (3x the expected length) and split each occurance into 3 smaller, equally-sized signals and randomly one-to-one mapped each to a fold. From here, the subsamples were further split into 10 second slices with a 50% overlap between each, discarding any excess. This resulted in 54,989 samples (22,020 AFIB, 32,969 N) per fold. 

### Models
When considering what type model to apply to this problem, I immediately jumped to Convolutional Neural Networks. CNNs have proven themselves as very capable signal classifiers in various other tasks, so I thought that they should be my go-to answer for this problem. However, the exact architecture of a CNN can vary widely so I've compared various designs in this project. The only common elements for the models is that they each take 10-second, 2-lead ECGs as their input and output a prediction of Normal Sinus Rhythym (0) or AFib (1).

To generate a baseline I used two models: First a 1-D variation of the Pytorch MobileNetV2 implementation and second the model described in Hsieh et. al, 2020. I also created a self-made CNN (though admittedly I have very little experience with them).
To train a model, I held out one fold for validation and trained on the remaining data and repeated this for each model and fold.

### Ensembles
Lastly, I grouped each fold by architecture into ensembles by averaging their outputs (without performing any further training). I then evaluated the ensembles on the entire dataset to determine if averaging outputs was an effective approach for merging the various models together.


<style>
.level0 {
    text-align: center !important;
}

#ind-model-perf {
    margin: auto !important;
    display: flex;
    justify-content: center;
}
</style>


<div id='ind-model-perf'><style  type="text/css" >
</style><table id="T_5258d_" ><caption>Individual Model Performance</caption><thead>    <tr>        <th class="index_name level0" >Model</th>        <th class="col_heading level0 col0" colspan="3">Custom</th>        <th class="col_heading level0 col3" colspan="3">Hsieh</th>        <th class="col_heading level0 col6" colspan="3">MobileNetV2</th>    </tr>    <tr>        <th class="blank level1" ></th>        <th class="col_heading level1 col0" >AUC</th>        <th class="col_heading level1 col1" >Accuracy</th>        <th class="col_heading level1 col2" >F1 Score</th>        <th class="col_heading level1 col3" >AUC</th>        <th class="col_heading level1 col4" >Accuracy</th>        <th class="col_heading level1 col5" >F1 Score</th>        <th class="col_heading level1 col6" >AUC</th>        <th class="col_heading level1 col7" >Accuracy</th>        <th class="col_heading level1 col8" >F1 Score</th>    </tr>    <tr>        <th class="index_name level0" >Fold</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_5258d_level0_row0" class="row_heading level0 row0" >1</th>
                        <td id="T_5258d_row0_col0" class="data row0 col0" >0.9981</td>
                        <td id="T_5258d_row0_col1" class="data row0 col1" >0.9884</td>
                        <td id="T_5258d_row0_col2" class="data row0 col2" >0.9855</td>
                        <td id="T_5258d_row0_col3" class="data row0 col3" >0.9993</td>
                        <td id="T_5258d_row0_col4" class="data row0 col4" >0.9929</td>
                        <td id="T_5258d_row0_col5" class="data row0 col5" >0.9911</td>
                        <td id="T_5258d_row0_col6" class="data row0 col6" >0.9994</td>
                        <td id="T_5258d_row0_col7" class="data row0 col7" >0.9925</td>
                        <td id="T_5258d_row0_col8" class="data row0 col8" >0.9907</td>
            </tr>
            <tr>
                        <th id="T_5258d_level0_row1" class="row_heading level0 row1" >2</th>
                        <td id="T_5258d_row1_col0" class="data row1 col0" >0.9995</td>
                        <td id="T_5258d_row1_col1" class="data row1 col1" >0.9940</td>
                        <td id="T_5258d_row1_col2" class="data row1 col2" >0.9925</td>
                        <td id="T_5258d_row1_col3" class="data row1 col3" >0.9995</td>
                        <td id="T_5258d_row1_col4" class="data row1 col4" >0.9943</td>
                        <td id="T_5258d_row1_col5" class="data row1 col5" >0.9928</td>
                        <td id="T_5258d_row1_col6" class="data row1 col6" >0.9994</td>
                        <td id="T_5258d_row1_col7" class="data row1 col7" >0.9938</td>
                        <td id="T_5258d_row1_col8" class="data row1 col8" >0.9922</td>
            </tr>
            <tr>
                        <th id="T_5258d_level0_row2" class="row_heading level0 row2" >3</th>
                        <td id="T_5258d_row2_col0" class="data row2 col0" >0.9997</td>
                        <td id="T_5258d_row2_col1" class="data row2 col1" >0.9951</td>
                        <td id="T_5258d_row2_col2" class="data row2 col2" >0.9939</td>
                        <td id="T_5258d_row2_col3" class="data row2 col3" >0.9992</td>
                        <td id="T_5258d_row2_col4" class="data row2 col4" >0.9857</td>
                        <td id="T_5258d_row2_col5" class="data row2 col5" >0.9823</td>
                        <td id="T_5258d_row2_col6" class="data row2 col6" >0.9996</td>
                        <td id="T_5258d_row2_col7" class="data row2 col7" >0.9850</td>
                        <td id="T_5258d_row2_col8" class="data row2 col8" >0.9815</td>
            </tr>
    </tbody></table></div>


<style>
#perf-table-wrap {
    display: flex !important;
    justify-content: center;
    margin: auto !important;
    align-items: center !important;
    padding: 10px;
}
</style>

<div id='perf-table-wrap'>
    <style  type="text/css" >
</style><table id="T_657cf_" style='margin:10px !important;'><caption>Averaged Performance</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >F1 Score</th>        <th class="col_heading level0 col2" >AUC</th>    </tr>    <tr>        <th class="index_name level0" >Model</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_657cf_level0_row0" class="row_heading level0 row0" >Custom</th>
                        <td id="T_657cf_row0_col0" class="data row0 col0" >0.9925</td>
                        <td id="T_657cf_row0_col1" class="data row0 col1" >0.9907</td>
                        <td id="T_657cf_row0_col2" class="data row0 col2" >0.9991</td>
            </tr>
            <tr>
                        <th id="T_657cf_level0_row1" class="row_heading level0 row1" >Hsieh</th>
                        <td id="T_657cf_row1_col0" class="data row1 col0" >0.9910</td>
                        <td id="T_657cf_row1_col1" class="data row1 col1" >0.9888</td>
                        <td id="T_657cf_row1_col2" class="data row1 col2" >0.9993</td>
            </tr>
            <tr>
                        <th id="T_657cf_level0_row2" class="row_heading level0 row2" >MobileNetV2</th>
                        <td id="T_657cf_row2_col0" class="data row2 col0" >0.9904</td>
                        <td id="T_657cf_row2_col1" class="data row2 col1" >0.9881</td>
                        <td id="T_657cf_row2_col2" class="data row2 col2" >0.9994</td>
            </tr>
    </tbody></table><style  type="text/css" >
</style><table id="T_d1ef3_" style='margin:10px; !important;'><caption>Averaged Ensemble Performance</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Accuracy</th>        <th class="col_heading level0 col1" >F1 Score</th>        <th class="col_heading level0 col2" >AUC</th>    </tr>    <tr>        <th class="index_name level0" >Model</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_d1ef3_level0_row0" class="row_heading level0 row0" >Custom</th>
                        <td id="T_d1ef3_row0_col0" class="data row0 col0" >0.9938</td>
                        <td id="T_d1ef3_row0_col1" class="data row0 col1" >0.9922</td>
                        <td id="T_d1ef3_row0_col2" class="data row0 col2" >0.9994</td>
            </tr>
            <tr>
                        <th id="T_d1ef3_level0_row1" class="row_heading level0 row1" >MobileNetV2</th>
                        <td id="T_d1ef3_row1_col0" class="data row1 col0" >0.9963</td>
                        <td id="T_d1ef3_row1_col1" class="data row1 col1" >0.9954</td>
                        <td id="T_d1ef3_row1_col2" class="data row1 col2" >0.9999</td>
            </tr>
            <tr>
                        <th id="T_d1ef3_level0_row2" class="row_heading level0 row2" >Hsieh</th>
                        <td id="T_d1ef3_row2_col0" class="data row2 col0" >0.9956</td>
                        <td id="T_d1ef3_row2_col1" class="data row2 col1" >0.9945</td>
                        <td id="T_d1ef3_row2_col2" class="data row2 col2" >0.9998</td>
            </tr>
    </tbody></table></div>

## Conclusion

Coming soon!
