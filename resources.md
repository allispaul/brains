# Resources
Here's a place to put useful things we want to have easy access to later: Kaggle notebooks, stuff about EEGs, ML algorithms to try, etc.

## The datasets
- [Understanding Competition Data and EfficientNetB2 Starter - LB 0.43 🎉](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010) - Kaggle discussion post explaining how the dataset is defined.

## Training strategies
- [EffnetB0 model trained twice](https://www.kaggle.com/code/seanbearden/effnetb0-2-pop-model-train-twice-lb-0-39) - 0.39 LB from the insight that there are two distinct training populations, one with 1-7 judges, one with 10-28 judges. The space of possible probability distributions you can get with fewer judges is more sparse. He trains first on the few-judge population and then on the many-judge population.
  - This shows how useful EDA can be! Is there more to notice here -- for example, are there common disagreements that judges have, and do these differ between the two populations?

## Haven't read yet
Things one of us has found and hasn't had time to read. *Looking for something to do?* Read one of these, extract the important insights, move it out of this section, and if you're up for it, try to implement it!
- [Magic Formula to Convert EEG to Spectrograms](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/469760)
- [1D ResNet Architecture Baseline](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/471666) - claims 0.48 LB from EEGs
- [Grad Cam - What is Important in Spectrograms?](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/472976) - model interpretation tool with attached notebook
- [CatBoost Starter](https://www.kaggle.com/code/cdeotte/catboost-starter-lb-0-60) - 0.60 LB, another algorithm worth learning about
- [WaveNet Starter](https://www.kaggle.com/code/cdeotte/wavenet-starter-lb-0-52) - 0.52 LB, another algorithm worth learning about
- [Exploring EEG - A Beginner's Guide](https://www.kaggle.com/code/yorkyong/exploring-eeg-a-beginner-s-guide)
- [3 Model Ensemble](https://www.kaggle.com/code/kitsuha/3-model-ensemble-lb-0-37) - 0.37 LB, currently the highest-scoring public notebook. Many of the top scores seem to be similar ensembles.
