# Reflection

## Time Spent

| Task | Notebook | Hours |
|------|-------|-------|
| Exploratory Data Analysis | eda.ipynb | | $\sim$ 7h |
| Data Preprocessing | data_processing.ipynb | $\sim$ 5h |
| Discriminatory Analysis | discriminatory_analysis.ipynb | $\sim$ 4h |
| Biomarker Embedding | biomarker_embedding.ipynb | $\sim$ 12h |
| **Total** | ------- | **$\sim$ 28h** |

## Challenges

This assignment was very challenging for me primarily because I had no prior experience with this type of data. I had to invest significant time researching the domain and reading documentation just to build a basic understanding of what the data represented. I also relied on LLMs to help simplify explanations of how the data was produced, since I have no expertise in chemistry and the technical terminology in the README and the associated literature was quite dense.

I could only start the assignment about a week after the interview, which left me with limited time. To compensate, I used a coding agent (Claude Code) to speed up the implementation, particularly for data visualization, and copy pasted some code that were already implemented in previous course assignments I took (from the course Foundation Models and Generative AI), like the multi-head attention and BERT, but all the analytical ideas and interpretations were my own.

I also did not have the time or the computational resources to properly tune the hyperparameters of the deep learning model, so the results are far from optimal. That said, it was encouraging to see that even with a rough setup, the embedding visualization produced by my model was cleaner than what the glycowork baseline produced, and comparable results on some downstream tasks with no hyperparameter tuning and very few epochs. 

I also, due to time and resource constraints, didn't monitor a validation loss during the training and plot a validation learning curve to see if my model is overfitting. But due, to the low amount of data, in order to do that well, it needs to be a k-fold cross-validation which would've taken too much time.

I was very excited about the GlycoBERT idea, that I put all my focus into it from the beginning and not try simple machine learning models first.

## Feedback
Overall, I really enjoyed this assignment. It has been a real challenge for me, and I learned a lot from it, especially in terms of data exploration and processing.
