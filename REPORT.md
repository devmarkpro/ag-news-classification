# AG News Classification: A Comparative Analysis of Machine Learning Approaches

## Problem Statement

With so much news content being published online every day, we need smart systems that can automatically sort articles into different categories. This project tackles exactly that problem using the AG News dataset, which contains news articles from four main areas: World news, Sports, Business, and Science/Technology.

I wanted to find out which machine learning approach works best for this task. So I tested three different methods: a traditional approach using Logistic Regression, a neural network called TextCNN, and a modern transformer model called DistilBERT. My goal was to see how well each one performs, how much computing power they need, and which one would be most practical to use in real applications.

What makes this study interesting is that I didn't just run basic experiments. I used advanced optimization techniques, especially with the TextCNN model where I employed Optuna to automatically find the best settings. This gave me some surprising results that I'll share throughout this report.

## Dataset Description and Analysis

### Dataset Characteristics

I used the AG News dataset for all my experiments. It's a well-balanced dataset with 127,600 news articles split evenly across four categories, with 120,000 for training and 7,600 for testing. Each article has both a title and description, which gives us good text content to work with.

When I first explored the dataset, I discovered some interesting patterns:

**Text Length**: Most articles are around 240 characters long with about 40 words each. This consistency across categories made me realize that the models would need to focus on content rather than just article length to make good predictions.

**Perfect Balance**: What I really liked about this dataset is that it has exactly the same number of articles in each category (31,900 each). This means I didn't have to worry about the models being biased toward one type of news.

**Different Vocabularies**: Each category has its own special words. Sports articles talk about teams and games, Business articles mention companies and markets, World news discusses countries and politics, and Science/Technology articles use technical terms. This made me confident that the models could learn to distinguish between categories based on these word patterns.

**Reading Level**: All articles are written at about the same difficulty level (college-level), so the models wouldn't get confused by different writing styles - they could focus on the actual content.

## Exploratory Data Analysis

Before jumping into building models, I spent time really understanding what I was working with. I created several visualizations and ran statistical analyses to get a feel for the data.

### Statistical Analysis

Looking at the statistics, I found that the articles are quite similar in length across categories. World news averages 243 characters, Business 242, Science/Technology 238, and Sports 226. The small differences told me that length wouldn't be a useful feature for classification.

Word counts were also consistent, ranging from 35 to 42 words per article. Interestingly, Sports articles were the most consistent in length, while Science/Technology articles varied more - probably because some tech articles are about simple apps while others discuss complex research.

### Vocabulary and Content Analysis

Word frequency analysis identified distinct vocabulary patterns for each news category. World news articles frequently featured terms related to government, countries, and international relations. Sports articles contained team names, player references, and performance-related terminology. Business articles emphasized financial terms, company names, and market-related vocabulary. Science/Technology articles included technical terminology, research-related words, and innovation concepts.

The analysis revealed minimal overlap in category-specific terminology, supporting the hypothesis that vocabulary-based classification approaches would be effective. Stop word removal and frequency analysis identified the most discriminative terms for each category, providing insights for feature engineering in traditional machine learning approaches.

### Topic Modeling Analysis

Latent Dirichlet Allocation topic modeling was applied to discover underlying thematic structures within the dataset. The analysis identified six distinct topics that aligned closely with the four predefined categories, with some topics representing subcategories within broader news areas. This finding validated the dataset's coherent structure and suggested that unsupervised methods could potentially identify meaningful categorizations.

The topic modeling results showed clear semantic separation between categories, with minimal topic overlap. World news topics focused on political events and international affairs, Sports topics centered on competitions and athletic performance, Business topics emphasized economic activities and corporate news, while Science/Technology topics highlighted research developments and technological innovations.

### Correlation and Feature Analysis

Correlation analysis between different text features revealed interesting relationships. Title length and description length showed moderate positive correlation (r=0.45), indicating that longer titles tend to accompany longer descriptions. However, the correlation between text length and category assignment was minimal, suggesting that classification success would depend on content rather than length characteristics.

Feature correlation analysis informed preprocessing decisions for each modeling approach. The weak correlation between simple length features and categories supported the need for more sophisticated feature extraction methods, such as TF-IDF vectorization for traditional approaches and learned embeddings for neural network models.

### Visualization and Pattern Recognition

Comprehensive visualizations were generated to illustrate dataset characteristics and support analytical findings. The following figures provide detailed insights into the dataset structure and content patterns.

#### Figure 1: Overview Dashboard

![Overview Dashboard](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/eda/overview_dashboard.png)

The overview dashboard presents six complementary visualizations that establish the fundamental characteristics of the AG News dataset. The class distribution pie chart (top-left) confirms perfect balance across all four categories, with each class containing exactly 25% of the total samples. This balanced distribution eliminates potential bias concerns and ensures fair model evaluation.

The text length distribution histogram (top-center) reveals a normal distribution centered around 240 characters, with minimal skewness. This consistency indicates standardized article formatting across the dataset. The word count box plots (top-right) demonstrate similar patterns across categories, with Sports articles showing the most consistent length and Science/Technology articles exhibiting the greatest variability.

The train-test split visualization (bottom-left) confirms that the balanced distribution is maintained across both training and testing sets, ensuring representative evaluation. The title versus description length scatter plot (bottom-center) shows moderate positive correlation, indicating that longer titles generally accompany more detailed descriptions. The violin plots (bottom-right) provide detailed distribution shapes, revealing that all categories follow similar length patterns with slight variations in spread.

#### Figure 2: Word Clouds Analysis

![Word Clouds](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/eda/word_clouds_professional.png)

The word cloud visualizations provide intuitive representations of vocabulary differences across news categories. Each quadrant represents a distinct category, with word size proportional to frequency within that category.

The World news word cloud (top-left) prominently features terms related to international affairs, government, and geopolitical events. Key terms include "government," "country," "international," "president," and "minister," reflecting the global political focus of this category.

The Sports word cloud (top-right) emphasizes competitive terminology and performance metrics. Dominant words include "team," "game," "player," "season," "win," and "championship," clearly distinguishing sports content from other categories.

The Business word cloud (bottom-left) highlights financial and corporate terminology. Frequent terms include "company," "market," "business," "financial," "economic," and "industry," demonstrating the commercial focus of this category.

The Science/Technology word cloud (bottom-right) showcases technical and research-oriented vocabulary. Key terms include "technology," "research," "development," "computer," "software," and "innovation," reflecting the technological and scientific nature of these articles.

The minimal overlap between category-specific terms validates the hypothesis that vocabulary-based classification approaches would be highly effective for this dataset.

#### Figure 3: Top Words Analysis

![Top Words Analysis](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/eda/top_words_analysis.png)

The top words analysis provides quantitative frequency analysis for the most discriminative terms in each category. Each subplot displays the 15 most frequent words after stop word removal, with horizontal bars indicating relative frequency.

World news demonstrates high frequency of political and geographical terms, with "government" and "country" appearing most frequently. The presence of terms like "international," "president," and "minister" reinforces the political nature of this category.

Sports articles show clear dominance of competitive terminology, with "team" and "game" leading the frequency rankings. The consistent appearance of performance-related terms like "player," "season," and "win" creates a distinct vocabulary signature.

Business articles emphasize commercial terminology, with "company" and "market" showing highest frequencies. The presence of financial terms like "business," "economic," and "industry" creates clear separation from other categories.

Science/Technology articles feature technical vocabulary, with "technology" and "research" dominating the frequency rankings. Terms like "computer," "software," and "development" establish the technological focus of this category.

The frequency analysis reveals minimal vocabulary overlap between categories, supporting the effectiveness of term-based classification approaches and validating the dataset's semantic coherence.

#### Figure 4: Statistical Analysis Summary

![Statistical Analysis](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/eda/statistical_analysis.png)

The statistical analysis summary provides comprehensive quantitative insights through six complementary visualizations. The heatmap (top-left) displays average statistics across categories, revealing subtle but consistent differences in text characteristics. World and Business articles show similar length patterns, while Sports articles tend to be slightly shorter and Science/Technology articles exhibit greater variability.

The text length distribution box plots (top-center) illustrate the consistency of article lengths across categories, with all categories showing similar median values and interquartile ranges. The word count distribution (top-right) demonstrates parallel patterns, confirming that length differences are minimal and unlikely to serve as primary classification features.

The correlation matrix (bottom-left) reveals the relationships between different text features. The moderate correlation between title and description lengths (r=0.45) indicates some relationship between these components, while the weak correlation with category assignment confirms that classification must rely on content rather than structural features.

The class distribution bar chart (bottom-center) provides another confirmation of perfect balance across categories, with each class containing exactly 31,900 samples. The title versus description scatter plot (bottom-right) visualizes the positive correlation between these length measures, with color coding revealing that this relationship is consistent across all categories.

These visualizations collectively demonstrate that the AG News dataset provides an ideal foundation for text classification research, with balanced classes, consistent structure, and clear content differentiation that supports effective model development and evaluation.

The exploratory analysis confirmed that the AG News dataset exhibits the characteristics necessary for effective text classification: balanced class distribution, distinct vocabulary patterns, consistent text structure, and clear semantic separation between categories. These findings informed the design and preprocessing decisions for all three modeling approaches.

## Experimental Setup

### Hardware Specifications

All experiments were conducted on a high-performance Apple Silicon system with the following specifications:

**System Configuration**:
- **Processor**: Apple M3 Max chip
  - 16-core CPU (12 performance cores + 4 efficiency cores)
  - 40-core GPU with hardware-accelerated machine learning
  - 16-core Neural Engine for AI workloads
- **Memory**: 48 GB unified memory architecture
- **Storage**: 1TB SSD with high-bandwidth memory controller
- **Operating System**: macOS Tahoe with Metal Performance Shaders (MPS) support

## Approach

Three distinct machine learning approaches were implemented and evaluated, representing different paradigms in text classification: traditional machine learning with feature engineering, deep learning with convolutional architectures, and transformer-based models.

### Logistic Regression Baseline Model

The baseline approach employs Logistic Regression with TF-IDF vectorization, achieving 91.53% test accuracy. This traditional machine learning method serves as a performance benchmark and demonstrates the effectiveness of well-engineered features combined with classical statistical learning approaches.

#### Data Preprocessing

Text preprocessing involves several critical steps to optimize feature extraction:

1. **Text Concatenation**: Article titles and descriptions are combined using period separation, creating comprehensive text representations while maintaining semantic boundaries.

2. **TF-IDF Vectorization**: The Term Frequency-Inverse Document Frequency approach transforms text into numerical features. Configuration parameters include:
   - N-gram range: (1,2) capturing both individual words and bigrams
   - Minimum document frequency: 2, filtering rare terms
   - Maximum features: 200,000, balancing vocabulary coverage with computational efficiency
   - Final vocabulary size: 200,000 features

3. **Feature Scaling**: TF-IDF naturally provides normalized features, eliminating the need for additional scaling procedures.

4. **Data Split**: Training data (96,000 samples) and validation data (24,000 samples) maintain balanced class distribution for reliable performance evaluation.

#### Model Architecture

Logistic Regression with L2 regularization provides a linear decision boundary in the high-dimensional TF-IDF space. The model employs one-vs-rest classification for multi-class prediction, training separate binary classifiers for each news category.

Key hyperparameters include:
- Maximum iterations: 2000, ensuring convergence
- Regularization: L2 penalty preventing overfitting
- Solver: Limited-memory BFGS for efficient optimization
- Multi-core processing: Utilized 8 workers on 16-core system for accelerated training

#### Performance Analysis

The baseline model achieves exceptional performance with remarkable computational efficiency:

**Overall Performance Metrics**:
- **Test Accuracy**: 91.53%
- **Validation Accuracy**: 91.88%
- **Test ROC AUC**: 0.9832
- **Validation ROC AUC**: 0.9853

**Detailed Classification Performance**:

The model demonstrates consistent performance across all categories with detailed per-class metrics:

**Test Set Performance by Category**:
- **World News (Class 0)**: Precision 93.06%, Recall 90.32%, F1-Score 91.67%
- **Sports (Class 1)**: Precision 94.91%, Recall 98.11%, F1-Score 96.48%
- **Business (Class 2)**: Precision 88.87%, Recall 87.84%, F1-Score 88.35%
- **Sci/Tech (Class 3)**: Precision 89.18%, Recall 89.84%, F1-Score 89.51%

**Performance Insights**:
1. **Sports Classification Excellence**: The model achieves the highest performance on Sports articles (96.48% F1-score), likely due to distinctive sports-specific vocabulary and terminology.

2. **Balanced Performance**: Macro-averaged precision, recall, and F1-scores all converge around 91.5%, indicating consistent performance across categories without significant bias.

3. **Business Category Challenge**: Business articles show slightly lower performance (88.35% F1-score), possibly due to vocabulary overlap with World news in economic and political contexts.

4. **ROC AUC Excellence**: The 98.32% test ROC AUC indicates exceptional discriminative capability, demonstrating that TF-IDF features effectively separate news categories in the feature space.

#### Performance Visualizations

The following figures demonstrate the baseline model's classification performance:

##### Figure 7: Baseline Logistic Regression Confusion Matrix

![Baseline Confusion Matrix](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/baseline_lr/plots/confusion_matrix_test.png)

The confusion matrix reveals strong diagonal dominance, indicating accurate classification across all categories. Sports articles show the highest classification accuracy with minimal confusion with other categories. The most frequent misclassifications occur between World and Business news, reflecting their semantic similarity in covering economic and political topics. Science/Technology articles demonstrate clear separation from other categories, validating the effectiveness of technical vocabulary in classification.

##### Figure 8: Baseline Logistic Regression ROC Curves

![Baseline ROC Curves](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/baseline_lr/plots/roc_curves_test.png)

The ROC curves demonstrate exceptional discriminative performance with all individual class AUC scores exceeding 0.97. Sports articles achieve the highest individual AUC (approaching 0.99), while all categories maintain excellent separation from others. The curves' steep rise and proximity to the top-left corner confirm the model's ability to achieve high sensitivity while maintaining low false positive rates across all news categories.

#### Model Efficiency and Scalability

The baseline model demonstrates several practical advantages:

1. **Training Efficiency**: Complete training pipeline executes within minutes on standard hardware, making it suitable for rapid prototyping and resource-constrained environments.

2. **Memory Efficiency**: Despite the 200,000-feature TF-IDF representation, the sparse matrix implementation ensures efficient memory utilization.

3. **Interpretability**: Linear model coefficients provide direct insight into feature importance, enabling analysis of which terms most strongly influence each category's classification.

4. **Scalability**: Multi-core processing capabilities (8 workers utilized) demonstrate the model's ability to leverage modern hardware for accelerated training.

The exceptional performance of this baseline model (91.53% accuracy, 98.32% ROC AUC) establishes a strong foundation for comparison with more complex deep learning approaches, demonstrating that well-engineered traditional features can achieve remarkable results in text classification tasks.

### TextCNN Architecture with Hyperparameter Optimization

This is where things got really interesting. I decided to try a convolutional neural network designed specifically for text, called TextCNN. What made this experiment special was that I used Optuna, an advanced optimization tool, to automatically find the best settings. The final result was 84.43% accuracy on the test set, but the journey to get there taught me a lot about the challenges of neural network optimization.

#### Architecture Design

Think of TextCNN like a smart reader that looks for different patterns in text. Here's how I built it:

1. **Word Embeddings**: First, I convert each word into a 200-dimensional vector. Through Optuna's optimization, I found that 200 dimensions worked better than 100 or 300 - it gave the model enough information without making it too complex.

2. **Convolutional Filters**: This is the clever part. I use filters of different sizes (2, 3, 4, and 5 words) to catch different patterns. A 2-word filter might catch "stock market," while a 5-word filter could catch "the president announced today that." Optuna found that 256 channels worked best - more than the typical 128 I started with.

3. **Max Pooling**: From all the patterns each filter finds, I keep only the strongest signal. This helps the model focus on the most important features.

4. **Final Classification**: A simple layer with 50% dropout makes the final decision. The dropout helps prevent the model from memorizing the training data too much.

#### Data Preprocessing

TextCNN preprocessing differs significantly from traditional approaches:

1. **Tokenization**: Text undergoes word-level tokenization, creating vocabulary mappings for neural network processing.

2. **Vocabulary Construction**: Custom vocabulary building with minimum frequency thresholds (2 occurrences) and maximum vocabulary size (50,000 words) balances coverage with computational requirements.

3. **Sequence Padding**: Variable-length texts are padded to uniform length (200 tokens), enabling efficient batch processing while preserving semantic content.

4. **Label Encoding**: Category labels are converted to zero-indexed integers for neural network compatibility.

#### Hyperparameter Optimization with Optuna

Here's where I got really excited about this project. Instead of guessing the best settings, I let Optuna do the hard work. Optuna is like having a smart assistant that tries different combinations of settings and learns from each attempt.

**My Strategy**: I set up 20 different experiments, each potentially running for 20 training rounds. But here's the clever part - Optuna uses something called MedianPruner that stops bad experiments early. If an experiment is performing worse than average after a few rounds, it gets cancelled. This saved me tons of computing time!

**What Optuna Tested**: I gave Optuna a huge space to explore - 10 different settings with multiple options each:
- **Word vector size**: 100, 200, or 300 dimensions
- **Filter channels**: 128, 192, or 256 
- **Filter sizes**: Different combinations like (3,4,5) or (2,3,4,5)
- **Dropout**: From 10% to 60%
- **Learning rate**: From very slow (0.0001) to fast (0.005)
- **Batch size**: 64, 128, or 256 articles at once
- **Text length**: 160, 200, or 256 words maximum
- **And more**: Gradient clipping and optimizer choice (Adam vs AdamW)

This created thousands of possible combinations!

**What Happened During Optimization**:

The results were fascinating! Out of 20 trials, Optuna only let 6 finish completely - it stopped the other 14 early because they weren't promising. This saved me about 60% of the computing time, which took about 4.5 hours total.

**The Winner** (Trial 3): Got 92.31% accuracy on validation data with these settings:
- **Word vectors**: 200 dimensions (the middle option)
- **Channels**: 256 (the highest option)
- **Filter sizes**: (2, 3, 4, 5) - including 2-word patterns was key!
- **Dropout**: 50% (moderate regularization)
- **Learning rate**: 0.00198 (quite moderate)
- **Batch size**: 256 (the largest option)
- **Text length**: 200 words
- **Optimizer**: Adam (beat AdamW in this case)

**What I Learned**:
- **Huge variation**: The worst trial got only 68.68% while the best got 92.31% - this shows how much the settings matter!
- **Pruning works**: Stopping bad experiments early saved me tons of time without missing good solutions
- **Surprising discoveries**:
  - Including 2-word filters (not just 3,4,5) made a big difference
  - More channels (256) was better than the typical 128
  - Larger batches (256) helped the model learn more stably
  - Moderate learning rates worked best - not too fast, not too slow

#### Model Performance Analysis

Here's where I learned an important lesson about machine learning. The model looked amazing during optimization - 92.31% accuracy on validation data! But when I tested it on completely new data, it only got 84.43%. 

**The Numbers**:
- **Validation**: 92.30% (looked great!)
- **Test**: 84.43% (the reality)
- **Gap**: 7.87% difference (this was concerning)
- **Training time**: 14 rounds before it stopped improving

This gap taught me that even with sophisticated optimization like Optuna, you can still overfit to your validation data. The model got really good at the validation set but struggled with truly new examples.

#### Performance Visualizations

The following figures demonstrate the TextCNN model's classification performance:

##### Figure 5: TextCNN Confusion Matrix

![TextCNN Confusion Matrix](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/textcnn/plots/confusion_matrix_test.png)

The confusion matrix reveals excellent classification performance across all news categories. The model demonstrates particularly strong performance in distinguishing between Sports and other categories, with minimal misclassification. The diagonal dominance indicates robust category separation, with most errors occurring between semantically related categories such as World and Business news.

##### Figure 6: TextCNN ROC Curves

![TextCNN ROC Curves](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/textcnn/plots/roc_curves_test.png)

The ROC curves demonstrate exceptional discriminative capability with an overall AUC of 0.9824. All individual class AUC scores exceed 0.97, indicating excellent binary classification performance for each category versus all others. The curves' proximity to the top-left corner confirms the model's ability to achieve high true positive rates while maintaining low false positive rates across all categories.

#### Optuna Training Dynamics Visualization

The following figures provide detailed insights into the Optuna optimization process and training dynamics of the best-performing TextCNN model:

##### Figure 7: Best Model Training Progress - Validation Accuracy

![Best Model Validation Accuracy](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/textcnn/plots/optuna/best_model_val_acc.png)

This plot shows the validation accuracy progression of the best-performing model (Trial 3) throughout its training epochs. The curve demonstrates rapid initial learning, with accuracy climbing from approximately 82% to over 90% within the first 40 training steps. The model then continues to improve more gradually, reaching its peak validation accuracy of 92.31% around step 160. The smooth, monotonic increase indicates stable training dynamics without significant oscillations or overfitting during the training process.

##### Figure 8: Best Model Training Progress - Validation Loss

![Best Model Validation Loss](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/textcnn/plots/optuna/best_model_val_loss.png)

The validation loss curve complements the accuracy plot, showing a steep initial decrease from approximately 0.67 to 0.43 within the first 60 steps, followed by gradual convergence to around 0.41. The smooth exponential decay pattern indicates effective learning without instability. The loss stabilization after step 80 suggests the model reached its optimal performance capacity, with minimal improvement in later epochs.

##### Figure 9: Optuna Trial Comparison - Validation Accuracy Across All Trials

![Validation Accuracy Across Trials](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/textcnn/plots/optuna/validation_accuracy.png)

This comprehensive view displays the validation accuracy trajectories for the first 10 Optuna trials, illustrating the dramatic performance variation across different hyperparameter configurations. The final model (teal line) demonstrates superior and stable performance, consistently achieving the highest accuracy. Several trials (trial_019, trial_018, trial_016) show competitive performance around 90-91%, while others exhibit significantly lower performance or early termination due to Optuna's pruning mechanism. Trial_013 (green line) shows particularly poor performance, starting at only 35% accuracy, demonstrating how critical proper hyperparameter selection is for TextCNN performance.

##### Figure 10: Optuna Trial Comparison - Validation Loss Across All Trials

![Validation Loss Across Trials](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/textcnn/plots/optuna/validation_loss.png)

The validation loss comparison reveals the optimization landscape across different trials. The final model maintains the lowest and most stable loss trajectory (teal line), converging to approximately 0.41. Poor-performing trials like trial_013 show high initial loss (>1.3) and unstable training dynamics. The plot clearly illustrates how Optuna's MedianPruner effectively identified and terminated unpromising trials early, as evidenced by the truncated lines for several trials. The diversity in loss trajectories highlights the sensitivity of TextCNN performance to hyperparameter choices and validates Optuna's role in navigating this complex optimization space.

#### Discussion of Results

The Optuna-optimized TextCNN demonstrates sophisticated pattern recognition capabilities, though with notable generalization challenges:

1. **Extended Convolutional Filters**: The optimal (2, 3, 4, 5) kernel size combination captures a broader range of n-gram patterns from bigrams to 5-grams, enabling detection of both short phrases and longer contextual patterns more effectively than traditional configurations.

2. **Enhanced Channel Architecture**: 256 channels provided superior feature extraction capacity, suggesting that news classification benefits from increased representational complexity to capture nuanced category distinctions.

3. **Optimization Insights**: The comprehensive Optuna search revealed critical hyperparameter interactions:
   - **Batch size scaling**: Larger batches (256) improved gradient stability and convergence
   - **Learning rate sensitivity**: Moderate rates (~0.002) prevented both underfitting and instability
   - **Regularization balance**: 0.5 dropout with minimal weight decay achieved optimal bias-variance tradeoff

4. **Generalization Analysis**: The 7.87% validation-test gap indicates potential overfitting to the validation set, suggesting the need for:
   - **Cross-validation**: More robust validation strategies for hyperparameter selection
   - **Regularization enhancement**: Additional techniques like data augmentation or ensemble methods
   - **Architecture refinement**: Potential benefits from batch normalization or residual connections

5. **Computational Efficiency**: Despite increased complexity, the optimized architecture maintained reasonable training times (~4.5 hours for full optimization) while exploring a comprehensive hyperparameter space.

The architecture successfully identifies category-specific terminology and phrase patterns, with Optuna optimization revealing that news classification benefits from more sophisticated convolutional architectures than traditional text classification tasks. However, the generalization gap highlights the importance of robust validation strategies in hyperparameter optimization.

### DistilBERT Transformer Model

The DistilBERT approach represents state-of-the-art transformer architecture, achieving 94.70% test accuracy through pre-trained language understanding and systematic fine-tuning optimization.

#### Architecture Overview

DistilBERT employs a distilled version of BERT, maintaining 97% of BERT's performance while reducing computational requirements by 40%. The architecture features:

1. **Transformer Encoder**: Six transformer layers with multi-head attention mechanisms capture complex linguistic relationships and contextual dependencies.

2. **Pre-trained Representations**: Extensive pre-training on large text corpora provides robust language understanding, enabling effective transfer learning for news classification.

3. **Classification Head**: A linear layer with dropout regularization performs final category prediction, leveraging rich contextual representations from the transformer backbone.

#### Data Preprocessing

DistilBERT preprocessing leverages specialized tokenization and encoding:

1. **Tokenization**: WordPiece tokenization handles out-of-vocabulary words through subword segmentation, ensuring comprehensive text representation.

2. **Special Tokens**: `[CLS]` tokens enable classification representation, while `[SEP]` tokens separate different text segments when applicable.

3. **Attention Masks**: Binary masks distinguish actual tokens from padding, ensuring accurate attention computation during training.

4. **Sequence Truncation**: Text sequences are limited to 256 tokens, balancing computational efficiency with content preservation.

5. **Multi-core Processing**: Utilized 4 workers on 16-core system for efficient data processing during training.

#### Training Strategy and Optimization

Fine-tuning employed systematic hyperparameter optimization with careful monitoring of training dynamics:

**Training Configuration**:
- **Learning Rate**: 2e-5 with linear decay schedule, providing stable fine-tuning without catastrophic forgetting
- **Batch Size**: 32 samples per batch, balancing memory requirements with gradient stability
- **Training Epochs**: 3 epochs with comprehensive evaluation after each epoch
- **Optimization**: AdamW optimizer with weight decay (0.01) for stable convergence and regularization

**Training Dynamics Analysis**:

The training process demonstrated excellent convergence characteristics:

1. **Epoch 1 Performance**: Achieved 94.11% validation accuracy with 94.10% macro F1-score, indicating rapid adaptation to the news classification task.

2. **Epoch 2 Improvement**: Validation accuracy increased to 94.75% with 94.76% macro F1-score, demonstrating continued learning without overfitting.

3. **Epoch 3 Stabilization**: Final validation accuracy of 94.94% with 94.94% macro F1-score, showing optimal convergence.

4. **Loss Progression**: Training loss decreased consistently from 0.90 to 0.08 over 3 epochs, while validation loss stabilized around 0.17, indicating excellent generalization.

5. **Gradient Stability**: Gradient norms remained stable throughout training (typically 1-7), confirming proper optimization dynamics.

#### Final Model Performance

The optimized DistilBERT model achieved exceptional performance across all metrics:

**Overall Performance Metrics**:
- **Test Accuracy**: 94.70%
- **Validation Accuracy**: 94.94%
- **Test F1-Score (Macro)**: 94.70%
- **Validation F1-Score (Macro)**: 94.94%
- **Test ROC AUC (Macro)**: 99.32%
- **Validation ROC AUC (Macro)**: 99.39%

**Training Efficiency**:
- **Total Training Time**: 65.3 minutes (3,919 seconds)
- **Training Throughput**: 73.47 samples/second
- **Evaluation Speed**: 232.46 samples/second on validation, 144.48 samples/second on test

#### Performance Visualizations

The following figures demonstrate the DistilBERT model's superior classification performance:

##### Figure 9: DistilBERT Confusion Matrix

![DistilBERT Confusion Matrix](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/distilbert/plots/confusion_matrix_test.png)

The confusion matrix reveals exceptional classification performance with strong diagonal dominance across all categories. DistilBERT demonstrates superior accuracy compared to previous models, with minimal misclassification errors. The model shows particularly strong performance in distinguishing between all news categories, with most confusion occurring between semantically related World and Business news. The high accuracy across all categories validates the effectiveness of pre-trained transformer representations for news classification.

##### Figure 10: DistilBERT ROC Curves

![DistilBERT ROC Curves](https://raw.githubusercontent.com/devmarkpro/ag-news-classification/main/outputs/distilbert/plots/roc_curves_test.png)

The ROC curves demonstrate exceptional discriminative capability with an outstanding macro AUC of 0.9932 and micro AUC of 0.9943. All individual class curves approach the ideal top-left corner, indicating near-perfect separation between categories. The consistently high AUC scores across all classes (exceeding 0.99) confirm DistilBERT's superior ability to distinguish between news categories compared to traditional and CNN-based approaches.

#### Architecture Advantages and Analysis

DistilBERT's superior performance stems from several key architectural advantages:

1. **Contextual Understanding**: Self-attention mechanisms capture long-range dependencies and contextual relationships, enabling nuanced understanding of article content that surpasses n-gram based approaches.

2. **Transfer Learning Excellence**: Pre-trained representations on large corpora provide robust linguistic knowledge, enabling rapid adaptation to news classification with minimal task-specific training.

3. **Bidirectional Processing**: Unlike sequential models, DistilBERT processes text bidirectionally, capturing both forward and backward contextual information for comprehensive understanding.

4. **Subword Tokenization**: WordPiece tokenization handles domain-specific terminology and rare words effectively, crucial for news classification across diverse topics.

5. **Attention-based Feature Learning**: Multi-head attention mechanisms automatically identify and weight important textual features, eliminating the need for manual feature engineering.

#### Computational Efficiency

Despite its superior performance, DistilBERT maintains reasonable computational requirements:

- **Model Size**: Reduced by 40% compared to full BERT while maintaining 97% of performance
- **Training Speed**: 73.47 samples/second enables practical fine-tuning on modern hardware
- **Memory Efficiency**: Optimized architecture allows training with standard GPU memory constraints
- **Inference Speed**: 144-232 samples/second enables real-time classification applications

The exceptional performance of DistilBERT (94.70% accuracy, 99.32% ROC AUC) establishes it as the superior approach for news classification, demonstrating the effectiveness of pre-trained transformer models in capturing complex linguistic patterns and semantic relationships inherent in news text classification tasks.

## Results and Discussion

Working with these three different approaches taught me some surprising things about text classification that I didn't expect when I started this university project.

### Comparative Performance Analysis

The final scores really challenged what I thought would happen:

**Simple Logistic Regression (91.53% accuracy)**: This was my "simple" baseline, but it actually performed better than my fancy neural network! It was fast to train, easy to understand, and worked really well. Sometimes the old methods are still the best.

**TextCNN with Smart Optimization (84.43% test accuracy, but 92.31% on validation)**: This was the most interesting part of my project. I used Optuna to automatically find the best settings, and it looked amazing during training. But then it didn't work as well on new data. This taught me that even smart optimization can trick you - your model might just be memorizing the validation data instead of really learning.

**DistilBERT Transformer (94.70% accuracy)**: This was the clear winner. It's a pre-trained model that already knows a lot about language, so it could understand the news articles much better than the other approaches.

### Key Findings

1. **Validation vs. Real Performance**: Just because your model looks good during training doesn't mean it will work well on new data. My TextCNN got 92% on validation but only 84% on the test set. This was a big lesson about overfitting.

2. **Simple Can Beat Complex**: My basic Logistic Regression actually beat the neural network! It was faster, easier to understand, and more reliable. This showed me that newer doesn't always mean better.

3. **Smart Optimization Tools Work**: Optuna saved me lots of time by stopping bad experiments early. It tested 20 different settings but only let 6 finish completely, saving about 60% of computing time.

4. **Pre-trained Models Are Powerful**: DistilBERT won because it already learned from millions of texts before I even started. It's like having a student who already knows a lot about language before taking the news classification test.

### Practical Implications

**If you want something reliable and fast**: Use Logistic Regression. It got 91.53% accuracy, trains quickly, and you can understand how it makes decisions.

**If you want to learn about neural networks**: Try TextCNN with Optuna optimization. You'll learn a lot about hyperparameter tuning, but be careful about overfitting to your validation data.

**If you need the best accuracy**: Go with DistilBERT. It got 94.70% accuracy and works well on new data, though it needs more computing power.

### Future Research Directions

1. **Better Validation**: Find ways to test hyperparameter optimization that better predict how well models will work on completely new data.

2. **Combining Methods**: Maybe combine the simple Logistic Regression with neural networks to get the best of both worlds.

3. **Reducing Overfitting**: Try new techniques to make neural networks generalize better, so the validation and test scores are closer.

4. **Lighter Transformers**: Look for ways to make models like DistilBERT even faster while keeping their accuracy.

This project taught me that doing good experiments is just as important as having fancy algorithms. Using tools like Optuna and proper evaluation methods helped me understand what really works and what just looks good on paper.
## Conclusion

This comparative study of three machine learning approaches for AG News Classification revealed several important insights about text classification performance and methodology.

  The experimental results showed that **DistilBERT achieved the highest accuracy at 94.70%**, demonstrating the power of pre-trained transformer models for text understanding. However, the study also revealed that **traditional Logistic Regression performed surprisingly well at 91.53%**, actually outperforming the optimized TextCNN model (84.43% test accuracy) despite being much simpler.

  The most significant finding was the **generalization gap observed in the TextCNN approach**. While Optuna optimization achieved 92.31% validation accuracy, the test performance dropped to 84.43%, highlighting the critical importance of robust validation strategies in hyperparameter optimization. This 7.87% gap serves as a valuable lesson about the risks of overfitting to validation data, even when using sophisticated optimization techniques.

  **Key contributions of this research include:**
 1. **Empirical comparison** of three distinct approaches on a balanced news classification dataset
 2. **Demonstration of Optuna's effectiveness** in automated hyperparameter optimization, achieving 60% computational savings through intelligent pruning
 3. **Evidence that traditional methods remain competitive** - Logistic Regression's strong performance challenges assumptions about deep learning superiority in all scenarios
 4. **Identification of generalization challenges** in neural network optimization for text classification

  The study confirms that **model selection should consider not only accuracy but also reliability, interpretability, and computational requirements**. For practical applications requiring fast, interpretable results, Logistic Regression remains highly viable. For maximum accuracy where computational resources are available, DistilBERT provides superior performance with robust generalization.

  This research contributes to the understanding of text classification methodologies and provides a framework for future comparing studies in natural language processing applications.

## References

[1] Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NeurIPS. https://arxiv.org/abs/1509.01626

[2] AG News Classification Dataset (Kaggle mirror). https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

[3] Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP. https://arxiv.org/abs/1408.5882

[4] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. https://arxiv.org/abs/1910.01108

[5] Wolf, T., Debut, L., Sanh, V., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP: System Demonstrations. https://arxiv.org/abs/1910.03771

[6] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD. https://doi.org/10.1145/3292500.3330701

[7] Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. JMLR, 3, 993–1022. https://jmlr.org/papers/v3/blei03a.html

[8] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

[9] Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS. https://papers.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

[10] DistilBERT base uncased model card. https://huggingface.co/distilbert-base-uncased

[11] PyTorch Tutorial: Word Embeddings – Encoding Lexical Semantics. https://docs.pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

[12] Optuna Tutorial. https://optuna.readthedocs.io/en/stable/tutorial/index.html

[13] PyTorch Tutorial: NLP From Scratch: Classifying Names with a Character-Level RNN. https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html