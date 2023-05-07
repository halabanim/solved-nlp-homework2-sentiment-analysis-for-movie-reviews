Download Link: https://assignmentchef.com/product/solved-nlp-homework2-sentiment-analysis-for-movie-reviews
<br>
<h1>1          Manual classification</h1>

To get started, read these two reviews, and decide which one is negative and which one is positive. Provide a short motivation, and try to anticipate what could pose an issue for automatic sentiment identification.

<strong>R1 </strong>Busy Phillips put in one hell of a performance, both comedic and dramatic. Erika Christensen was good but Busy stole the show. It was a nice touch after The Smokers, a movie starring Busy, which wasnt all that great. If Busy doesnt get a nomination of any kind for this film it would be a disaster. […]

<strong>R2 </strong>This movie was awful. The ending was absolutely horrible. There was no plot to the movie whatsoever. The only thing that was decent about the movie was the acting done by Robert DuVall and James Earl Jones. Their performances were excellent! The only problem was that the movie did not do their acting performances any justice. […]

<strong>What to submit: </strong>Your guesses, motivation &amp; possible issues for automatic sentiment identification.

<h1>Dataset</h1>

In this assignment you will use a dataset of movie reviews from the Internet Movie Database (IMDB)<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>.

The movies directory contains two subdirectories:

<ul>

 <li>train These documents will be used to train your language model. (600 docs)</li>

 <li>test These documents will be used to test your model. (50 docs) The documents are named as [sentiment]-[review ID].txt. The text file txt contains the correct labels for the documents in test. There is no need to modify the files or their directories. Load them using the provided Python 3 Notebook.</li>

</ul>

<h1>2          Tokenization</h1>

The first step is to tokenize the data. Tokenization splits up a character sequence into smaller pieces (tokens). An example tokenization is:

<strong>Original sentence: </strong>“If you have the chance, watch it. Although, a warning, you’ll cry your eyes out.”

<strong>Tokens: </strong>[If, you, have, the, chance, ,, watch, it, ., Although, ,, a, warning,

,, you, ‘ll, cry, your, eyes, out, .]

<h2>2.1          Making your own tokenizer</h2>

For this assignment, make a simple tokenizer. Write 3 sentences and try the tokenizer out on them.

<strong>What to submit: </strong>Provide a description of how your tokenizer works. Report the tokens you obtain when using your tokenizer on your example sentences.

<h2>2.2          Using an off-the-shelf tokenizer</h2>

Compare the tokenizer you implemented in the previous question with one from NLTK, using the sentences provided in the Notebook.

<strong>What to submit: </strong>Reflect and answer these questions: What are the differences in the two tokenizer outputs? Which one is better? While coding your tokenizer, did you foresee all these inputs? Is there a single ‘perfect tokenizer’?

<h2>2.3          Vocabulary</h2>

Run the NLTK tokenizer on all documents in the train directory and keep track of the unigram frequencies. Since our dataset is small, it is a good idea to apply heavy normalizations, for example removing punctuation and transforming each sentence to lowercase. After you implement the normalization, the sentence “If you have the chance, watch it. Although, a warning, you’ll cry your eyes out.” should look similar to this:

<strong>Normalized tokens: </strong>[if, you, have, the, chance, watch, it, although, a, warning, you, ‘ll, cry, your, eyes, out]

Answer the following questions using the documents in the train directory:

<ul>

 <li>How many unique n-grams are there? (where n=1,2,3).</li>

 <li>Report the top 10 most frequent words (unigrams) and their frequencies. What kind of word are these?</li>

 <li>How many words occur 1, 2, 3, 4 times in the corpus? Which kind of distribution is this?</li>

</ul>

Since words that do not occur often don’t add much information to the classification, keep only the words that occur at least 25 times as your vocabulary. Write your code such that all words that are not in your vocabulary are ignored in the rest of this assignment.

<strong>What to submit: </strong>Answer to the above 3 points.

<h1>3          Text classification with a unigram language model</h1>

Recall that for a text with words <em>w</em><sub>1 </sub><em>…w<sub>n </sub></em>, we calculate the probability as follows using a unigram language model:

<em>n</em>

<em>P</em>(<em>w</em><sub>1</sub><em>,w</em><sub>2</sub><em>,…,w<sub>n</sub></em>) = <em>P</em>(<em>w</em><sub>1</sub>)<em>P</em>(<em>w</em><sub>2</sub>)<em>…P</em>(<em>w<sub>n</sub></em>) = ∏<em>P</em>(<em>w<sub>i</sub></em>)

<em>i</em>=1

In order to avoid underflow, it is better to calculate this in log space (base 2)<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>:

<em>n                                  n</em>

log<em>P</em>(<em>w</em><sub>1</sub><em>,w</em><sub>2</sub><em>,…,w<sub>n</sub></em>) = log∏<em>P</em>(<em>w<sub>i</sub></em>) = ∑log<em>P</em>(<em>w<sub>i</sub></em>)

<em>i</em>=1                              <em>i</em>=1

In our dataset we have two classes: positive (Pos) and negative (Neg). For each class, we will calculate a separate language model. This is the training or learning phase. In the apply phase, we will classify new texts as positive or negative. For testing our machine learning classifier, we apply the models on the documents in the test part of the corpus.

<ol>

 <li><strong>TRAIN </strong>For the documents in the train directory, build two language models. One using the positive reviews, and one using the negative reviews. For example, we calculate the probability for the positive language model as follows.</li>

</ol>

<em>n</em>

<em>P</em>(<em>w</em><sub>1</sub><em>,w</em><sub>2</sub><em>,…,w<sub>n</sub></em>|<em>Pos</em>) = ∏<em>P</em>(<em>w<sub>i</sub></em>|<em>Pos</em>)

<em>i</em>=1

Where we are using the conditional probability (<em>P</em>(<em>w<sub>i</sub></em>|<em>Pos</em>) instead of just <em>P</em>(<em>w<sub>i</sub></em>)), because we are calculating the probabilities using only positive reviews. We estimate the conditional probabilities:

where <em>C</em>(<em>w<sub>i</sub>,Pos</em>) is the frequency of word <em>w<sub>i </sub></em>in the positive reviews and <em>w</em>∈<em>V </em><em>C</em>(<em>w,Pos</em>) the total number of words in the positive reviews<a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a>.

<strong>Smoothing </strong>Use smoothing to avoid zero probabilities:

Where <em>V </em>is the size of your vocabulary. Use two settings: when <em>k </em>= 1 and a value for <em>k </em>that you have selected yourself.

<ol start="2">

 <li><strong>TEST </strong>For the reviews in the test directory, calculate the probability for both language models. Assign each review the class for which it has the highest probability. Using the MAP (Maximum Aposteriori Probability) rule:</li>

</ol>

<a href="#_ftnref1" name="_ftn1">[1]</a> The dataset is a subset of the original dataset by Maas et al. The full dataset can be found at <a href="http://ai.stanford.edu/~amaas/data/sentiment/">http://ai.stanford.edu/~amaas/data/sentiment/</a>.

<a href="#_ftnref2" name="_ftn2">[2]</a> Since the probabilities are “small numbers”, the more we multiply them together, the smaller they become, up to a point where the computer cannot represent these number accurately anymore (<a href="https://en.wikipedia.org/wiki/Arithmetic_underflow">https://en.wikipedia.org/wiki/Arithmetic_underflow</a>). By moving everything in log space we sidestep the problem (recall that the logarithm of a product is the <em>sum </em>of the individual logarithms).

<a href="#_ftnref3" name="_ftn3">[3]</a> Take a few minutes to understand the formula, and reflect on the meaning of this sentence.