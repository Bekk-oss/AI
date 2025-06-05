Predicting the Stock Market with Stacked LSTM and NLP Techniques

Introduction
Predicting stock prices is notoriously challenging – markets are influenced by a myriad of factors, from
historical trends to breaking news. This project showcases a deep learning approach to forecasting stock
prices (specifically, next-day closing prices) by combining time-series analysis with natural language
processing. In simple terms, we built a hybrid model that uses a stacked bidirectional LSTM to learn
patterns from historical stock data and a fine-tuned DistilBERT model to glean insights from news text.
By integrating both numeric data and news sentiment, the model aims to capture a more holistic
picture of the market’s state each day and produce more accurate predictions than models relying on
prices alone.
The motivation is straightforward: stock movements often reflect not just past prices but also investor
reactions to news and events. Classic methods either focus on technical indicators or basic sentiment
analysis, but here we blend the two in one neural network. The result is a forecasting tool that can
potentially detect when news is likely to drive a big price change and adjust its prediction accordingly.
This README walks through the project’s approach, including the data used, the model architecture,
results achieved, and key takeaways for future development.
Dataset and Problem Set
Data Sources: We leverage two primary data sources in this project – historical stock price data and
financial news articles. For the price data, we gathered daily historical prices (open, close, high, low,
volume) for our target stock. In our case study, we focused on Tesla (TSLA) as an example, using roughly
1.5 years of daily prices (about mid-2021 through early 2023) obtained via the Yahoo Finance API.
Each day’s record includes the closing price, which we want to predict for the next day. For the textual
data, we used a large financial news dataset (the FinSen dataset ), which contains news articles with
timestamps and tickers. We filtered this dataset to pull out articles related to our stock (e.g, any news
mentioning "Tesla" or the ticker "TSLA"). Each news article provides a title and content, along with
the publication date.

Data Preparation: We aligned the news with the stock prices by date. Essentially, for each trading day
in our dataset, we compile the relevant news up to that day that could influence the stock. We then pair
that day’s news content with the historical price sequence. We also normalize the numeric features
(such as prices) to make training easier. In the Tesla example, our final time-aligned dataset spans
628 trading days, each with a normalized price vector and any news articles from that day or recent
days.

Prediction Task: The learning task is defined as a regression problem – predict the next trading day’s
closing price given all information up to the current day. During training, the model looks at the last N
days of stock data (e.g., a window of recent prices) and the corresponding news, and learns to output
the next day’s price. We optimize the model to minimize the Root Mean Squared Error (RMSE) between
our predictions and the actual closing prices. Ultimately, success is measured by how low the prediction error is on unseen test data (lower RMSE and MAE, higher $R^2$ indicating more variance explained).

Model Description

Our model is a hybrid deep neural network that fuses a recurrent model for time series with a
transformer-based language model for text. Here’s an overview of the architecture :
News Text Encoder – DistilBERT: We utilize DistilBERT (a lightweight version of BERT) to
process daily news articles. DistilBERT is a transformer model that captures contextual word
meanings but is smaller and faster than the original BERT, while retaining ~97% of BERT’s
language understanding capabilities. In our pipeline, we fine-tune DistilBERT on the financial
news data so it can better grasp finance-specific context and sentiment. Given a day’s news text
(e.g., concatenated headlines or an article summary), DistilBERT produces a 768-dimensional
vector embedding that represents the important information (topics, sentiment, keywords) of
that day’s news. This embedding is essentially a summary of “what happened in the news” in
numerical form. (If there’s no news on a given day, a neutral or zero vector can be used as a
placeholder so that the model doesn’t get biased by missing data.)
Stock Price Sequence Encoder – Stacked Bi-LSTM: For the numeric time-series data, we use a
stacked Bidirectional LSTM network. This is a recurrent neural network that reads the
sequence of recent daily prices (and any other technical features) both forward and backward in
time. The bidirectional aspect means the model can learn patterns that might be more obvious
when looking at the sequence in reverse, as well, although in practice we only predict forward,
This helps in learning more robust features. We stack two LSTM layers on top of each other:
The first layer processes the input sequence (we set its input size to match the 768-dim of the
text embedding, after combining text and numeric features) and produces intermediate hidden
states, and the second LSTM layer refines these representations further. Each LSTM layer has
128 hidden units in each direction, so when combined (forward + backward), we have 256-
dimensional hidden state outputs per time step. We apply dropout regularization between layers
to avoid overfitting. The outcome of this module is a sequence of hidden state vectors (one
per day in the look-back window), each encoding the recent price trend information (enriched by
any fused text info, as described below).


Model Architecture Diagram 

Potential Extensions: There are many ways to build on this work. The approach is quite flexible and
can be applied to other stocks or even broader market indices, especially in scenarios where news
heavily influences prices. Future developments could include scaling the model up to multiple
stocks simultaneously – for instance, training a single model that predicts several related stocks,
which could allow it to learn inter-stock relationships. We could also incorporate additional data
sources such as social media sentiment (Twitter feeds, Reddit discussions), earnings call transcripts, or
macroeconomic indicators to further enrich the input. Another idea is to experiment with more
advanced models: for example, using FinBERT (a BERT variant pre-trained on financial text) in place of
DistilBERT for potentially better textual understanding, or trying out a full Transformer-based time
series model instead of LSTM for the price data. These could potentially improve performance even
more, albeit with increased complexity.

In summary, Predicting Stock Market with Stacked LSTM + NLP demonstrates the value of combining
deep learning techniques from both the time-series and NLP domains to tackle a complex prediction
task. We showed that even for a highly volatile stock, incorporating news data can yield a tangible
improvement in forecast accuracy. We hope this project can serve as a stepping stone for others
interested in multi-modal financial modeling. Whether you’re looking to extend the model, apply it to
new data, or integrate it into a trading strategy, we welcome you to use the code, raise issues, and
contribute! Together, by blending qualitative and quantitative data, we can continue to enhance
financial predictions and gain deeper insights into what drives market movements.
