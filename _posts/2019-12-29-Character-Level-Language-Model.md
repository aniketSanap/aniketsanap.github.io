---
layout: post
title:  "Character level language model using LSTMs"
categories: [ NLP, LSTM, PyTorch, Python ]
# tags: []
image: assets/images/posts/CharacterLevelLanguageModel/cover_page.png
description: "Predicting the next character in a sequence"
featured: true
hidden: true
date: 2019-12-28
image_dir: assets/images/posts/CharacterLevelLanguageModel
---

Computer vision had its big boom at the start of this decade with the ImageNet Large Scale Visual Recognition Challenge. Deep CNN architectures were hands down the best approach to tackle vision problems. With the use of transfer learning, these models could be used for a variety of different tasks and applications which led to rapid progress in the field of computer vision. Transfer learning is commonly used in Natural Language Processing through building a language model first and then modifying it to suite our use case. This makes sense because human language is very complex. There is no one correct approach to writing or speaking anything. When we directly train our model on a language specific task, our model has no prior information about the complex structure or any of the nuances of human language. We cannot expect a model which has just been introduced to english to perform well on any language specific task directly without understanding the language first. Hence, we train a language model on a large dataset (like wikipedia) and then use this model which has an understanding of the language on our task.

A language model is a probability distribution over a sequence of tokens. In this post we will built a simple character level language model. Hence, the tokens for our model will be characters. Every language model needs a vocabulary. The vocabulary consists of all the tokens which are part of your input and which should be predicted by your output. For a character level language model, the vocabulary consists of every distinct character in your dataset. If you just want the code then you can find it on my [github](https://github.com/aniketSanap/CharacterLevelLanguageModel). Well then, lets start programming!

## The corpus

We obviously need a dataset. You can use any large corpus of data to train this model. As this is just a simple character based model, we will be using the text from [this site](https://norvig.com/big.txt) instead of using something like wikipedia. The text looks something like this:

```
ADVENTURE  I.  A SCANDAL IN BOHEMIA

I.


To Sherlock Holmes she is always the woman. I have seldom heard him mention her under any other name. 
...
```

We will start by reading the file and splitting it into a training set and a small validation set. We will only be using the validation set to keep an eye on it's loss and making sure that the model does not overfit to the training set. 

```python
with open('file.txt', 'r') as f:
    text = f.read()

text_size = len(text)
split_ratio = 0.9
train_text = text[:int(split_ratio * text_size)]
test_text = text[int(split_ratio * text_size):]

```

### Building the vocabulary

Creating the vocabulary is one of the first steps in preprocessing the text before passing it to your model. For our model, it contains a mapping of all the possible characters with a certain unique integer. The string to int mapping can be a dictionary with the characters as keys and the corresponding indices as values. The integer to character mapping can be a simple list where the index is implied. For a general range of characters you can use from `string import printable`. `printable` contains every printable character on your screen and your text probably won't contain any additional characters.
In code it looks like this:

```python
itos = list(printable)
stoi = {char:int_ for int_, char in enumerate(itos)}
```

## Preprocessing

#### One hot encoding

Generally text based tasks need a lot of preprocessing as you can't pass text directly to your model. In an attempt to keep this model simple, we will use a one hot encoded vector for our input over using word embeddings. The size of our one hot encoded vector will be the size of our vocabulary. For those unaware, a one hot encoded vector is simply a zero vector with a single 1 at the index of our character. We will create a separate one hot vector for every single character as input. 
Creating the one hot vector of a character will look something like this: 

```python
# Finding the index of the input char
index = stoi[char]

# Defining the size of the vector
size = len(itos)

# Creating the actual encoded vector
vector = torch.zeros(size)
vector[index] = 1
```

Here is our one hot encoder function which will work on an entire string

```python
def one_hot_encoder(sequence, stoi, sequence_length):
    size = len(stoi)
    encoded = []
    for int_ in sequence:
        temp = torch.zeros(size)
        temp[int_] = 1
        encoded.append(temp)

    for i in range(sequence_length - len(sequence) - 1):
        temp = torch.zeros(size)
        encoded.append(temp)

    return torch.stack(encoded)
```

You may have noticed that we used `sequence_length - len(sequence) - 1` for our loop instead of just `sequence_length - len(sequence)`. This will be explained shortly.


#### Splitting the data into sequences

As our dataset is very large, we will split it into sequences of length `sequence_length = 100` each. This will allow us to process multiple sequences in parallel leading to faster training. We will do this splitting as well as the one hot encoding during run time using PyTorch's datasets and dataloaders. A summary of what we have done so far can be observed through our dataset class.

```python
class LanguageModelDatset(Dataset):
    def __init__(self, text, sequence_length=100):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = 100
        self.text = text
        self.itos = list(printable)
        self.stoi = {char:int_ for int_, char in enumerate(self.itos)}
        self.text_size = len(self.text)
        
    def one_hot_encoder(self, sequence):
        size = len(self.stoi)

        encoded = []
        for int_ in sequence:
            temp = torch.zeros(size)
            temp[int_] = 1
            encoded.append(temp)

        for i in range(self.sequence_length - len(sequence) - 1):
            temp = torch.zeros(size)
            encoded.append(temp)

        return torch.stack(encoded)
        
    def __len__(self):
        return self.text_size // self.sequence_length
    
    def __getitem__(self, idx):
        sequence = self.text[idx * self.sequence_length:(idx + 1) * self.sequence_length]
        sequence = [self.stoi[x] for x in sequence]
        x = sequence[:-1]
        y = sequence[1:]
        x = self.one_hot_encoder(x)
        
        return x, torch.tensor(y)
```

If you don't know what PyTorch datasets and dataloaders are, I highly recommend you check them out. They are a great way of abstracting your data and PyTorch takes care of a lot of stuff (like parallel loading) under the hood. All you need to know for this post is that you have to define your own `__len__()` function to return the length of your dataset and your own `__getitem__()` function which takes an index parameter to return the i<sup>th</sup> value of your dataset. This allows you to index your dataset just like you would a list. In our `__getitem__()` function, we have to define our input and label (x and y). Our input is going to be a sequence of characters. The labels for a language model is the same sequence but offset by one word. This is better understood using code.

```py
input_ = ['H', 'E', 'L', 'L', 'O', ' ', 'W', 'O', 'R', 'L']
output = ['E', 'L', 'L', 'O', ' ', 'W', 'O', 'R', 'L', 'D']
```
This is because our language model always tries to predict the next character in the sequence given the previous characters. 
This is achieved by taking 100 characters from our corpus. The first 99 of which will become our x and the last 99 of which will become our y.

```py
x = sequence[:-1]
y = sequence[1:]
```
This is why we need to have `sequence_length - len(sequence) - 1` in our one hot encoder. Alternatively, we can just index 101 characters at a time.

## Building the model

We will be using an LSTM based architecture for this model. Our input size will be equal to the size of our one hot encoded vector. This is because our model will process one character at a time for each batch. It will do this sequentially for our entire sequence. The size of our output layer will also be equal to the size of our one hot vector. This is because the output tells us which character should be next. So the size has to be equal to the total number of characters in our vocabulary, which is of course, the size of our one hot vector.

Here is our model architecture in code:

```py
class CharLSTM(nn.Module):
    def __init__(self, hidden_size=256, n_hidden_layers=3, dropout_prob=0.5):
        super().__init__()
        self.legal_chars = printable
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_layers = n_hidden_layers
        
        self.lstm_layer = nn.LSTM(
            input_size=len(self.legal_chars), 
            hidden_size=hidden_size, 
            num_layers=n_hidden_layers, 
            dropout=dropout_prob, 
            batch_first=True
        )
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, len(self.legal_chars))
        self.to(self.device)

    def forward(self, x):
        self.hidden = self.hidden.detach()
        self.cell = self.cell.detach()
        x = x.to(self.device)
        lstm_output, (self.hidden, self.cell) = self.lstm_layer(x, (self.hidden, self.cell))
        output = self.dropout_layer(lstm_output)
        output = output.reshape(-1, self.hidden_size)
        return self.fc(output)

    def init_hidden(self, batch_size):
        # hidden state and cell state
        self.hidden, self.cell = [torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device), 
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)]
```

A couple of things to note while working with RNNs and their variants:
1. `init_hidden()`: This is a method which initializes the hidden and cell state for our LSTM. It will initially be set to zero and will change with every sequence. 
2. As we are keeping the same hidden state throughout, it is necessary to detach it from our graph to prevent BPTT (Backpropagation Through Time) over our entire dataset. 

## It's time to train!

We will be using the standard pytorch training loop with some modifications. 
1. `nn.utils.clip_grad_norm_(model.parameters(), clip)`: This is used to prevent the exploding gradient problem in RNNs. 
2. Initializing the hiddens state at the start of every epoch.
3. Ignoring the last batch of every epoch if its size is not equal to our `batch_size`. This is to prevent issues with hidden state size.

```py
opt = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
print_every = 500

for epoch in range(num_epochs):
    model.train()
    model.init_hidden(batch_size)
    counter = 0
    for x, y in train_dataloader:
        if x.shape[0] != batch_size:
            continue
        counter += 1
        opt.zero_grad()
        output = model(x)
        loss = criterion(output, y.long().view(-1,).to(model.device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()

        if counter % print_every == 0:
            print("Epoch: {}/{}...".format(epoch+1, num_epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.4f}...".format(loss.item()))
            
    with torch.no_grad():
        model.eval()
        test_loss = 0
        total = len(test_dataset)
        for x, y in test_dataloader:
            if x.shape[0] != batch_size:
                continue
            opt.zero_grad()
            output = model(x)
            loss = criterion(output, y.long().view(-1,).to(model.device))
            test_loss += loss.item()

        print(f'=' * 8 + f'\nTest loss: {test_loss/total}\n' + '=' * 8)
        if test_loss < max_test_loss:
            max_test_loss = test_loss
            save_weights(model, model_path)
```

I trained this model for 30 epochs while making sure that the validation loss was still decreasing. 

## Testing our model

To test our model, we will be writing 2 functions. The first one called `predict` will take a character and hidden state as input and return the next character and the next hidden state. The second function called `generate_chars` will take input a sequence of characters using which our model will start its story. This function will call predict `n_chars` number of times (`n_chars` will be a parameter which defines the length of the output sequence).

```py
def predict_next_char(model, char, h=None):
    global test_dataset
    model.eval()
    char_vector = test_dataset.one_hot_encoder([char])[0]

    # shape: (batch_size, sequence_length, vector_size)
    char_vector = char_vector.reshape(1, 1, *char_vector.shape)
    with torch.no_grad():
        model.hidden, model.cell = h
        output = model.forward(char_vector.to(model.device))
        h = (model.hidden, model.cell)
        probs = F.softmax(output, dim=1)

        # Sampling from a distribution to add randomness
        dist = torch.distributions.Categorical(probs)
        index = dist.sample().item()
        return index, h

def generate_chars(model, n_chars, prime='The'):
    global test_dataset
    model.eval()

    # Batch size: 1
    model.init_hidden(1)
    h = (model.hidden, model.cell)

    for char in [test_dataset.stoi[x] for x in prime]:
        _, h = predict_next_char(model, char, h)

    chars = []

    for i in range(n_chars):
        char, h = predict_next_char(model, char, h)
        chars.append(char)

    return prime + ''.join([test_dataset.itos[x] for x in chars])
```

Notice that in the `predict` function, we are not simply taking softmax and using the character with the highest probability as the output. This is to ensure non deterministic behaviour of our model. Otherwise it is possible that our model will predict the same text every single time given the same input. To avoid this, we add a bit of randomness to the predictions by using a categorical distribution from `torch.distributions.Categorical`. We map our output probabilities and sample from this distribution hence making it slightly random. Another important point is that we have to call `model.eval()` before making any predictions. This is to ensure that factors like dropout are not active during inference. 

Here is what the output looks like: 

```
Thenoud. That may advents and then were what he had realized him to with it his word really he had hoped to gravely.

The Girls were to
directly down as Moscow "and many name in Frenchsiatures: still tried
to go
to the man and took peasants with recembling officer replied.

Without expectated her.

"To last after it of them; if he asked Doroth, do what a words she wrote out. Might Russia, and how he knew us to the princess. See Pierre close, I. When she saw the God, as he had khow. Napoleon's beauty-embings
he
opposite if and down to cur, beamed his spoze, remained gazed at his long dipfushius fur rash,
went in land. The
staff so endarked her stage.

There, evidently sat people, paper the Emperor respect and had a quarter appear mighter
and huers from the twill freed large. Count Molming which was loved to very offended kind collar out, and herself that like Borodino unattended to Prince Vasili. "I must be done not become meeting his will, so. Ing a restraint from her resist up to a swerf and dispensation, Natasha, blostest gossip. These
everywhere he frew four same to their faces which drawn out himself, but Pierre never just always like her and heard I may cause the officers for that-sobling have meant not la in her producefre princess.

On the Sonya, was understood here they
had thought the princess, the yellow an one had understanding a moment--motionless a thousands
of a valand quarters. After the rubber word of how one interest less pulled on the bullet in the family.

The Man.
Petya spoke Pierre
in his hearing an angry plundings. The hands of Pierre's resoluting that
who would have so not having
offended, sides to
the men meeting Miss Rostov, had a suture was employed at the brought, and that reformed remorsing and provising as particularly finely and funday the weep of side, while he was the entristent fiolly in a schabaces were ridicules surely-electations to the clowar!" he said that had been
taken in
the
Russian French hamp lagerral Rather--what was not repu
```

These are the first 2000 characters after training for 40 epochs. It is actually quite fun to see the model progress. It starts out printing random characters. Slowly but surely it starts to understand the structure of the corpus. Eventually you see it reduce the number of spelling mistakes. While the text doesn't make any sense, let me put this in context. This is a model which has no understanding of the concept of language, let alone the english language. We have given it a pretty small corpus of data (with a tiny sequence length of 100) and all it has learnt is to predict the next character. I have been studying english for my entire life and I still make spelling msitakes every single day :p. I have trained this very simple and small model architecture for less than an hour and I must say, I'm pretty impressed with the results. 

If you want a link to the full code, you can find it on my [github](https://github.com/aniketSanap/CharacterLevelLanguageModel). If you have any doubts, concerns or suggestions, feel free to leave a comment. Have a nice day!