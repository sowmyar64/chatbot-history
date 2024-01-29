{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c4c40958",
   "metadata": {},
   "outputs": [],
   "source": [
    "import io\n",
    "import random\n",
    "import string # to process standard python strings\n",
    "import warnings\n",
    "import numpy as np\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "532fee49",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: nltk in d:\\anaconda\\lib\\site-packages (3.6.5)\n",
      "Requirement already satisfied: click in d:\\anaconda\\lib\\site-packages (from nltk) (8.0.3)\n",
      "Requirement already satisfied: joblib in d:\\anaconda\\lib\\site-packages (from nltk) (1.1.0)\n",
      "Requirement already satisfied: regex>=2021.8.3 in d:\\anaconda\\lib\\site-packages (from nltk) (2021.8.3)\n",
      "Requirement already satisfied: tqdm in d:\\anaconda\\lib\\site-packages (from nltk) (4.62.3)\n",
      "Requirement already satisfied: colorama in d:\\anaconda\\lib\\site-packages (from click->nltk) (0.4.4)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install nltk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "05e5013c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import nltk\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "nltk.download('popular', quiet=True) #downloadsNama packages\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "de2f5019",
   "metadata": {},
   "outputs": [],
   "source": [
    "f=open('C:\\\\Users\\\\vinay\\\\Downloads\\\\History.txt','r',errors = 'ignore')\n",
    "raw=f.read()\n",
    "raw = raw.lower() #converts to lowercase to reduce repetition of  words like The and the or When and when"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "42e8a745",
   "metadata": {},
   "outputs": [],
   "source": [
    "sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences\n",
    "word_tokens = nltk.word_tokenize(raw)# converts to list of words\n",
    "lemmer = nltk.stem.WordNetLemmatizer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a006d51f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def LemTokens(tokens):\n",
    "    return [lemmer.lemmatize(token) for token in tokens]\n",
    "\n",
    "\n",
    "remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "d3c81e46",
   "metadata": {},
   "outputs": [],
   "source": [
    "def LemNormalize(text):\n",
    "    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "67c86fed",
   "metadata": {},
   "outputs": [],
   "source": [
    "def response(user_response):\n",
    "    robot_response=''\n",
    "    sent_tokens.append(user_response)\n",
    "    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')\n",
    "    tfidf = TfidfVec.fit_transform(sent_tokens)\n",
    "    vals = cosine_similarity(tfidf[-1], tfidf)\n",
    "    idx=vals.argsort()[0][-2]\n",
    "    flat = vals.flatten()\n",
    "    flat.sort()\n",
    "    req_tfidf = flat[-2]\n",
    "    if(req_tfidf==0):\n",
    "        robot_response=robot_response+\"I think I need to read more about that...\"\n",
    "        return robot_response\n",
    "    else:\n",
    "        robot_response = robot_response+sent_tokens[idx]\n",
    "        return robot_response\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "884af297",
   "metadata": {},
   "outputs": [],
   "source": [
    "GREETING_INPUTS = (\"namastey\", \"namaskaram\", \"hello\", \"hi\", \"whats up\", \"hey\")\n",
    "GREETING_RESPONSES = [\"namastey\", \"namaskaram\", \"hello\", \"hi\", \"whats up\", \"hey\"]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "db43be40",
   "metadata": {},
   "outputs": [],
   "source": [
    "def greeting(sentence):\n",
    "    for word in sentence.split():\n",
    "        if word.lower() in GREETING_INPUTS:\n",
    "            return random.choice(GREETING_RESPONSES)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ba6bb39",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Spongebot: Namaskaram I am Spongebot. I am an expert on history of India, you can ask me anything. Go ahead!\n",
      "tell me about india history\n",
      "Spongebot: indian cultural influence (greater india)\n",
      "\n",
      "timeline of indian history.\n",
      "tell me about time line of indian history\n",
      "Spongebot: indian cultural influence (greater india)\n",
      "\n",
      "timeline of indian history.\n",
      "bye\n",
      "Spongebot: I think I need to read more about that...\n"
     ]
    }
   ],
   "source": [
    "flag=True\n",
    "print(\"Spongebot: Namaskaram I am Spongebot. I am an expert on history of India, you can ask me anything. Go ahead!\")\n",
    "while(flag==True):\n",
    "    user_response = input()\n",
    "    user_response=user_response.lower()\n",
    "    if(user_response!='bye!!'):\n",
    "        if(user_response=='thanks' or user_response=='thank you' ):\n",
    "            flag=False\n",
    "            print(\"Spongebot: Anytime\")\n",
    "        else:\n",
    "            if(greeting(user_response)!=None):\n",
    "                print(\"Spongebot: \"+greeting(user_response))\n",
    "            else:\n",
    "                print(\"Spongebot: \",end=\"\")\n",
    "                print(response(user_response))\n",
    "                sent_tokens.remove(user_response)\n",
    "    else:\n",
    "        flag=False\n",
    "        print(\"Spongebot: take care..\")\n",
    "##code ends here"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7decc8b0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}