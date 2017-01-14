##Natual Language Processing and Word Representation
####embedding 
 tensorlayer.nlp.generate_skip_gram_batch(data, batch_size, num_skips, skip_window, data_index=0)
	generate a training batch for the skip-gram model
parameters:
	data: a list
		context
	batch_size: int
		batch size to return
	num_skips: int
		times to reuse an input to generate a label
	skip_window: int
		skip window, words to consider left and right
	data_index: int
		index of the context location
return:
	batch: a list 
		input
	labels: a list
		labels
	data_index: int
		index of the context location

####simple sampling
 tensorlayer.nlp.sample(a=[], temperature=1.0)
	sample an index from a probability array.
parameters:
	a: a list
		list of probabilities
	temperature: float or Non
		the higher the more uniform
		if none, it will be np.argmax(a)

####Vector representations
class tensorlayer.nlp.SimpleVocabulary(vocab, unk_id)
	simple vocabulary wrapper
parameters:
	vocab: a dictionary of word to word_id
	unk_id: id of the "unknown word"
method:
	word_to_id(word) 返回一个词的id


class tensorlayer.nlp.Vocabulary(vocab_file, start_word='<S>', end_word='</S>', unk_word='<UNK>')
	create vocabulary class from a given vocabulary and its id-word, word-id convert
parameters:
	vocab_file: file containing the vocabulary, where the words are the first whitespace-separated token on each line and word ids are the corresponding line numbers.
	start_word: special word denoting sentence start 句子开始标志词
	end_word: 句子结尾词
	unk_word: unknown 词
method:
	id_to_word(word_id) 返回word_id所对应的词 
	word_to_id(word) 返回一个词的id


####process sentence
 tensorlayer.nlp.process_sentence(sentence, start_word='<S>', end_word='</S>')
	将句子转化为一串词
	return: a list of strings

####create vocabulary
 tensorlayer.nlp.create_vocab(sentences, word_count_output_file, min_word_count=1)
	create the vocabulary of word to word_id
parameters:
	sentences: a list of lists of strings
	word_counts_output_file: a string
		file name
	min_word_count: a int 
return: tl.nlp.SimpleVocabulary object

####Read words from file
 tensorlayer.nlp.simple_read_words(filename='nietzsche.txt')
	read context from file without any preprocessing
	return the context in a string

 tensorlayer.nlp.read_words(filename='nietzsche.txt', replace=['\n','<eos>'])
	read a file and convert to list format context


####build vocabulary, word dictionary, word tokenization
 tensorlayer.nlp.build_vocab(data)
	build vocabulary, giventhe context in list format.
	return the vocabulary, which is a dictionary for word to id.

 tensorlayer.nlp.build_reverse_dictionary(word_to_id)
	given a dictionary for converting word to integer id. 
	return a reverse dictionary for converting a id to word.

 tensorlayer.nlp.build_words_dataset(words=[], vocabulary_size=50000, printable=True, unk_key='UNK')
	build the words dictionary and replace rare words with 'UNK' token. The most common word has the smallest integer id.
parameters:
	words: a list of string or byte
	vocabulary_size: int
		the maximum vocabulary size, limiting the vocabulary size. the the script replaces rare words with 'UNK' token
	printable: boolean
		whether to print the read vocabulary size of the given words.
	unk_key: a string
return:
	data: a lsit of integer
	count: a list of tuple and list
	dictionary: a dictionary
		word_to_id, mapping word tp unique ids
	reverse_dictionary: a dictionary
		id_to_word, mapping id to unique word.

####function for translation
word tokenization:
 tensorlayer.nlp.basic_tokenizer(sentence, _WORD_SPLIT=<_sre.SRE_Pattern object>)
	基本分词，_WORD_SPLIT: 正则表达式用于分词

 tensorlayer.nlp.create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, tokenizer=None, normalize_digits=True, _DIGIT_RE=<_sre.SRE_Pattern object>, _START_VOCAB=['_PAD', '_GO', '_EOS','_UNK'])
	create vocabulary file from data file.
parameters:
	vocabulary_path: path where the vocabulary will be created.词表存放路径
	data_path: data file that will be used to create vocabulary. 数据文件路径，文件每行一个句子
	max_vocabulary_size: limit on the size of the created vocabulary.
	tokenizer: a function to use to tokenize each data sentence.如果为空，则使用basic_tokenizer
	normalize_digits: boolean
		if true, all digits are replaced by 0s.

 tensorlayer.nlp.initialize_vocabulary(vocabulary_path)
	initialize vocabulary from file, return the word_to_id(dictionary) and id_to_word(list).

 tensorlayer.nlp.sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True, UNK_ID=3, _DIGIT_RE=<_sre.SRE_Pattern object>)
	convert a string to list of integers
	
 tensorlayer.nlp.data_to_token_ids(data_path, target_path, vocabulary_path, tokennizer=None, normalize_digits=True, UNK_ID=3, _DIGIT_RE=<_sre.SRE_Pattern object>)
	tokenize data file and turn into token-ids using given vocabulary file.
	

