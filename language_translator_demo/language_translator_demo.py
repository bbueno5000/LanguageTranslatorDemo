"""
DOCSTRING
"""
import logging
import math
import nltk
import numpy
import os
import random
import re
import sys
import tensorflow
import time

tensorflow.app.flags.DEFINE_boolean(
    "use_fp16", False, "Train using fp16 instead of fp32.")
tensorflow.app.flags.DEFINE_float(
    "learning_rate", 0.5, "Learning rate.")
tensorflow.app.flags.DEFINE_float(
    "learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tensorflow.app.flags.DEFINE_float(
    "max_gradient_norm", 5.0, "Clip gradients to this norm.")
tensorflow.app.flags.DEFINE_integer(
    "batch_size", 32, "Batch size to use during training.")
tensorflow.app.flags.DEFINE_integer(
    "size", 512, "Size of each model layer.")
tensorflow.app.flags.DEFINE_integer(
    "num_layers", 2, "Number of layers in the model.")
tensorflow.app.flags.DEFINE_integer(
    "s_vocab_size", 30000, "Source language vocabulary size.")
tensorflow.app.flags.DEFINE_integer(
    "t_vocab_size", 30000, "Target language vocabulary size.")
tensorflow.app.flags.DEFINE_integer(
    "max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tensorflow.app.flags.DEFINE_integer(
    "steps_per_checkpoint", 200, "How many training steps to do per checkpoint.")
tensorflow.app.flags.DEFINE_string(
    "data_dir", "/Users/fancyshmancy/Development/nlp/proj2/data/", "Data directory")
tensorflow.app.flags.DEFINE_string(
    "train_dir", "/Users/fancyshmancy/Development/nlp/proj2/runs/de_en_lstm_reg/",
    "Training directory.")
FLAGS = tensorflow.app.flags.FLAGS

class DataUtils:
    """
    Utilities for downloading data from NMT, tokenizing, vocabularies.
    """
    def __init__(self, *args, **kwargs):
       # Special vocabulary symbols - we always put them at the start.
        self._PAD = b"_PAD"
        self._GO = b"_GO"
        self._EOS = b"_EOS"
        self._UNK = b"_UNK"
        self._START_VOCAB = [self._PAD, self._GO, self._EOS, self._UNK]
        self.PAD_ID = 0
        self.GO_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3
        # Regular expressions used to tokenize.
        self._WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
        self._DIGIT_RE = re.compile(br"\d")
        return super().__init__(*args, **kwargs)

    def basic_tokenizer(self, sentence):
        """
        Very basic tokenizer: split the sentence into a list of tokens.
        """
        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(self._WORD_SPLIT.split(space_separated_fragment))
        return [w for w in words if w]

    def create_vocabulary(
        self,
        vocabulary_path,
        data_path,
        max_vocabulary_size,
        tokenizer=None,
        normalize_digits=True):
        """
        Create vocabulary file (if it does not exist yet) from data file.
        Data file is assumed to contain one sentence per line. Each sentence is
        tokenized and digits are normalized (if normalize_digits is set).
        Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
        We write it to vocabulary_path in a one-token-per-line format, so that later
        token in the first line gets id=0, second line gets id=1, and so on.
    
        Args:
            vocabulary_path: path where the vocabulary will be created.
            data_path: data file that will be used to create vocabulary.
            max_vocabulary_size: limit on the size of the created vocabulary.
            tokenizer: a function to use to tokenize each data sentence;
                if None, basic_tokenizer will be used.
            normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        if not tensorflow.python.platform.gfile.Exists(vocabulary_path):
            print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
            vocab = {}
            with tensorflow.python.platform.gfile.GFile(data_path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    line = tensorflow.compat.as_bytes(line)
                    tokens = tokenizer(line) if tokenizer else self.basic_tokenizer(line)
                    for w in tokens:
                        word = self._DIGIT_RE.sub(b"0", w) if normalize_digits else w
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
                vocab_list = self._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
                if len(vocab_list) > max_vocabulary_size:
                    vocab_list = vocab_list[:max_vocabulary_size]
                with tensorflow.python.platform.gfile.GFile(
                    vocabulary_path, mode="wb") as vocab_file:
                    for w in vocab_list:
                        vocab_file.write(w + b"\n")

    def data_to_token_ids(
        self,
        data_path,
        target_path,
        vocabulary_path,
        tokenizer=None,
        normalize_digits=True):
        """
        Tokenize data file and turn into token-ids using given vocabulary file.
        This function loads data line-by-line from data_path, calls the above
        sentence_to_token_ids, and saves the result to target_path. See comment
        for sentence_to_token_ids on the details of token-ids format.
  
        Args:
            data_path: path to the data file in one-sentence-per-line format.
            target_path: path where the file with token-ids will be created.
            vocabulary_path: path to the vocabulary file.
            tokenizer: a function to use to tokenize each sentence;
                if None, basic_tokenizer will be used.
            normalize_digits: Boolean; if true, all digits are replaced by 0s.
        """
        if not tensorflow.python.platform.gfile.Exists(target_path):
            print("Tokenizing data in %s" % data_path)
            vocab, _ = self.initialize_vocabulary(vocabulary_path)
            with tensorflow.python.platform.gfile.GFile(
                data_path, mode="rb") as data_file:
                with tensorflow.python.platform.gfile.GFile(
                    target_path, mode="w") as tokens_file:
                    counter = 0
                    for line in data_file:
                        counter += 1
                        if counter % 100000 == 0:
                            print("  tokenizing line %d" % counter)
                        token_ids = self.sentence_to_token_ids(
                            tensorflow.compat.as_bytes(line),
                            vocab, tokenizer, normalize_digits)
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

    def initialize_vocabulary(self, vocabulary_path):
        """
        Initialize vocabulary from file.
        We assume the vocabulary is stored one-item-per-line, so a file:
            dog
            cat
        will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
        also return the reversed-vocabulary ["dog", "cat"].
  
        Args:
            vocabulary_path: path to the file containing the vocabulary.
  
        Returns:
            a pair: the vocabulary (a dictionary mapping string to integers), and
            the reversed vocabulary (a list, which reverses the vocabulary mapping).
    
        Raises:
            ValueError: if the provided vocabulary_path does not exist.
        """
        if tensorflow.python.platform.gfile.Exists(vocabulary_path):
            rev_vocab = []
            with tensorflow.python.platform.gfile.GFile(vocabulary_path, mode="rb") as f:
                rev_vocab.extend(f.readlines())
            rev_vocab = [line.strip() for line in rev_vocab]
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
            return vocab, rev_vocab
        else:
            raise ValueError("Vocabulary file %s not found.", vocabulary_path)

    def parse_files_to_lists(self, data_path, lang, xml):
        """
        DOCSTRING
        """
        if not xml:
            with tensorflow.python.platform.gfile.GFile(data_path+lang, mode="r") as f:
                texts = f.readlines()
                texts = (t for t in texts if "</" not in t)
        else:
            import xml.etree.ElementTree as ET
            filename = data_path + lang + '.xml'
            tree = ET.parse(filename)
            texts = (seg.text for seg in tree.iter('seg'))
        return texts

    def prepare_data(self, data_dir, s_vocabulary_size, t_vocabulary_size, source, target):
        """
        Get TED talk data from data_dir, create vocabularies and tokenize data.
  
        Args:
            data_dir: directory in which the data sets will be stored.
            ja_vocabulary_size: size of the Japanese vocabulary to create and use.
            en_vocabulary_size: size of the English vocabulary to create and use.
            tokenizer: a function to use to tokenize each data sentence;
                if None, basic_tokenizer will be used.
    
        Returns:
            A tuple of 6 elements:
                (1) path to the token-ids for Japanese training data-set,
                (2) path to the token-ids for English training data-set,
                (3) path to the token-ids for Japanese development data-set,
                (4) path to the token-ids for English development data-set,
                (5) path to the Japanese vocabulary file,
                (6) path to the English vocabulary file.
        """
        _data_dir = data_dir
        _train_path = os.path.join(_data_dir, 'train.')
        _dev_path = os.path.join(_data_dir, 'dev.')
        if not os.path.isfile(os.path.join(data_dir, "train."+source)):
            data_dir = os.path.join(data_dir, "%s-%s/" % (source, target))
            # Get nmt data to the specified directory.
            train_path = os.path.join(data_dir, "train.tags.%s-%s." % (source, target))
            dev_path = os.path.join(data_dir, "IWSLT15.TED.dev2010.%s-%s." % (source, target))
            # Parse xml files into lists of texts.
            s_texts_train = self.parse_files_to_lists(train_path, source, False)
            t_texts_train = self.parse_files_to_lists(train_path, target, False)
            s_texts_dev = self.parse_files_to_lists(dev_path, source, True)
            t_texts_dev = self.parse_files_to_lists(dev_path, target, True)
            # Write out training set and dev sets.
            with tensorflow.python.platform.gfile.GFile(_train_path+source, mode="w") as f:
                for line in s_texts_train:
                    f.write(line)
            with tensorflow.python.platform.gfile.GFile(_train_path+target, mode="w") as f:
                for line in t_texts_train:
                    f.write(line)
            with tensorflow.python.platform.gfile.GFile(_dev_path+source, mode="w") as f:
                for line in s_texts_dev:
                    f.write(line+"\n")
            with tensorflow.python.platform.gfile.GFile(_dev_path+target, mode="w") as f:
                for line in t_texts_dev:
                    f.write(line+"\n")
        # Create vocabularies of the appropriate sizes.
        s_vocab_path = os.path.join(_data_dir, "vocab%d.%s" % (s_vocabulary_size, source))
        t_vocab_path = os.path.join(_data_dir, "vocab%d.%s" % (t_vocabulary_size, target))
        self.create_vocabulary(s_vocab_path, _train_path + source, s_vocabulary_size)
        self.create_vocabulary(t_vocab_path, _train_path + target, t_vocabulary_size)
        # Create token ids for the training data.
        s_train_ids_path = _train_path + ("ids%d.%s" % (s_vocabulary_size, source))
        t_train_ids_path = _train_path + ("ids%d.%s" % (t_vocabulary_size, target))
        self.data_to_token_ids(_train_path + source, s_train_ids_path, s_vocab_path)
        self.data_to_token_ids(_train_path + target, t_train_ids_path, t_vocab_path)
        # Create token ids for the development data.
        s_dev_ids_path = _dev_path + ("ids%d.%s" % (s_vocabulary_size, source))
        t_dev_ids_path = _dev_path + ("ids%d.%s" % (t_vocabulary_size, target))
        self.data_to_token_ids(_dev_path + source, s_dev_ids_path, s_vocab_path)
        self.data_to_token_ids(_dev_path + target, t_dev_ids_path, t_vocab_path)
        return (s_train_ids_path,
                t_train_ids_path,
                s_dev_ids_path,
                t_dev_ids_path,
                s_vocab_path,
                t_vocab_path)

    def sentence_to_token_ids(
        self,
        sentence,
        vocabulary,
        tokenizer=None,
        normalize_digits=True):
        """
        Convert a string to list of integers representing token-ids.
        For example, a sentence "I have a dog" may become tokenized into
        ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
        "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
  
        Args:
            sentence: the sentence in bytes format to convert to token-ids.
            vocabulary: a dictionary mapping tokens to integers.
            tokenizer: a function to use to tokenize each sentence;
                if None, basic_tokenizer will be used.
            normalize_digits: Boolean; if true, all digits are replaced by 0s.
  
        Returns:
            a list of integers, the token-ids for the sentence.
        """
        if tokenizer:
            words = tokenizer(sentence)
        else:
            words = self. basic_tokenizer(sentence)
        if not normalize_digits:
            return [vocabulary.get(w, self.UNK_ID) for w in words]
        # Normalize digits by 0 before looking words up in the vocabulary.
        return [vocabulary.get(self._DIGIT_RE.sub(b"0", w), self.UNK_ID) for w in words]

class Seq2SeqModel:
    """
    Sequence-to-sequence model with attention and for multiple buckets.
    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
        http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
        http://arxiv.org/abs/1412.2007

    Copyright 2015 The TensorFlow Authors. All Rights Reserved.
    """
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        buckets,
        size,
        num_layers,
        max_gradient_norm,
        batch_size,
        learning_rate,
        learning_rate_decay_factor,
        attention=True,
        use_lstm=False, 
        num_samples=512,
        forward_only=False,
        dtype=tensorflow.float32):
        """
        Create the model.
    
        Args:
            source_vocab_size: size of the source vocabulary.
            target_vocab_size: size of the target vocabulary.
            buckets: a list of pairs (I, O), where I specifies maximum input length
                that will be processed in that bucket, and O specifies maximum output
                length. Training instances that have inputs longer than I or outputs
                longer than O will be pushed to the next bucket and padded accordingly.
                We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
            size: number of units in each layer of the model.
            num_layers: number of layers in the model.
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            learning_rate: learning rate to start with.
            learning_rate_decay_factor: decay learning rate by this much when needed.
            use_lstm: if true, we use LSTM cells instead of GRU cells.
            num_samples: number of samples for sampled softmax.
            forward_only: if set, we do not construct the backward pass in the model.
            dtype: the data type to use to store internal variables.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tensorflow.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.attention = attention
        self.global_step = tensorflow.Variable(0, trainable=False)
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tensorflow.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tensorflow.transpose(w_t)
            b = tensorflow.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)
            def sampled_loss(inputs, labels):
                labels = tensorflow.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tensorflow.cast(w_t, tensorflow.float32)
                local_b = tensorflow.cast(b, tensorflow.float32)
                local_inputs = tensorflow.cast(inputs, tensorflow.float32)
                return tensorflow.cast(
                    tensorflow.nn.sampled_softmax_loss(
                        local_w_t, local_b, local_inputs, labels,
                        num_samples, self.target_vocab_size), dtype)
            softmax_loss_function = sampled_loss
        # Create the internal multi-layer cell for our RNN.
        single_cell = tensorflow.nn.rnn_cell.GRUCell(size)
        if use_lstm:
            single_cell = tensorflow.nn.rnn_cell.LSTMCell(size)
        single_cell = tensorflow.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=0.75)
        cell = single_cell
        if num_layers > 1:
            cell = tensorflow.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode, attention):
            if attention:
                return tensorflow.nn.seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=size,
                    output_projection=output_projection,
                    feed_previous=do_decode, dtype=dtype)
            return tensorflow.nn.seq2seq.embedding_rnn_seq2seq(
                        encoder_inputs,
                        decoder_inputs,
                        cell,
                        num_encoder_symbols=source_vocab_size,
                        num_decoder_symbols=target_vocab_size,
                        embedding_size=size,
                        output_projection=output_projection,
                        feed_previous=do_decode,
                        dtype=dtype)
        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tensorflow.placeholder(
                tensorflow.int32, shape=[None], name="encoder{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tensorflow.placeholder(
                tensorflow.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tensorflow.placeholder(
                dtype, shape=[None], name="weight{0}".format(i)))
        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in range(len(self.decoder_inputs) - 1)]
        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tensorflow.nn.seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, buckets,
                    lambda x, y: seq2seq_f(x, y, True, self.attention),
                    softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [tensorflow.matmul(output, output_projection[0]) +
                                       output_projection[1] for output in self.outputs[b]]
        else:
            self.outputs, self.losses = tensorflow.nn.seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, buckets,
                    lambda x, y: seq2seq_f(x, y, False, self.attention),
                    softmax_loss_function=softmax_loss_function)
        # Gradients and SGD update operation for training the model.
        params = tensorflow.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tensorflow.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                gradients = tensorflow.gradients(self.losses[b], params)
                clipped_gradients, norm = tensorflow.clip_by_global_norm(
                    gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, params), global_step=self.global_step))
        self.saver = tensorflow.train.Saver(tensorflow.all_variables())

    def step(
        self,
        session,
        encoder_inputs,
        decoder_inputs,
        target_weights,
        bucket_id,
        forward_only):
        """
        Run a step of the model feeding the given inputs.
    
        Args:
            session: tensorflow session to use.
            encoder_inputs: list of numpy int vectors to feed as encoder inputs.
            decoder_inputs: list of numpy int vectors to feed as decoder inputs.
            target_weights: list of numpy float vectors to feed as target weights.
            bucket_id: which bucket of the model to use.
            forward_only: whether to do the backward step or only forward.
    
        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.
    
        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket," 
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))
        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = numpy.zeros([self.batch_size], dtype=numpy.int32)
        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id], # Update Op that does SGD.
                           self.gradient_norms[bucket_id], # Gradient norm.
                           self.losses[bucket_id]] # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]] # Loss for this batch.
            for l in range(decoder_size): # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:] # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id):
        """
        Get a random batch of data from the specified bucket, prepare for step.
        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.
    
        Args:
            data: a tuple of size len(self.buckets) in which each element contains
                lists of pairs of input and output data that we use to create a batch.
            bucket_id: integer, which bucket to get the batch for.
        
        Returns:
            The triple (encoder_inputs, decoder_inputs, target_weights) for
            the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])
            # Encoder inputs are padded and then reversed.
            encoder_pad = [tensorflow.models.rnn.translate.data_utils.PAD_ID] \
            * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([tensorflow.models.rnn.translate.data_utils.GO_ID]
                                  + decoder_input
                                  + [tensorflow.models.rnn.translate.data_utils.PAD_ID]
                                  * decoder_pad_size)
        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                    numpy.array([encoder_inputs[batch_idx][length_idx]
                              for batch_idx in range(self.batch_size)], dtype=numpy.int32))
        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                    numpy.array([decoder_inputs[batch_idx][length_idx]
                              for batch_idx in range(self.batch_size)], dtype=numpy.int32))
            # Create target_weights to be 0 for targets that are padding.
            batch_weight = numpy.ones(self.batch_size, dtype=numpy.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 \
                or target == tensorflow.models.rnn.translate.data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

class Translate():
    """
    DOCSTRING
    """
    def __init__(self, *args, **kwargs):
        # We use a number of buckets and pad to the closest one for efficiency.
        # See seq2seq_model.Seq2SeqModel for details of how they work.
        self._buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
        return super().__init__(*args, **kwargs)

    def create_model(self, session, use_lstm, forward_only):
        """
        Create translation model and initialize or load parameters in session.
        """
        dtype = tensorflow.float16 if FLAGS.use_fp16 else tensorflow.float32
        model = seq2seq_model.Seq2SeqModel(
            FLAGS.s_vocab_size, FLAGS.t_vocab_size, self._buckets, FLAGS.size,
            FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
            FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
            use_lstm=use_lstm, forward_only=forward_only, dtype=dtype)
        ckpt = tensorflow.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tensorflow.initialize_all_variables())
        return model

    def read_data(self, source_path, target_path, max_size=None):
        """
        Read data from source and target files and put into buckets.

        Args:
        source_path: path to the files with token-ids for the source language.
        target_path: path to the file with token-ids for the target language;
            it must be aligned with the source file: n-th line contains the desired
            output for n-th line from the source_path.
        max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

        Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
        """
        data_set = [[] for _ in self._buckets]
        with tensorflow.gfile.GFile(source_path, mode="r") as source_file:
            with tensorflow.gfile.GFile(target_path, mode="r") as target_file:
                source, target = source_file.readline(), target_file.readline()
                counter = 0
                while source and target and (not max_size or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    source_ids = [int(x) for x in source.split()]
                    target_ids = [int(x) for x in target.split()]
                    target_ids.append(DataUtils.EOS_ID)
                    for bucket_id, (source_size, target_size) in enumerate(self._buckets):
                        if len(source_ids) < source_size and len(target_ids) < target_size:
                            data_set[bucket_id].append([source_ids, target_ids])
                            break
                    source, target = source_file.readline(), target_file.readline()
        return data_set

    def testBLEU(self):
        """
        DOCSTRING
        """
        source = sys.argv[1]
        target = sys.argv[2]
        with tensorflow.Session() as sess:
            # Create model and load parameters.
            model = self.create_model(sess, True, True)
            model.batch_size = 1  # We decode one sentence at a time.
            # Load vocabularies.
            s_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.%s" %
                                        (FLAGS.s_vocab_size, source))
            t_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.%s" %
                                        (FLAGS.t_vocab_size, target))
            s_vocab, _ = DataUtils.initialize_vocabulary(s_vocab_path)
            _, rev_t_vocab = DataUtils.initialize_vocabulary(t_vocab_path)
            # Decode from standard input.
            BLEUscore = {0:[], 1:[], 2:[], 3:[]}
            s_test_path = os.path.join(FLAGS.data_dir, "test.%s" % source)
            t_test_path = os.path.join(FLAGS.data_dir, "test.%s" % target)
            f_s = open(s_test_path, 'r')
            f_t = open(t_test_path, 'r')
            #print(f_s.readline())
            step = 0
            for sentence in f_s:
                print(step)
                #sentence = f_ja.readline()
                # Get token-ids for the input sentence.
                token_ids = DataUtils.sentence_to_token_ids(
                    tensorflow.compat.as_bytes(sentence), s_vocab)
                # Which bucket does it belong to?
                bucket_id = len(self._buckets) - 1
                for i, bucket in enumerate(self._buckets):
                    if bucket[0] >= len(token_ids):
                        bucket_id = i
                        break
                else:
                    logging.warning("Sentence truncated: %s", sentence) 
                # Get a 1-element batch to feed the sentence to the model.
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(token_ids, [])]}, bucket_id)
                # Get output logits for the sentence.
                _, _, output_logits = model.step(
                    sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                # This is a greedy decoder - outputs are just argmaxes of output_logits.
                outputs = [int(numpy.argmax(logit, axis=1)) for logit in output_logits]
                # If there is an EOS symbol in outputs, cut them at that point.
                if DataUtils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(DataUtils.EOS_ID)]
                # Print out Japanese sentence corresponding to outputs.
                candidate = [tensorflow.compat.as_str(rev_t_vocab[output]) for output in outputs]
                reference = f_t.readline().split(' ')
                try:
                    temp_score = nltk.translate.bleu_score.sentence_bleu(
                        [reference], candidate)
                except:
                    temp_score = nltk.translate.bleu_score.sentence_bleu(
                        [reference], candidate, weights=(.5, .5))
                BLEUscore[bucket_id].append(temp_score)
                step += 1
                print(temp_score)
            for key,val in BLEUscore.iteritems():
                print(key, ": ", numpy.mean(val))
            #print(numpy.mean(BLEUscore))

    def train(self):
        """
        Train a translation model using NMT data.
        """
        source = sys.argv[1]
        target = sys.argv[2]
        # Prepare NMT data.
        print("Preparing NMT data in %s" % FLAGS.data_dir)
        print("    source language: %s" % source)
        print("    target language: %s" % target)
        s_train, t_train, s_dev, t_dev, _, _, _, _ = DataUtils.prepare_data(
            FLAGS.data_dir, FLAGS.s_vocab_size, FLAGS.t_vocab_size, source, target)
        with tensorflow.Session() as sess:
            # Create model.
            print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
            model = self.create_model(sess, False, False)
            # Read data into buckets and compute their sizes.
            print("Reading development and training data (limit: %d)." 
                  % FLAGS.max_train_data_size)
            dev_set = self.read_data(s_dev, t_dev)
            train_set = self.read_data(s_train, t_train, FLAGS.max_train_data_size)
            train_bucket_sizes = [len(train_set[b]) for b in range(len(self._buckets))]
            train_total_size = float(sum(train_bucket_sizes))
            # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
            # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
            # the size if i-th training bucket, as used later.
            train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) /
                                   train_total_size for i in range(len(train_bucket_sizes))]
            # This is the training loop.
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            perplexity = 1e10
            train_steps, train_ppx, bucket_ppx = [], [], {0:[], 1:[], 2:[], 3:[]}
            while perplexity>20.:
                # Choose a bucket according to data distribution. We pick a random number
                # in [0, 1] and use the corresponding interval in train_buckets_scale.
                random_number_01 = numpy.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale)) if
                                 train_buckets_scale[i] > random_number_01])
                # Get a batch and make a step.
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    train_set, bucket_id)
                _, step_loss, _ = model.step(
                    sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1
                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    train_steps.append(current_step)
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    train_ppx.append(perplexity)
                    print("global step %d learning rate %.4f step-time %.2f perplexity "
                          "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                    step_time, perplexity))
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)
                    # Save checkpoint and zero timer and loss.
                    checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    step_time, loss, eval_loss_tot = 0.0, 0.0, 0.0
                    # Run evals on development set and print their perplexity.
                    for bucket_id in range(len(self._buckets)):
                        if len(dev_set[bucket_id]) == 0:
                            print("  eval: empty bucket %d" % (bucket_id))
                            continue
                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                            dev_set, bucket_id)
                        _, eval_loss, _ = model.step(
                            sess, encoder_inputs, decoder_inputs,
                            target_weights, bucket_id, True)
                        eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float("inf")
                        bucket_ppx[bucket_id].append(eval_ppx)
                        print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                        eval_loss_tot += eval_loss
                    eval_loss_avg = eval_loss_tot / len(self._buckets)
                    eval_ppx = math.exp(float(eval_loss_avg)) if eval_loss < 300 else float("inf")
                    print("  eval: mean perplexity %.2f" % eval_ppx)
                    sys.stdout.flush()
            print(train_steps)
            print(train_ppx)
            print(bucket_ppx)



def main(_):
    self.train()
    self.testBLEU()

if __name__ == '__main__':
    tensorflow.app.run()
