#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("GloveModel")
    .Output("vocab_words: string")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("window_size: int = 5")
    .Attr("min_count: int = 0")
    .Doc(R"doc(
Parses a text file and creates the coocurrence matrix and batches
of examples necessary to train a GloVe model.

vocab_words: A vector of words in the corpus.
)doc");

} // end namespace tensorflow
