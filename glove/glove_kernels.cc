#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

const int num_precalc_examples = 6;

namespace {

bool ScanWord(StringPiece * input, string *word) {
  str_util::RemoveLeadingWhitespace(input);
  StringPiece tmp;

  if (str_util::ConsumeNonWhitespace(input, &tmp)) {
    word->assign(tmp.data(), tmp.size());
    return true;
  } else {
    return false;
  }
}

}  // end namespace


class GloveModelOp : public OpKernel {
  public:
   explicit GloveModelOp(OpKernelConstruction* ctx)
       : OpKernel(ctx) {
     string filename;
     OP_REQUIRES_OK(ctx, ctx->GetAttr("filename", &filename));
     OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
     OP_REQUIRES_OK(ctx, ctx->GetAttr("min_count", &min_count_));
     OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
     OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));

     mutex_lock l(mu_);
     co_ocurrence_input_pos_ = 0;
     co_ocurrence_output_pos_ = 0;
     precalc_index_ = 0;
     current_epoch_ = 0;

     for (int i = 0; i < num_precalc_examples; i++) {
       NextExample(&precalc_examples_[i].input,
                   &precalc_examples_[i].label,
                   &precalc_examples_[i].ccount);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    Tensor examples(DT_INT32, TensorShape({batch_size_}));
    auto Texamples = examples.flat<int32>();
    Tensor labels(DT_INT32, TensorShape({batch_size_}));
    auto Tlabels = labels.flat<int32>();
    Tensor ccounts(DT_FLOAT, TensorShape({batch_size_}));
    auto Tccounts = ccounts.flat<float>();
    Tensor current_epoch(DT_INT32, TensorShape({}));
    {
        mutex_lock l(mu_);
        for (int i = 0; i < batch_size_; i++) {
          Texamples(i) = precalc_examples_[precalc_index_].input;
          Tlabels(i) = precalc_examples_[precalc_index_].label;
          Tccounts(i) = precalc_examples_[precalc_index_].ccount;

          //std::cout<<"Batch: "<<precalc_examples_[precalc_index_].input<<" "<<precalc_examples_[precalc_index_].label<<" "<<precalc_examples_[precalc_index_].ccount<<std::endl;

          precalc_index_++;

          if (precalc_index_ >= num_precalc_examples) {
            precalc_index_ = 0;

            for (int j = 0; j < num_precalc_examples; j++) {
                NextExample(&precalc_examples_[j].input,
                            &precalc_examples_[j].label,
                            &precalc_examples_[j].ccount);
            }
          }

        }
        current_epoch.scalar<int32>()() = current_epoch_;
    }
    ctx->set_output(0, vocab_words_);
    ctx->set_output(1, indices_);
    ctx->set_output(2, values_);
    ctx->set_output(3, examples);
    ctx->set_output(4, labels);
    ctx->set_output(5, ccounts);
    ctx->set_output(6, current_epoch);
  }

  private:
   struct Example {
     int32 input;
     int32 label;
     float ccount;
   };

   Tensor vocab_words_;
   Tensor freq_;
   Tensor indices_;
   Tensor values_;
   int64 corpus_size_;
   int32 window_size_;
   int32 vocab_size_;
   int32 min_count_;
   int32 batch_size_;
   std::vector<int32> corpus_;

   mutex mu_;
   int32 co_ocurrence_input_pos_ = 0 GUARDED_BY(mu_);
   int32 co_ocurrence_output_pos_ = 0 GUARDED_BY(mu_);
   int32 current_epoch_ GUARDED_BY(mu_);
   std::unordered_map<int32, float> coocurrences_ GUARDED_BY(mu_);
   int precalc_index_ = 0 GUARDED_BY(mu_);
   std::vector<Example> precalc_examples_ GUARDED_BY(mu_);

   typedef std::pair<string, int32> WordFreq;

   void NextExample(int32* input, int32* label, float* ccount) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    int32 index;

    if (co_ocurrence_input_pos_ == vocab_size_ - 1) {
        if(co_ocurrence_output_pos_ == vocab_size_ - 1) {
          co_ocurrence_input_pos_ = 0;
          co_ocurrence_output_pos_ = 0;
          current_epoch_++;
        }
    }

    if (co_ocurrence_output_pos_ >= vocab_size_) {
      co_ocurrence_output_pos_ = 0;
    }

    for (int32 i = co_ocurrence_input_pos_; i < vocab_size_; i++) {
      for (int32 j = co_ocurrence_output_pos_; j < vocab_size_; j++) {
        index = static_cast<int32>(i * vocab_size_ + j);

        if (coocurrences_[index] != 0) {
            *input = i;
            *label = j;
            *ccount = coocurrences_[index];

            co_ocurrence_input_pos_ = j + 1 == vocab_size_ ? i + 1: i;
            co_ocurrence_output_pos_ = j + 1;
            return;
        }
      }
    }

   }

   std::unordered_map<string, int32>
   CreateWordFrequencies(const string& data) {
    StringPiece input = data;

    string w;
    corpus_size_ = 0;
    std::unordered_map<string, int32> word_freq;
    while (ScanWord(&input, &w)) {
      ++(word_freq[w]);
      ++corpus_size_;
    }

    return word_freq;
   }

   std::vector<WordFreq>
   CreateVocabulary(std::unordered_map<string, int32> word_freq) {
    std::vector<WordFreq> ordered;
    for (const auto& p : word_freq) {
      if (p.second >= min_count_) ordered.push_back(p);
    }

    std::sort(ordered.begin(), ordered.end(),
              [](const WordFreq& x, const WordFreq& y) {
                return x.second > y.second;
              });
    vocab_size_ = static_cast<int32>(1 + ordered.size());
    return ordered;
   }

   std::unordered_map<string, int32>
   CreateWord2Index(std::vector<WordFreq> vocabulary) {
    std::unordered_map<string, int32> word_id;
    Tensor word(DT_STRING, TensorShape({vocab_size_}));
    Tensor freq(DT_INT32, TensorShape({vocab_size_}));
    word.flat<string>()(0) = "UNK";
    int64 total_counted = 0;

    for (std::size_t i = 0; i < vocabulary.size(); ++i) {
      const auto& w = vocabulary[i].first;
      auto id = i + 1;
      word.flat<string>()(id) = w;
      auto word_count = vocabulary[i].second;
      freq.flat<int32>()(id) = word_count;
      total_counted += word_count;
      word_id[w] = id;
    }

    freq.flat<int32>()(0) = corpus_size_ - total_counted;
    vocab_words_ = word;
    freq_ = freq;

    return word_id;
   }

   void CreateCorpus(const string& data,
                     std::unordered_map<string, int32> word_id) {

    static const int32 kUnkId = 0;
    StringPiece input = data;
    string w;

    corpus_.reserve(corpus_size_);
    while (ScanWord(&input, &w)) {
      corpus_.push_back(gtl::FindWithDefault(word_id, w, kUnkId));
    }

   }

   void CreateCoocurrences() {
    int32 center_word, context_word, start_index, end_index, dist, index;
    int32 size = static_cast<int32>(corpus_.size());

    typedef std::pair<int32, int32> CooccurIndices;
    std::vector<CooccurIndices> valid_indices;

    for (int32 i = 0; i < size; ++i) {
      center_word = corpus_[i];
      start_index = (i - window_size_) > 0 ? (i - window_size_): 0;
      end_index = (i + window_size_) > size - 1 ? size - 1 : (i + window_size_);

      for (int32 j = start_index; j <= end_index; j++) {
        if (j == i) {
          continue;
        }
        context_word = corpus_[j];
        index = static_cast<int32>(center_word * vocab_size_ + context_word);
        dist = (j - i) > 0? (j - i) : (j - i) * -1;
        int32 actual_value = coocurrences_[index];

        if (!actual_value) {
          valid_indices.push_back(CooccurIndices(corpus_[i], corpus_[j]));
        }

        coocurrences_[index] = actual_value + 1.0;
      }

    }

    int32 indices_size = static_cast<int32>(valid_indices.size());
    Tensor indices(DT_INT64, TensorShape({indices_size, 2}));
    Tensor values(DT_INT32, TensorShape({indices_size}));

    for(std::size_t i = 0; i<valid_indices.size(); i++) {
      center_word = valid_indices[i].first;
      context_word = valid_indices[i].second;

      indices.matrix<int64>()(i, 0) = center_word;
      indices.matrix<int64>()(i, 1) = context_word;
      values.flat<int32>()(i) = coocurrences_[center_word * vocab_size_ + context_word];
    }

    indices_ = indices;
    values_ = values;
   }

   Status Init(Env *env, const string& filename) {
    string data;
    TF_RETURN_IF_ERROR(ReadFileToString(env, filename, &data));
    auto word_freq = CreateWordFrequencies(data);

   // if (corpus_size_ < window_size_ * 10) {
   //   return errors::InvalidArgument("The text file ", filename,
   //                                  " contains too little data: ",
   //                                  corpus_size_, "words");
   // }

    auto ordered = CreateVocabulary(word_freq);
    word_freq.clear();
    auto word_id = CreateWord2Index(ordered);
    CreateCorpus(data, word_id);
    CreateCoocurrences();
    precalc_examples_.resize(num_precalc_examples);

    return Status::OK();
   }

};

REGISTER_KERNEL_BUILDER(Name("GloveModel").Device(DEVICE_CPU), GloveModelOp);

}  // end of namespace tensorflow
