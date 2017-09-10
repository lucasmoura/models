#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

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
     OP_REQUIRES_OK(ctx, Init(ctx->env(), filename));
  }
  
  void Compute(OpKernelContext* ctx) override {
    ctx->set_output(0, vocab_words_);
  }

  private:
   Tensor vocab_words_;
   Tensor freq_;
   int64 corpus_size_;
   int32 window_size_;
   int32 vocab_size_;
   int32 min_count_;
   std::vector<int32> corpus_;
   std::unordered_map<int32, float> coocurrences_;

   typedef std::pair<string, int32> WordFreq;

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

    for (int32 i = 0; i < size; ++i) {
      center_word = corpus_[i];
      start_index = (i - window_size_) > 0 ? (i - window_size_): 0;
      end_index = (i + window_size_) > size - 1 ? size - 1 : (i + window_size_);
      std::cout<<end_index<<std::endl;
      std::cout<<(i + window_size_)<<std::endl;

      for (int32 j = start_index; j <= end_index; j++) {
        if (j == i) {
          continue;
        }
        std::cout<<corpus_[i]<<", "<<corpus_[j]<<std::endl;
        context_word = corpus_[j];
        index = static_cast<int32>(center_word * vocab_size_ + context_word);
        dist = (j - i) > 0? (j - i) : (j - i) * -1;
        int32 actual_value = coocurrences_[index];
        coocurrences_[index] = actual_value + 1.0;
      }

    }

    std::cout<<"Printing coocurrence matrix"<<std::endl;
    std::cout<<"Vocabulary size: "<<vocab_size_<<std::endl;
    std::cout<<"Printing corpus: "<<std::endl;

    for(std::size_t i = 0; i < corpus_.size(); i++)
    {
      if (i == corpus_.size() - 1)
        std::cout<<corpus_[i]<<std::endl;
      else
        std::cout<<corpus_[i]<<", ";
    }

    for (int32 i = 0; i < vocab_size_; i++)
    {
      for (int32 j = 0; j < vocab_size_; j++)
      {
        std::cout<<coocurrences_[i * vocab_size_ + j]<<" ";
      }

      std::cout<<std::endl;
    }

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

    return Status::OK();
   }
    
};

REGISTER_KERNEL_BUILDER(Name("GloveModel").Device(DEVICE_CPU), GloveModelOp);

}  // end of namespace
