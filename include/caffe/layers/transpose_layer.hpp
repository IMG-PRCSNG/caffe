#ifndef CAFFE_TRANSPOSE_LAYER_HPP_
#define CAFFE_TRANSPOSE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Ftype, typename Btype>
class TransposeLayer : public Layer<Ftype, Btype> {
 public:
  explicit TransposeLayer(const LayerParameter& param)
      : Layer<Ftype,Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Transpose"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
      NOT_IMPLEMENTED;
  }
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

 private:
  TransposeParameter transpose_param_;
  vector<int> permute(const vector<int>& vec);
  TBlob<int> bottom_counts_;
  TBlob<int> top_counts_;
  TBlob<int> forward_map_;
  TBlob<int> backward_map_;
  TBlob<int> buf_;
};

} // namespace caffe

#endif /* ifndef CAFFE_TRANSPOSE_LAYER_HPP_ */
