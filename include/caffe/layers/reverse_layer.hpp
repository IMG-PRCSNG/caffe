#ifndef CAFFE_REVERSE_LAYER_HPP_
#define CAFFE_REVERSE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Permute the input blob by changing the memory order of the data.
 *
 * TODO(weiliu89): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Ftype, typename Btype>
class ReverseLayer : public Layer<Ftype, Btype> {
 public:
  explicit ReverseLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Reverse"; }
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
  ReverseParameter reverse_param_;
  TBlob<int> bottom_counts_;
  int axis_;
};

}  // namespace caffe

#endif  // CAFFE_REVERSE_LAYER_HPP_
